"""
Semi-supervised VAE (SS-VAE) on dm-dt maps — the generative entry of the benchmark.
Input: the 4 coarse dm-dt maps of features.py (all bands, g, r, i; 5 dt rows x 17 dm
bins, each row a distribution over dm) and TAB_COLUMNS (period, LS power, colours,
r amplitude; no Gaia quantity). The maps need no period: the LS period is wrong for
69% of labelled RR Lyrae (vs Gaia DR3), and a VAE on the phase fold reached RR
AUC-PR 0.17 on held-out fields while trees on its own tabular inputs reached 0.49.

Per outer fold the VAE is retrained from scratch on the training fields only: their
labelled objects (SupCon on the original fine-grained labels, classes with >=
C.MIN_CLASS_SIZE members) and the unlabelled objects outside the test and validation
fields. A head on the latent mean then gives the 5 class probabilities: a logistic
regression here (what a linear model can read from the latent), gradient-boosted
trees in vae_hgb.py (what the latent holds), both with the parameters tuned here.
generative=False drops the ELBO, hence the decoder and the unlabelled objects: the same
encoder, SupCon and head, purely discriminative (mlp_dmdt.py).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
import features
from bench import degrade

NAME = "vae"
HEAD = "linear"
USES_VAL = False
EPOCHS, BATCH, HIDDEN = 100, 256, 256
BETA_WARM, LAMBDA_WARM = 0.25, 0.4   # fractions of EPOCHS: KL warm-up, then SupCon's
N_ROWS, N_DM = len(C.DMDT_MAPS) * len(C.DMDT_DT_GROUPS), len(C.DMDT_DM_EDGES) - 1
TAB_DIM = len(degrade.TAB_COLUMNS) + 1
_IN = None
DEFAULT = {"latent_dim": 32, "lr": 1e-3, "beta_max": 1.0, "lambda_max": 10.0, "temp": 0.07,
           "C": 1.0}   # the settings of the first SS-VAE


# ── Model ─────────────────────────────────────────────────────────────────────

class SSVAE(nn.Module):
    """
    MLP encoder / decoder. The encoder also reads which dt rows were empty (never
    sampled at that time scale): an empty row is filled with the population mean, so
    without the mask it would look like a typical star. The decoder gives logits per
    dt row (a distribution over dm) and, from its own head, the tabular vector —
    without that head nothing forces period and colours through to the latent.
    """

    def __init__(self, latent_dim, tab_dim):
        super().__init__()
        n_in = N_ROWS * N_DM + N_ROWS + tab_dim
        self.enc = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.GELU())
        self.fc_mu = nn.Linear(HIDDEN, latent_dim)
        self.fc_lv = nn.Linear(HIDDEN, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, N_ROWS * N_DM))
        self.dec_tab = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(),
                                     nn.Linear(64, tab_dim))

    def encode(self, x, mask, tab):
        h = self.enc(torch.cat([x.flatten(1), mask, tab], 1))
        return self.fc_mu(h), self.fc_lv(h)

    def forward(self, x, mask, tab):
        mu, logvar = self.encode(x, mask, tab)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(z).view(-1, N_ROWS, N_DM), self.dec_tab(z), mu, logvar


# ── Losses ────────────────────────────────────────────────────────────────────

def vae_loss(logits, tab_recon, x, mask, tab, mu, logvar, beta):
    """
    Masked ELBO. Each sampled dt row is a distribution over dm, reconstructed by
    cross-entropy; empty rows are not fitted. The reconstruction is summed over rows,
    not averaged (averaging would silently rescale beta), with each object normalised
    by its own number of sampled rows so that sparse curves weigh like dense ones.
    """
    ce = -(x * F.log_softmax(logits, -1)).sum(-1)                      # (B, N_ROWS)
    per_obj = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)
    recon = (per_obj * N_ROWS).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).clamp(min=C.FREE_BITS).sum(1).mean()
    tab_loss = ((tab_recon - tab) ** 2).sum(1).mean()
    return recon + C.TAB_RECON_WEIGHT * tab_loss + beta * kl


def supcon_loss(features, labels, temp):
    """
    Supervised contrastive loss (Khosla et al. 2020) on L2-normalised features.

    Rows labelled -1 (unlabelled, or a class excluded from SupCon) are dropped: they
    must not pair with each other, and treating "unknown" as a class would pull
    genuinely different objects together.
    """
    keep = labels >= 0
    if keep.sum() < 2:
        return torch.tensor(0.0, device=features.device)
    features, labels = features[keep], labels[keep]
    n = features.size(0)
    if n < 2:
        return torch.tensor(0.0, device=features.device)
    sim = torch.mm(features, features.T).clamp(-1.0, 1.0) / temp
    eye = torch.eye(n, device=features.device, dtype=torch.bool)
    sim.masked_fill_(eye, float("-inf"))
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    has_pos = pos_mask.any(1)
    if not has_pos.any():
        return torch.tensor(0.0, device=features.device)
    n_pos = pos_mask.float().sum(1).clamp(min=1)
    log_prob = F.log_softmax(sim, dim=1)
    loss = -(log_prob.masked_fill(~pos_mask, 0.0)).sum(1) / n_pos
    return loss[has_pos].mean()


# ── Benchmark interface ───────────────────────────────────────────────────────

def space(trial):
    return {"latent_dim": trial.suggest_categorical("latent_dim", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "beta_max": trial.suggest_float("beta_max", 0.1, 1.0, log=True),
            "lambda_max": trial.suggest_float("lambda_max", 1.0, 30.0, log=True),
            "temp": trial.suggest_categorical("temp", [0.05, 0.07, 0.1, 0.2]),
            "C": trial.suggest_float("C", 1e-2, 10.0, log=True)}


def _maps(x):
    """dm-dt block of feature rows → (maps (n, N_ROWS, N_DM), sampled-row mask (n, N_ROWS))."""
    dm_mean = np.load(C.FEATURES_NPZ)["dmdt_mean"].reshape(N_ROWS, N_DM)
    maps = x[:, -N_ROWS * N_DM:].reshape(-1, N_ROWS, N_DM).astype(np.float32)
    # empty rows were filled with the population mean by features.py
    return maps, (~np.all(maps == dm_mean, axis=2)).astype(np.float32)


def _inputs(d):
    """(maps, mask, raw tab) of every object, in d's order."""
    global _IN
    if _IN is None:
        assert d.names[-N_ROWS * N_DM:] == features.dmdt_names()   # maps x dt row x dm bin
        per = pd.read_csv(C.PERIODS_CSV, dtype={"diaObjectId": str}).set_index("diaObjectId")
        _IN = (*_maps(d.x), per.reindex(d.ids)[degrade.TAB_COLUMNS].to_numpy(np.float64))
    return _IN


def _tab(raw, stats=None):
    """log period, z-scored, NaN → 0, plus the fraction of colours/amplitude missing."""
    t = raw.copy()
    t[:, 0] = np.log10(np.clip(t[:, 0], 1e-3, None))
    if stats is None:
        stats = np.nanmean(t, 0), np.nanstd(t, 0) + 1e-6
    z = np.nan_to_num((t - stats[0]) / stats[1])
    return np.hstack([z, np.isnan(t[:, 2:]).mean(1, keepdims=True)]).astype(np.float32), stats


def _t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(dev)


class _Model:
    def __init__(self, vae, stats, n_cls):
        self.vae, self.stats, self.n_cls = vae, stats, n_cls
        self.scaler = self.head = None

    @torch.no_grad()
    def latent(self, maps, mask, raw_tab):
        self.vae.eval()
        dev = next(self.vae.parameters()).device
        x, m, t = _t(maps, dev), _t(mask, dev), _t(_tab(raw_tab, self.stats)[0], dev)
        return np.vstack([self.vae.encode(x[i:i + 4096], m[i:i + 4096], t[i:i + 4096])[0]
                          .cpu().numpy() for i in range(0, len(x), 4096)])

    def proba(self, maps, mask, raw_tab):
        out = np.zeros((len(maps), self.n_cls))
        z = self.scaler.transform(self.latent(maps, mask, raw_tab))
        out[:, self.head.classes_] = self.head.predict_proba(z)
        return out


def train(d, tr, va, params, seed, head=None, generative=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps, mask, raw = _inputs(d)
    test_fold = sorted(set(range(C.N_FIELD_FOLDS)) - set(d.folds[tr]) - set(d.folds[va]))
    unl = np.where(~d.lab & ~np.isin(d.folds, test_fold)
                   & ~np.isin(d.fields, np.unique(d.fields[va])))[0]
    rows = np.concatenate([tr, unl]) if generative else tr

    # SupCon labels: fine-grained labels of the training rows, large enough classes
    names, counts = np.unique(d.labels[tr].astype(str), return_counts=True)
    keep = {n: i for i, n in enumerate(names[counts >= C.MIN_CLASS_SIZE])}
    sc_y = np.array([keep.get(str(l), -1) for l in d.labels[tr]] + [-1] * (len(rows) - len(tr)))

    tab, stats = _tab(raw[rows])
    x, m, t = _t(maps[rows], dev), _t(mask[rows], dev), _t(tab, dev)
    sc_y = torch.from_numpy(sc_y).to(dev)
    vae = SSVAE(params["latent_dim"], TAB_DIM).to(dev)
    opt = torch.optim.AdamW(vae.parameters(), lr=params["lr"], weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    b_end, l_end = BETA_WARM * EPOCHS, (BETA_WARM + LAMBDA_WARM) * EPOCHS
    for ep in range(1, EPOCHS + 1):
        beta = params.get("beta_max", 0.0) * min(1.0, ep / b_end)
        lam = params.get("lambda_max", 0.0) * float(np.clip((ep - b_end) / (l_end - b_end), 0, 1))
        vae.train()
        for b in np.array_split(rng.permutation(len(rows)), max(1, len(rows) // BATCH)):
            b = torch.from_numpy(b).to(dev)
            if not generative:   # SupCon alone from the first epoch
                mu = vae.encode(x[b], m[b], t[b])[0]
                loss = supcon_loss(F.normalize(mu, dim=1), sc_y[b], params["temp"])
            else:
                logits, tab_recon, mu, logvar = vae(x[b], m[b], t[b])
                loss = vae_loss(logits, tab_recon, x[b], m[b], t[b], mu, logvar, beta)
            if generative and lam > 0:
                loss = loss + lam * supcon_loss(F.normalize(mu, dim=1), sc_y[b], params["temp"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), C.GRAD_CLIP)
            opt.step()
        sched.step()

    model = _Model(vae, stats, len(d.cls))
    z = model.latent(maps[tr], mask[tr], raw[tr])
    model.scaler = StandardScaler().fit(z)
    if (head or HEAD) == "linear":
        model.head = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000,
                                        random_state=seed)
    else:
        model.head = HistGradientBoostingClassifier(
            max_iter=C.HGB_MAX_ITER, learning_rate=C.HGB_LR, class_weight="balanced",
            random_state=seed)
    model.head.fit(model.scaler.transform(z), d.y[tr])
    return model


def predict(model, d, rows):
    maps, mask, raw = _inputs(d)
    return model.proba(maps[rows], mask[rows], raw[rows])


def embed(model, d, rows):
    maps, mask, raw = _inputs(d)
    return model.latent(maps[rows], mask[rows], raw[rows])


def predict_degraded(model, d, idx):
    x = degrade.load()[0][idx]
    return model.proba(*_maps(x), degrade.load_folded()[1][idx])
