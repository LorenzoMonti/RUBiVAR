"""
Semi-supervised VAE (SS-VAE) + linear probe — the generative entry of the
benchmark. Input: the phase fold on the LS period (6 bands + mask) and TAB_COLUMNS
(period, LS power, colours, r amplitude; no Gaia quantity).

Per outer fold the VAE is retrained from scratch on the training fields only: their
labelled objects (SupCon on the original fine-grained labels, classes with >=
C.MIN_CLASS_SIZE members) and the unlabelled objects outside the test and validation
fields. A logistic regression on the latent mean then gives the 5 class probabilities:
the latent is judged by what a linear model can read from it.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
from bench import degrade

NAME = "vae"
USES_VAL = False
EPOCHS, BATCH = 100, 256
BETA_WARM, LAMBDA_WARM = 0.25, 0.4   # fractions of EPOCHS: KL warm-up, then SupCon's
TAB_DIM = len(degrade.TAB_COLUMNS) + 1
_IN = None


# ── Model ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Input is 2 x len(BANDS) channels: the folded flux, then the observation mask.

    The mask has to be an input, not just a term in the loss. Each band is normalised
    to zero mean, so a zero in the flux channel means both "no measurement here" and
    "flux at the band average" — indistinguishable. With RR Lyrae filling only about a
    third of their phase bins, the encoder would otherwise read a flat curve at the
    mean across most of the fold. The mask channel tells it which is which.
    """

    def __init__(self, latent_dim, tab_dim):
        super().__init__()
        n_in = 2 * len(C.BANDS)
        # padding_mode="circular": phase 1.0 wraps onto phase 0.0, so the fold has no
        # edge — zero-padding would invent one and break the translation equivariance
        # that makes a CNN the right choice for an arbitrary phase origin.
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, 32, 5, padding=2, padding_mode="circular"),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2, stride=2, padding_mode="circular"),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2, stride=2, padding_mode="circular"),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        # Measured, not derived from a formula: two stride-2 stages do not divide
        # every N_PHASE evenly, and guessing N_PHASE // 4 breaks for e.g. 50.
        with torch.no_grad():
            self.conv_len = self.conv(torch.zeros(1, n_in, C.N_PHASE)).shape[-1]
        self.flat_dim = 128 * self.conv_len
        self.fc_cnn = nn.Sequential(nn.Linear(self.flat_dim, 256), nn.GELU())
        self.fc_tab = nn.Sequential(nn.Linear(tab_dim, 32), nn.GELU())
        self.fc_mu = nn.Linear(256 + 32, latent_dim)
        self.fc_lv = nn.Linear(256 + 32, latent_dim)

    def forward(self, x, mask, tab):
        h_cnn = self.conv(torch.cat([x, mask], dim=1)).flatten(1)
        h = torch.cat([self.fc_cnn(h_cnn), self.fc_tab(tab)], dim=1)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 128 * self.seq_len), nn.GELU(),
        )
        # ConvTranspose1d supports only zero padding in PyTorch, so the decoder cannot
        # be made circular. It matters far less: the latent is built by the encoder,
        # and the decoder only has to reconstruct well enough to shape it.
        # A second head reconstructs the tabular vector. Without it nothing forces
        # period, colours and ls_power through to the latent: they enter the encoder
        # but are never reconstructed, so the ELBO has no reason to preserve them and
        # the 256-dim CNN branch drowns the 32-dim tabular branch.
        self.fc_tab_out = None    # set by SSVAE, which knows tab_dim
        self.dconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, len(C.BANDS), 5, padding=2),
        )

    def forward(self, z):
        out = self.dconv(self.fc(z).view(-1, 128, self.seq_len))
        # Two stride-2 stages give 4 * seq_len, which can overshoot N_PHASE when it
        # is not a multiple of 4 (50 → 52). Trim rather than constrain N_PHASE.
        return out[..., :C.N_PHASE], self.fc_tab_out(z)


class SSVAE(nn.Module):
    def __init__(self, latent_dim, tab_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim, tab_dim)
        self.decoder = Decoder(latent_dim, self.encoder.conv_len)
        self.decoder.fc_tab_out = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, tab_dim))

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x, mask, tab):
        mu, logvar = self.encoder(x, mask, tab)
        recon, tab_recon = self.decoder(self.reparameterize(mu, logvar))
        return recon, tab_recon, mu, logvar


# ── Losses ────────────────────────────────────────────────────────────────────

def vae_loss(recon, tab_recon, x, mask, tab, mu, logvar, beta):
    """
    Masked ELBO. Bands with no data must not be fitted to zero, hence the mask.

    The reconstruction is summed over the tensor's elements, not averaged over them.
    Averaging silently rescales beta by the number of elements (6 x N_PHASE = 300):
    an apparently mild beta=0.1 then acts like beta=30, the KL term outweighs the
    reconstruction several times over and the encoder collapses to the prior — a
    latent of isotropic noise, variance spread evenly over all 32 dimensions.

    Each object is normalised by its OWN valid-element count and then rescaled to the
    full tensor, so objects observed in three bands weigh the same as those observed
    in six instead of contributing proportionally less gradient.
    """
    n_elem = x.shape[1] * x.shape[2]
    per_obj = mask.sum(dim=[1, 2]).clamp(min=1)
    recon_per_elem = ((recon - x) ** 2 * mask).sum(dim=[1, 2]) / per_obj
    recon_loss = (recon_per_elem * n_elem).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).clamp(min=C.FREE_BITS).sum(1).mean()
    tab_loss = ((tab_recon - tab) ** 2).sum(1).mean()
    total = recon_loss + C.TAB_RECON_WEIGHT * tab_loss + beta * kl
    # recon_per_elem is what gets reported: comparable across runs and N_PHASE values.
    return total, recon_per_elem.mean(), kl, tab_loss


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
            "temp": trial.suggest_categorical("temp", [0.05, 0.1, 0.2]),
            "C": trial.suggest_float("C", 1e-2, 10.0, log=True)}


def _inputs(d):
    """(fold, raw tab) of every object, in d's order; NaN fold where none was written."""
    global _IN
    if _IN is None:
        fold = np.full((len(d.ids), len(C.BANDS), C.N_PHASE), np.nan, np.float32)
        for i, oid in enumerate(d.ids):
            path = os.path.join(C.FOLDED_DIR, f"folded_{oid}.npy")
            if os.path.exists(path):
                fold[i] = np.load(path)
        per = pd.read_csv(C.PERIODS_CSV, dtype={"diaObjectId": str}).set_index("diaObjectId")
        _IN = fold, per.reindex(d.ids)[degrade.TAB_COLUMNS].to_numpy(np.float64)
    return _IN


def _tab(raw, stats=None):
    """log period, z-scored, NaN → 0, plus the fraction of colours/amplitude missing."""
    t = raw.copy()
    t[:, 0] = np.log10(np.clip(t[:, 0], 1e-3, None))
    if stats is None:
        stats = np.nanmean(t, 0), np.nanstd(t, 0) + 1e-6
    z = np.nan_to_num((t - stats[0]) / stats[1])
    return np.hstack([z, np.isnan(t[:, 2:]).mean(1, keepdims=True)]).astype(np.float32), stats


def _tensors(fold, tab, dev):
    mask = ~np.isnan(fold)
    return (torch.from_numpy(np.where(mask, fold, 0).astype(np.float32)).to(dev),
            torch.from_numpy(mask.astype(np.float32)).to(dev), torch.from_numpy(tab).to(dev))


class _Model:
    def __init__(self, vae, stats, scaler, probe, n_cls):
        self.vae, self.stats, self.scaler, self.probe, self.n_cls = vae, stats, scaler, probe, n_cls

    @torch.no_grad()
    def latent(self, fold, raw_tab):
        self.vae.eval()
        x, m, t = _tensors(fold, _tab(raw_tab, self.stats)[0], next(self.vae.parameters()).device)
        return np.vstack([self.vae.encoder(x[i:i + 2048], m[i:i + 2048], t[i:i + 2048])[0]
                          .cpu().numpy() for i in range(0, len(x), 2048)])

    def proba(self, fold, raw_tab):
        out = np.zeros((len(fold), self.n_cls))
        out[:, self.probe.classes_] = self.probe.predict_proba(
            self.scaler.transform(self.latent(fold, raw_tab)))
        return out


def train(d, tr, va, params, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold, raw = _inputs(d)
    test_fold = sorted(set(range(C.N_FIELD_FOLDS)) - set(d.folds[tr]) - set(d.folds[va]))
    unl = np.where(~d.lab & ~np.isin(d.folds, test_fold)
                   & ~np.isin(d.fields, np.unique(d.fields[va])))[0]
    rows = np.concatenate([tr, unl])

    # SupCon labels: fine-grained labels of the training rows, large enough classes
    names, counts = np.unique(d.labels[tr].astype(str), return_counts=True)
    keep = {n: i for i, n in enumerate(names[counts >= C.MIN_CLASS_SIZE])}
    sc_y = np.array([keep.get(str(l), -1) for l in d.labels[tr]] + [-1] * len(unl))

    tab, stats = _tab(raw[rows])
    x, m, t = _tensors(fold[rows], tab, dev)
    sc_y = torch.from_numpy(sc_y).to(dev)
    vae = SSVAE(params["latent_dim"], TAB_DIM).to(dev)
    opt = torch.optim.AdamW(vae.parameters(), lr=params["lr"], weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    b_end, l_end = BETA_WARM * EPOCHS, (BETA_WARM + LAMBDA_WARM) * EPOCHS
    for ep in range(1, EPOCHS + 1):
        beta = params["beta_max"] * min(1.0, ep / b_end)
        lam = params["lambda_max"] * float(np.clip((ep - b_end) / (l_end - b_end), 0, 1))
        vae.train()
        for b in np.array_split(rng.permutation(len(rows)), max(1, len(rows) // BATCH)):
            b = torch.from_numpy(b).to(dev)
            recon, tab_recon, mu, logvar = vae(x[b], m[b], t[b])
            loss = vae_loss(recon, tab_recon, x[b], m[b], t[b], mu, logvar, beta)[0]
            if lam > 0:
                loss = loss + lam * supcon_loss(F.normalize(mu, dim=1), sc_y[b],
                                                      params["temp"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), C.GRAD_CLIP)
            opt.step()
        sched.step()

    model = _Model(vae, stats, None, None, len(d.cls))
    z = model.latent(fold[tr], raw[tr])
    model.scaler = StandardScaler().fit(z)
    model.probe = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000,
                                     random_state=seed).fit(model.scaler.transform(z), d.y[tr])
    return model


def predict(model, d, rows):
    fold, raw = _inputs(d)
    return model.proba(fold[rows], raw[rows])


def embed(model, d, rows):
    fold, raw = _inputs(d)
    return model.latent(fold[rows], raw[rows])


def predict_degraded(model, d, idx):
    fold, raw = degrade.load_folded()
    return model.proba(fold[idx], raw[idx])
