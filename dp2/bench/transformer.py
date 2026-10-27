"""
Transformer on raw multiband light curves — the DL entry of the benchmark. It gets no
period and no features: it has to find the variability from the tokens alone (see
sequences.py). Weighted cross-entropy, AdamW + one-cycle, bf16; early stopping on the
inner validation fields, whose classes with < 20 objects are left out of the stopping
loss (LPV live almost only in the bulge, so validation can hold a handful).
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from bench import degrade, sequences

NAME = "transformer"
USES_VAL = True
EPOCHS, PATIENCE, WORKERS = 30, 6, 8
_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = sequences.load()
    return _CACHE


class LCTransformer(nn.Module):
    def __init__(self, n_cls, d=128, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(3, d)
        self.band = nn.Embedding(len(C.BANDS), d)
        self.time = nn.Linear(2 * sequences.N_FREQ, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d, n_cls))

    def embed(self, x, te, band, pad):
        """The curve's representation: token states averaged over its observations."""
        h = self.inp(x) + self.band(band) + self.time(te)
        h = self.norm(self.enc(h, src_key_padding_mask=pad))
        keep = (~pad).unsqueeze(-1).float()
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1)

    def forward(self, x, te, band, pad):
        return self.head(self.embed(x, te, band, pad))


def space(trial):
    return {"d": trial.suggest_categorical("d", [64, 128, 192]),
            "layers": trial.suggest_int("layers", 2, 6),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "wd": trial.suggest_float("wd", 1e-3, 0.1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "batch": trial.suggest_categorical("batch", [32, 64, 128])}


def _loader(ds, batch, shuffle=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=WORKERS,
                      collate_fn=sequences.collate)


def _predict(model, loader, embed=False):
    """Class probabilities, or with embed=True the pooled representation."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, te, band, pad, _ in loader:
            args = x.to(dev), te.to(dev), band.to(dev), pad.to(dev)
            out.append(model.embed(*args).float().cpu().numpy() if embed
                       else torch.softmax(model(*args).float(), 1).cpu().numpy())
    return np.concatenate(out)


def train(d, tr, va, params, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev, n_cls, c = torch.device("cuda"), len(d.cls), _cache()
    counts = np.bincount(d.y[tr], minlength=n_cls)
    w = torch.tensor(len(tr) / (n_cls * np.maximum(counts, 1)), dtype=torch.float32)
    yv = d.y[va]
    w_val = w * torch.tensor(np.bincount(yv, minlength=n_cls) >= 20, dtype=torch.float32)

    train_dl = _loader(sequences.Curves(c, tr, d.y[tr], train=True),
                       params["batch"], shuffle=True)
    val_dl = _loader(sequences.Curves(c, va, yv), 256)
    model = LCTransformer(n_cls, params["d"], params["layers"],
                          dropout=params["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, params["lr"], pct_start=0.05,
                                                total_steps=EPOCHS * len(train_dl))
    lossf = nn.CrossEntropyLoss(weight=w.to(dev))
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, te, band, pad, yb in train_dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.to(dev), te.to(dev), band.to(dev), pad.to(dev))
            loss = lossf(logits.float(), yb.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        pv = _predict(model, val_dl)
        vloss = float(nn.functional.nll_loss(torch.log(torch.tensor(pv) + 1e-9),
                                             torch.tensor(yv), weight=w_val))
        print(f"    ep {ep:2d}  val loss {vloss:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, d, rows):
    return _predict(model, _loader(sequences.Curves(_cache(), rows, d.y[rows]), 256))


def embed(model, d, rows):
    return _predict(model, _loader(sequences.Curves(_cache(), rows, d.y[rows]), 256), True)


def predict_degraded(model, d, idx):
    _, rows, n_keep = degrade.load()
    return _predict(model, _loader(sequences.Curves(_cache(), rows[idx], d.y[rows[idx]],
                                                    n_keep=n_keep[idx]), 256))
