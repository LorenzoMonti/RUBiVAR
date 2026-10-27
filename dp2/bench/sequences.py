"""
Raw light curves as token sequences, for the sequence models (transformer).

Cache: catalogs/sequences.npz  ids, offsets (n+1), t (float64), mag, err (float32),
band (int8) — same object order as features.npz. Points with flux <= 0 are kept with
mag NaN so that the first-n-nights cut sees the same nights as the feature models
(degrade.py), and dropped only after the cut.

Token = one observation: magnitude minus its band's median (clipped), log error, the
band's median offset (colour), band index, and a sinusoidal encoding of the real time
over periods 0.02–1000 d. No period, no folding.
"""

import os
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
import features
from bench import protocol
from lightcurves import read_lightcurve

CACHE = os.path.join(C.CATALOG_DIR, "sequences.npz")
MAX_LEN = 512
N_FREQ = 32
PERIODS = np.geomspace(0.02, 1000, N_FREQ)


def _one(oid):
    df = read_lightcurve(os.path.join(C.LC_DIR, f"lc_{oid}.csv"))
    df = df.dropna(subset=["psfFlux", "psfFluxErr", "expMidptMJD"])
    df = df[(df["psfFluxErr"] > 0) & df["band"].isin(C.BANDS)].sort_values("expMidptMJD")
    f = df["psfFlux"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(f > 0, features.ZP_NJY - 2.5 * np.log10(f), np.nan)
    return (df["expMidptMJD"].to_numpy(), mag.astype(np.float32),
            (1.0857 * df["psfFluxErr"].to_numpy() / np.abs(f)).astype(np.float32),
            df["band"].map(C.BAND_IDX).to_numpy().astype(np.int8))


def build():
    ids, _, _ = features.load()
    with Pool(max(1, cpu_count() - 1)) as pool:
        res = pool.map(_one, ids, chunksize=64)
    offsets = np.concatenate([[0], np.cumsum([len(r[0]) for r in res])])
    np.savez(CACHE, ids=np.array(ids), offsets=offsets,
             **{k: np.concatenate([r[i] for r in res])
                for i, k in enumerate(("t", "mag", "err", "band"))})
    print(f"{len(ids):,} curves, {offsets[-1]:,} points → {CACHE}")


def load():
    if not os.path.exists(CACHE):
        raise FileNotFoundError(f"{CACHE} missing — run `main.py bench-cache`.")
    return dict(np.load(CACHE))


class Curves(Dataset):
    """
    Rows of the cache. Curves longer than MAX_LEN are cropped to random points when
    training, evenly spaced ones otherwise; n_keep (one per row) cuts each curve to its
    first nights, as features_degraded.npz.
    """

    def __init__(self, cache, rows, y, train=False, n_keep=None):
        self.c, self.rows, self.y, self.train, self.n_keep = cache, rows, y, train, n_keep

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, c = self.rows[i], self.c
        a, b = c["offsets"][r], c["offsets"][r + 1]
        t, mag, err, band = c["t"][a:b], c["mag"][a:b], c["err"][a:b], c["band"][a:b]
        keep = np.isfinite(mag)
        if self.n_keep is not None:
            keep &= protocol.first_nights(protocol.night_of(t), self.n_keep[i])
        t, mag, err, band = t[keep], mag[keep], err[keep], band[keep]
        if len(t) == 0:      # a cut can leave only non-positive fluxes: one blank token
            t, mag, err, band = np.zeros(1), np.zeros(1, np.float32), np.ones(1, np.float32), \
                np.zeros(1, np.int8)
        if len(t) > MAX_LEN:
            # seeded from torch, which the training seed fixes per worker and epoch
            rng = np.random.default_rng([torch.initial_seed() % 2**32, i])
            idx = (np.sort(rng.choice(len(t), MAX_LEN, replace=False))
                   if self.train
                   else np.linspace(0, len(t) - 1, MAX_LEN).astype(int))
            t, mag, err, band = t[idx], mag[idx], err[idx], band[idx]

        med = np.zeros(len(C.BANDS), np.float32)
        for bi in np.unique(band):
            med[bi] = np.median(mag[band == bi])
        ref = np.median(med[np.unique(band)])
        x = np.stack([np.clip(mag - med[band], -5, 5),
                      np.log10(np.clip(err, 1e-4, 10)),
                      np.clip(med[band] - ref, -5, 5)], 1).astype(np.float32)
        ph = 2 * np.pi * (t - t[0])[:, None] / PERIODS[None, :]
        te = np.concatenate([np.sin(ph), np.cos(ph)], 1).astype(np.float32)
        return x, te, band.astype(np.int64), self.y[i]


def collate(batch):
    n, B = max(len(b[0]) for b in batch), len(batch)
    x = np.zeros((B, n, 3), np.float32)
    te = np.zeros((B, n, 2 * N_FREQ), np.float32)
    band = np.zeros((B, n), np.int64)
    pad = np.ones((B, n), bool)
    for i, (xi, ti, bi, _) in enumerate(batch):
        x[i, :len(xi)], te[i, :len(xi)], band[i, :len(xi)], pad[i, :len(xi)] = xi, ti, bi, False
    y = np.array([b[3] for b in batch])
    return tuple(torch.from_numpy(a) for a in (x, te, band, pad, y))
