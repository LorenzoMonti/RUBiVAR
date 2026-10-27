"""
The few-nights test set: labelled objects with >= C.DEG_MIN_NIGHTS nights cut to their
first n nights (n in C.DEG_LEVELS), re-measured from scratch (LS period included)
exactly as the features stage does.

Output: catalogs/features_degraded.npz, one row per (object, n), deterministic:
  X       the 460 features of features.npz
  fold    (6, N_PHASE) phase fold, as the fold stage writes it (NaN = band unused)
  tab     TAB_COLUMNS, as periods.csv holds them for the real curves
  row     the object's row in features.npz;  n_keep
"""

import os
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

import config as C
import features
import lightcurves
from bench import protocol

# The VAE's tabular inputs, all measured on the curve itself (no Gaia quantity).
TAB_COLUMNS = ["period_days", "ls_power", "colour_ug", "colour_gr", "colour_ri", "amplitude_r"]


def _job(args):
    oid, n_keep = args
    df = lightcurves.read_lightcurve(os.path.join(C.LC_DIR, f"lc_{oid}.csv"))
    df = df.dropna(subset=["psfFlux", "psfFluxErr", "expMidptMJD"])
    df = df[df["psfFluxErr"] > 0]                 # as the fold stage
    df = df[protocol.first_nights(protocol.night_of(df["expMidptMJD"].to_numpy()), n_keep)]
    period, power = np.nan, np.nan
    fold = np.full((len(C.BANDS), C.N_PHASE), np.nan, np.float32)
    res = lightcurves.lomb_scargle_period(df) if len(df) >= C.MIN_POINTS else None
    if res is not None:
        period, power = res[:2]
        fold = lightcurves.fold_curve(df, period, res[2])
    feat, maps = features.curve_features(df, period, power)
    tab = {"period_days": period, "ls_power": power, **lightcurves.photometric_features(df)}
    return feat, maps.reshape(-1), fold, [tab[c] for c in TAB_COLUMNS]


def build(workers=None):
    d = protocol.load()
    _, names, _ = features.load()
    dm_mean = np.load(C.FEATURES_NPZ)["dmdt_mean"]
    classic = [n for n in names if not n.startswith("dmdt_")]
    sel = np.where(d.lab & (d.nights >= C.DEG_MIN_NIGHTS))[0]
    jobs = [(d.ids[i], n) for i in sel for n in C.DEG_LEVELS]
    with Pool(workers or max(1, cpu_count() - 1)) as pool:
        res = list(tqdm(pool.imap(_job, jobs, chunksize=16), total=len(jobs),
                        desc="first-n-nights features"))
    x = np.empty((len(res), len(names)), np.float32)
    for j, (feat, maps, _, _) in enumerate(res):
        x[j, :len(classic)] = [feat.get(c, np.nan) for c in classic]
        x[j, len(classic):] = np.where(np.isnan(maps), dm_mean, maps)
    np.savez_compressed(C.FEATURES_DEG_NPZ, X=x, row=np.repeat(sel, len(C.DEG_LEVELS)),
                        n_keep=np.tile(C.DEG_LEVELS, len(sel)),
                        fold=np.stack([r[2] for r in res]),
                        tab=np.array([r[3] for r in res], np.float64))
    print(f"{len(sel):,} objects x {len(C.DEG_LEVELS)} levels → {C.FEATURES_DEG_NPZ}")


def _file():
    if not os.path.exists(C.FEATURES_DEG_NPZ):
        raise FileNotFoundError(f"{C.FEATURES_DEG_NPZ} missing — run `main.py bench-cache`.")
    return np.load(C.FEATURES_DEG_NPZ)


def load():
    """(X, row, n_keep) of the few-nights test set."""
    a = _file()
    return a["X"], a["row"], a["n_keep"]


def load_folded():
    """(fold, tab) of the few-nights test set, for the VAE."""
    a = _file()
    return a["fold"], a["tab"]
