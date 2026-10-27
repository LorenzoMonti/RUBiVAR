"""
Classical light-curve features + coarse dm-dt maps, one row per object.

Input : lightcurves/lc_<diaObjectId>.csv, catalogs/periods.csv (fold stage)
Output: catalogs/features.npz   ids, names, X (float32, NaN = not measurable)

Per band (>= C.FEAT_MIN_BAND_POINTS points): amplitude and scatter, the shape of the
magnitude distribution (skewness, kurtosis, flux-percentile ratios, Stetson K, von
Neumann eta) and chi2 against a constant. In g, r, i a Fourier fit on the LS period
(phase-offset-invariant R21, R31, phi21, phi31) and a 2-harmonic fit on twice the
period, since eclipsing binaries are often found at half their period. Then colours,
inter-band amplitude ratios, log period and LS power, and the dm-dt block.

No feature counts points, nights or time span — see config, "Classical features".
"""

import glob
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from tqdm import tqdm

import config as C
from lightcurves import read_lightcurve

ZP_NJY = 31.4   # AB zero point for fluxes in nJy

_PERIODS = None   # diaObjectId -> (period_days, ls_power), set per worker


def _init_worker(periods):
    global _PERIODS
    _PERIODS = periods


def band_stats(m, e):
    """Distribution statistics of one band's magnitudes, in time order."""
    med, sd = np.median(m), np.std(m)
    p = np.percentile(m, [5, 10, 25, 32.5, 67.5, 75, 90, 95])
    r595 = p[-1] - p[0]
    w_mean = np.average(m, weights=1 / e**2)
    d = (m - m.mean()) / e * np.sqrt(len(m) / (len(m) - 1))
    nan = np.nan
    return {
        "amp": r595, "std": sd, "mad": np.median(np.abs(m - med)), "iqr": p[5] - p[2],
        "skew": skew(m), "kurt": kurtosis(m),
        "beyond1std": np.mean(np.abs(m - m.mean()) > sd),
        # Richards et al. flux-percentile ratios: shape of the distribution's core
        "fpr20": (p[4] - p[3]) / r595 if r595 > 0 else nan,
        "fpr50": (p[5] - p[2]) / r595 if r595 > 0 else nan,
        "fpr80": (p[6] - p[1]) / r595 if r595 > 0 else nan,
        # brief dips (eclipses) pull the mean faintward of the median, brief peaks the other way
        "med_minus_mean": (med - m.mean()) / sd if sd > 0 else nan,
        "eta": np.sum(np.diff(m) ** 2) / (len(m) - 1) / sd**2 if sd > 0 else nan,
        "chi2_red": np.sum(((m - w_mean) / e) ** 2) / (len(m) - 1),
        "stetsonK": np.mean(np.abs(d)) / np.sqrt(np.mean(d**2)),
    }


def fourier(t, m, e, period, n_harm):
    """Weighted Fourier fit; amplitude ratios and phase differences ignore the phase origin."""
    ph = 2 * np.pi * t / period
    cols = [np.ones_like(t)]
    for k in range(1, n_harm + 1):
        cols += [np.sin(k * ph), np.cos(k * ph)]
    a = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(a / e[:, None], m / e, rcond=None)
    amp = [np.hypot(coef[2 * k - 1], coef[2 * k]) for k in range(1, n_harm + 1)]
    phi = [np.arctan2(coef[2 * k], coef[2 * k - 1]) for k in range(1, n_harm + 1)]
    out = {"A1": amp[0],
           "R21": amp[1] / amp[0] if amp[0] > 0 else np.nan,
           "phi21": (phi[1] - 2 * phi[0]) % (2 * np.pi),
           "var_expl": 1 - np.var(m - a @ coef) / np.var(m) if np.var(m) > 0 else np.nan}
    if n_harm >= 3:
        out["R31"] = amp[2] / amp[0] if amp[0] > 0 else np.nan
        out["phi31"] = (phi[2] - 3 * phi[0]) % (2 * np.pi)
    return out


def dmdt(t, m):
    """Pairwise (dt, dm) histogram, each dt column normalised; NaN where a column is empty."""
    n_dt, n_dm = len(C.DMDT_DT_EDGES) - 1, len(C.DMDT_DM_EDGES) - 1
    if len(t) < 3:
        return np.full((n_dt, n_dm), np.nan)
    o = np.argsort(t)
    t, m = t[o], m[o]
    i, j = np.triu_indices(len(t), 1)
    h, _, _ = np.histogram2d(t[j] - t[i], m[j] - m[i],
                             bins=[C.DMDT_DT_EDGES, C.DMDT_DM_EDGES])
    col = h.sum(axis=1, keepdims=True)
    return np.where(col > 0, h / np.maximum(col, 1), np.nan)


def coarse_dmdt(fine):
    """Pool fine dt columns into C.DMDT_DT_GROUPS, averaging only the populated ones."""
    return np.stack([np.nanmean(fine[g], axis=0) if np.isfinite(fine[g, 0]).any()
                     else np.full(fine.shape[1], np.nan)
                     for g in C.DMDT_DT_GROUPS])


def process_one(fpath):
    oid = os.path.basename(fpath)[3:-4]
    df = read_lightcurve(fpath).dropna(subset=["psfFlux", "psfFluxErr", "expMidptMJD"])
    return (oid, *curve_features(df, *_PERIODS.get(oid, (np.nan, np.nan))))


def curve_features(df, period, power):
    """(classical feature dict, dm-dt maps) of one cleaned light curve and its LS period."""
    df = df[(df["psfFlux"] > 0) & (df["psfFluxErr"] > 0)]
    df = df.assign(mag=ZP_NJY - 2.5 * np.log10(df["psfFlux"]),
                   err=1.0857 * df["psfFluxErr"] / df["psfFlux"])
    df = df.sort_values("expMidptMJD")

    feat, med = {}, {}
    for b in C.BANDS:
        s = df[df["band"] == b]
        if len(s) < C.FEAT_MIN_BAND_POINTS:
            continue
        t, m, e = s["expMidptMJD"].to_numpy(), s["mag"].to_numpy(), s["err"].to_numpy()
        med[b] = np.median(m)
        feat.update({f"{b}_{k}": v for k, v in band_stats(m, e).items()})
        if b in "gri" and len(s) >= C.FOURIER_MIN_POINTS and np.isfinite(period):
            feat.update({f"{b}_{k}": v
                         for k, v in fourier(t, m, e, period, C.N_HARMONICS).items()})
            two_p = fourier(t, m, e, 2 * period, 2)
            feat.update({f"{b}_2P_{k}": two_p[k] for k in ("A1", "R21", "var_expl")})
    for a, b in zip(C.BANDS[:-1], C.BANDS[1:]):
        if a in med and b in med:
            feat[f"col_{a}{b}"] = med[a] - med[b]
    for a, b in (("g", "r"), ("r", "i")):
        if feat.get(f"{b}_amp", 0) > 0 and f"{a}_amp" in feat:
            feat[f"ampratio_{a}{b}"] = feat[f"{a}_amp"] / feat[f"{b}_amp"]
    if np.isfinite(period):
        feat["logP"], feat["ls_power"] = np.log10(period), power

    # dm-dt on band-median-subtracted magnitudes: "all" pools every band
    df["dmag"] = df["mag"] - df.groupby("band")["mag"].transform("median")
    maps = [coarse_dmdt(dmdt(df["expMidptMJD"].to_numpy(), df["dmag"].to_numpy()))]
    for b in C.DMDT_MAPS[1:]:
        s = df[df["band"] == b]
        maps.append(coarse_dmdt(dmdt(s["expMidptMJD"].to_numpy(), s["dmag"].to_numpy())))
    return feat, np.stack(maps).astype(np.float32)


def dmdt_names():
    n_dm = len(C.DMDT_DM_EDGES) - 1
    return [f"dmdt_{mp}_t{g}_m{k}" for mp in C.DMDT_MAPS
            for g in range(len(C.DMDT_DT_GROUPS)) for k in range(n_dm)]


def run(workers=None):
    C.ensure_dirs(C.CATALOG_DIR)
    files = sorted(glob.glob(os.path.join(C.LC_DIR, "lc_*.csv")))
    if not files:
        raise FileNotFoundError(f"No light curves in {C.LC_DIR}.")
    if not os.path.exists(C.PERIODS_CSV):
        raise FileNotFoundError(f"{C.PERIODS_CSV} missing — run the fold stage first.")
    per = pd.read_csv(C.PERIODS_CSV, dtype={"diaObjectId": str})
    periods = dict(zip(per["diaObjectId"], zip(per["period_days"], per["ls_power"])))

    n_workers = workers or max(1, cpu_count() - 1)
    print(f"Light curves : {len(files):,}   periods: {len(periods):,}   workers: {n_workers}")
    with Pool(n_workers, initializer=_init_worker, initargs=(periods,)) as pool:
        res = list(tqdm(pool.imap(process_one, files, chunksize=32),
                        total=len(files), desc="Features"))

    ids = np.array([r[0] for r in res])
    classic = pd.DataFrame([r[1] for r in res])
    classic = classic[sorted(classic.columns)]
    maps = np.stack([r[2] for r in res]).reshape(len(res), -1)
    # Columns still empty after pooling take the population mean, so "empty" cannot
    # stand in for a cadence pattern. Rows of an object with no data at all stay mean.
    dm_mean = np.nanmean(maps, axis=0)
    maps = np.where(np.isnan(maps), dm_mean, maps)

    names = list(classic.columns) + dmdt_names()
    x = np.hstack([classic.to_numpy(dtype=np.float32), maps]).astype(np.float32)
    np.savez_compressed(C.FEATURES_NPZ, ids=ids, names=np.array(names), X=x,
                        dmdt_mean=dm_mean)
    print(f"\nFeatures: {x.shape[0]:,} objects x {x.shape[1]} "
          f"({classic.shape[1]} classical + {maps.shape[1]} dm-dt) → {C.FEATURES_NPZ}")
    cover = classic.notna().mean().sort_values()
    print("Least available classical features:")
    print((cover.head(8) * 100).round(0).astype(int).astype(str).add("%").to_string())


def load():
    """(ids, names, X) from the features stage."""
    if not os.path.exists(C.FEATURES_NPZ):
        raise FileNotFoundError(f"{C.FEATURES_NPZ} missing — run the features stage.")
    d = np.load(C.FEATURES_NPZ)
    return list(d["ids"]), list(d["names"]), d["X"]
