"""
Lomb-Scargle period search + multiband phase-folding.

Input : lightcurves/lc_<diaObjectId>.csv  (band, psfFlux, psfFluxErr, expMidptMJD)
Output: folded/folded_<diaObjectId>.npy   (6, N_PHASE) float32, bands [u,g,r,i,z,y]
        catalogs/periods.csv              diaObjectId, period_days, ls_power, ...

Bands with too few detections are left as NaN in the fold; the VAE masks them out
of the reconstruction loss, so a partially covered object is still usable.
"""

import os
import glob
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from tqdm import tqdm

import config as C

# Set by _init_worker so the pool doesn't re-send them with every task.
_MIN_POINTS = C.MIN_POINTS
_MIN_PER_BAND = C.MIN_POINTS_PER_BAND


def _init_worker(min_points, min_per_band):
    global _MIN_POINTS, _MIN_PER_BAND
    _MIN_POINTS, _MIN_PER_BAND = min_points, min_per_band


def fold_curve(df, period, bands_used):
    """(len(BANDS), N_PHASE) float32 fold, NaN for bands not used."""
    folded = np.full((len(C.BANDS), C.N_PHASE), np.nan)
    for b in bands_used:
        sub = df[df["band"] == b]
        raw = phase_fold_band(sub["expMidptMJD"].to_numpy(),
                              sub["psfFlux"].to_numpy(),
                              sub["psfFluxErr"].to_numpy(), period)
        folded[C.BAND_IDX[b]] = normalize_band(raw)
    return folded.astype(np.float32)


def phase_fold_band(times, flux, flux_err, period):
    """
    Inverse-variance weighted mean of the flux in each phase bin.

    Vectorised through bincount rather than looping over bins: at DP2 scale the
    loop version dominated the runtime.
    """
    phase = (times % period) / period
    idx = np.minimum((phase * C.N_PHASE).astype(np.int64), C.N_PHASE - 1)
    w = 1.0 / flux_err ** 2
    num = np.bincount(idx, weights=w * flux, minlength=C.N_PHASE)
    den = np.bincount(idx, weights=w, minlength=C.N_PHASE)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def normalize_band(arr):
    """Zero mean, unit std. NaN array if the band carries too little signal."""
    valid = arr[~np.isnan(arr)]
    if len(valid) < 3:
        return np.full(C.N_PHASE, np.nan)
    sigma = np.nanstd(arr)
    if sigma < 1e-10:
        return np.full(C.N_PHASE, np.nan)
    return (arr - np.nanmean(arr)) / sigma


def lomb_scargle_period(df):
    """
    Multiband LS on the concatenated, per-band mean-subtracted series.
    Returns (period_days, power, bands_used) or None.
    """
    times, fluxes, errs, bands_used = [], [], [], []
    for b in C.BANDS:
        sub = df[df["band"] == b]
        if len(sub) < _MIN_PER_BAND:
            continue
        bands_used.append(b)
        f = sub["psfFlux"].to_numpy()
        times.append(sub["expMidptMJD"].to_numpy())
        fluxes.append(f - f.mean())          # zero-mean before combining bands
        errs.append(sub["psfFluxErr"].to_numpy())

    if not bands_used:
        return None

    t = np.concatenate(times)
    f = np.concatenate(fluxes)
    e = np.concatenate(errs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls = LombScargle(t, f, e, normalization="standard")
        freqs = np.linspace(C.FREQ_MIN, C.FREQ_MAX, C.N_FREQ)
        power = ls.power(freqs)

    best = int(np.argmax(power))
    return 1.0 / freqs[best], float(power[best]), bands_used


def read_lightcurve(fpath):
    """
    Read one light-curve CSV and normalise it to band / psfFlux / psfFluxErr /
    expMidptMJD, whatever the download actually named things.

    Two things vary and are resolved here rather than assumed:

    * the time column — DP1 called it expMidptMJD, DP2 takes it from dp2.Visit;
    * the flux column — ForcedSourceOnDiaObject carries both psfFlux (forced on the
      science image, total flux) and psfDiffFlux (forced on the difference image).
      C.FLUX_COLUMN picks one; whichever it is, it is renamed to psfFlux so the rest
      of the pipeline never has to care.

    Rows whose fit failed (`<flux>_flag` true) are dropped: forced photometry
    reports a value for every visit including the ones where the fit did not
    converge, and those are not measurements.
    """
    df = pd.read_csv(fpath)

    time_col = next((c for c in C.TIME_COLUMNS if c in df.columns), None)
    if time_col is None:
        raise KeyError(f"{fpath}: no time column among {C.TIME_COLUMNS}")

    flux_col = C.FLUX_COLUMN if C.FLUX_COLUMN in df.columns else "psfFlux"
    err_col = f"{flux_col}Err"
    if flux_col not in df.columns or err_col not in df.columns:
        raise KeyError(f"{fpath}: missing {flux_col}/{err_col}")

    flag_col = f"{flux_col}_flag"
    if flag_col in df.columns:
        failed = df[flag_col].astype(str).str.lower().isin(("true", "1"))
        df = df[~failed]

    # Select before renaming: with FLUX_COLUMN = psfDiffFlux, renaming first would
    # collide with the psfFlux column that is also present, yielding duplicates.
    return (df[["band", flux_col, err_col, time_col]]
            .rename(columns={flux_col: "psfFlux", err_col: "psfFluxErr",
                             time_col: "expMidptMJD"}))


def photometric_features(df):
    """
    Colours and amplitude — the physical information the fold throws away.

    Each band of the fold is normalised to zero mean and unit variance, so the
    absolute flux level, and with it the colour, is gone by the time the CNN sees
    anything. Colours place a star in the HR diagram (LPV red giants at g-r ~ 2.8,
    AGN flat and blue at ~0.2, RR Lyrae in the instability strip at ~0.4), which the
    light-curve shape cannot express.

    Returns NaN for any colour whose bands are missing; the caller fills and flags.
    """
    pos = df[df["psfFlux"] > 0]
    med = pos.groupby("band")["psfFlux"].median()

    def colour(a, b):
        if a in med and b in med and med[a] > 0 and med[b] > 0:
            return float(-2.5 * np.log10(med[a] / med[b]))
        return np.nan

    r_band = pos.loc[pos["band"] == "r", "psfFlux"].to_numpy()
    amp = np.nan
    if len(r_band) >= 10:
        lo, hi = np.percentile(r_band, [5, 95])
        if lo > 0:
            amp = float(2.5 * np.log10(hi / lo))   # magnitudes, 5-95 percentile range

    return {"colour_ug": colour("u", "g"), "colour_gr": colour("g", "r"),
            "colour_ri": colour("r", "i"), "amplitude_r": amp}


def sampling_features(df):
    """
    How well the cadence constrains the period — a property of the sampling, not of
    the star.

    Checked against Gaia DR3 periods, the LS period of RR Lyrae is right for 16% of
    objects observed on <= 8 nights and for 55% of those on 18-23 nights; when a
    single night holds more than 35% of the points it is right for 18%. The fold of
    a wrong period is noise, and without these numbers the network cannot tell.

    A night runs from MJD x.5 to x+1.5: the boundary falls at 12h UTC, i.e. in
    Chilean daytime, so no observing night is ever split.
    """
    t = df["expMidptMJD"].to_numpy()
    nights, counts = np.unique(np.floor(t - 0.5), return_counts=True)
    return {"n_nights": len(nights),
            "span_days": float(t.max() - t.min()),
            "top_night_frac": float(counts.max() / len(t))}


def process_one(fpath):
    obj_id = os.path.basename(fpath).replace("lc_", "").replace(".csv", "")
    out_npy = os.path.join(C.FOLDED_DIR, f"folded_{obj_id}.npy")

    try:
        df = read_lightcurve(fpath)
        df = df.dropna(subset=["psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df[df["psfFluxErr"] > 0]
    except Exception:
        return None

    if len(df) < _MIN_POINTS:
        return None

    res = lomb_scargle_period(df)
    if res is None:
        return None
    period, power, bands_used = res
    np.save(out_npy, fold_curve(df, period, bands_used))
    return {
        "diaObjectId": obj_id,
        "period_days": period,
        "ls_power": power,
        "bands_used": ",".join(bands_used),
        "n_points": len(df),
        **photometric_features(df),
        **sampling_features(df),
    }


def run(min_points=None, min_per_band=None, workers=None, force=False):
    min_points = C.MIN_POINTS if min_points is None else min_points
    min_per_band = C.MIN_POINTS_PER_BAND if min_per_band is None else min_per_band
    C.ensure_dirs(C.FOLDED_DIR, C.CATALOG_DIR)

    files = sorted(glob.glob(os.path.join(C.LC_DIR, "lc_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No light curves in {C.LC_DIR}. Download them first "
            f"(rsp/download_lc.py runs on the Rubin Science Platform)."
        )
    print(f"Light curves found : {len(files):,}")
    print(f"Quality gate       : >= {min_points} points, >= {min_per_band} per band")

    # Resume: the period of an already-folded object cannot be recovered from the
    # .npy, so anything missing from periods.csv is redone even if the fold exists.
    done = set()
    if os.path.exists(C.PERIODS_CSV) and not force:
        done = set(pd.read_csv(C.PERIODS_CSV, dtype={"diaObjectId": str})["diaObjectId"])
        files = [f for f in files
                 if os.path.basename(f)[3:-4] not in done]
        print(f"Resume             : {len(done):,} already done, {len(files):,} to go")

    if not files:
        print("Nothing to do.")
        return

    n_workers = workers or max(1, cpu_count() - 1)
    print(f"CPU workers        : {n_workers}\n")

    results = []
    with Pool(n_workers, initializer=_init_worker,
              initargs=(min_points, min_per_band)) as pool:
        for res in tqdm(pool.imap_unordered(process_one, files, chunksize=16),
                        total=len(files), desc="Folding"):
            if res is not None:
                results.append(res)

    if results:
        out = pd.DataFrame(results)
        if os.path.exists(C.PERIODS_CSV) and not force:
            prev = pd.read_csv(C.PERIODS_CSV, dtype={"diaObjectId": str})
            out = pd.concat([prev, out], ignore_index=True)
            out = out.drop_duplicates("diaObjectId", keep="last")
        out.to_csv(C.PERIODS_CSV, index=False)

    n_rejected = len(files) - len(results)
    print(f"\nFolded    : {len(results):,}")
    print(f"Rejected  : {n_rejected:,}  (below the quality gate)")
    print(f"Periods   : {C.PERIODS_CSV}")

    if os.path.exists(C.PERIODS_CSV):
        p = pd.read_csv(C.PERIODS_CSV)
        print(f"\nTotal in periods.csv: {len(p):,}")
        print("ls_power distribution:")
        for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
            n = int((p["ls_power"] >= thr).sum())
            print(f"  >= {thr}: {n:,} ({n / len(p) * 100:.1f}%)")
