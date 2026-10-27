"""
STEP 3c — Lomb-Scargle + phase-fold for lc_xmatch objects (>= 10 points).

Output:
  folded_xmatch/folded_<id>.npy   (6, 100) float32
  periods/periods_xmatch.csv      diaObjectId, period_days, ls_power, bands_used, n_points
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from step3_lomb_scargle import BANDS, BAND_IDX, N_PHASE, phase_fold_band, normalize_band

BASE             = os.path.dirname(os.path.abspath(__file__))
LC_XMATCH_DIR    = os.path.join(BASE, "lc_xmatch")
FOLDED_XMATCH    = os.path.join(BASE, "folded_xmatch")
PERIODS_XMATCH   = os.path.join(BASE, "periods", "periods_xmatch.csv")
CATALOG_CSV      = os.path.join(BASE, "crossmatch", "catalogue_labeled.csv")

MIN_POINTS = 10
FREQ_MIN   = 1.0 / 100.0
FREQ_MAX   = 24.0
N_FREQ     = 100_000


def get_target_ids():
    lab = pd.read_csv(CATALOG_CSV, dtype={"diaObjectId": str})
    return set(lab.loc[lab["lc_source"] == "xmatch", "diaObjectId"])


def process_one(obj_id):
    out_npy     = os.path.join(FOLDED_XMATCH, f"folded_{obj_id}.npy")
    fold_exists = os.path.exists(out_npy)

    fpath = os.path.join(LC_XMATCH_DIR, f"lc_{obj_id}.csv")
    try:
        df = pd.read_csv(fpath, usecols=["band", "psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df.dropna(subset=["psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df[df["psfFluxErr"] > 0]
    except Exception:
        return None

    if len(df) < MIN_POINTS:
        return None

    times_all, flux_all, err_all, bands_used = [], [], [], []
    for b in BANDS:
        sub = df[df["band"] == b]
        if len(sub) < 3:
            continue
        bands_used.append(b)
        t = sub["expMidptMJD"].values
        f = sub["psfFlux"].values - sub["psfFlux"].mean()
        e = sub["psfFluxErr"].values
        times_all.append(t)
        flux_all.append(f)
        err_all.append(e)

    if not bands_used:
        return None

    t_cat = np.concatenate(times_all)
    f_cat = np.concatenate(flux_all)
    e_cat = np.concatenate(err_all)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls    = LombScargle(t_cat, f_cat, e_cat, normalization="standard")
        freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)
        power = ls.power(freqs)

    best_idx = np.argmax(power)
    period   = 1.0 / freqs[best_idx]
    ls_power = float(power[best_idx])

    if not fold_exists:
        folded = np.full((6, N_PHASE), np.nan)
        for b in bands_used:
            sub = df[df["band"] == b]
            raw = phase_fold_band(sub["expMidptMJD"].values,
                                  sub["psfFlux"].values,
                                  sub["psfFluxErr"].values, period)
            folded[BAND_IDX[b]] = normalize_band(raw)
        np.save(out_npy, folded.astype(np.float32))

    return {
        "diaObjectId": obj_id,
        "period_days": period,
        "ls_power":    ls_power,
        "bands_used":  ",".join(bands_used),
        "n_points":    len(df),
        "fold_new":    not fold_exists,
    }


def main():

    os.makedirs(FOLDED_XMATCH, exist_ok=True)
    os.makedirs(os.path.dirname(PERIODS_XMATCH), exist_ok=True)

    target_ids = get_target_ids()
    print(f"Target xmatch objects (lc_source='xmatch'): {len(target_ids)}")

    files = [os.path.join(LC_XMATCH_DIR, f"lc_{i}.csv")
             for i in target_ids
             if os.path.exists(os.path.join(LC_XMATCH_DIR, f"lc_{i}.csv"))]
    print(f"LC files found: {len(files)}")

    already = len(glob.glob(os.path.join(FOLDED_XMATCH, "folded_*.npy")))
    if already:
        print(f"Resume: {already} already folded, skipping.")

    obj_ids = [os.path.basename(f).replace("lc_", "").replace(".csv", "") for f in files]

    n_workers = max(1, cpu_count() - 1)
    print(f"CPU workers: {n_workers}\n")

    results = []
    with Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_one, obj_ids), total=len(obj_ids)):
            if res is not None:
                results.append(res)

    fold_new = sum(1 for r in results if r.get("fold_new"))
    print(f"\nNew folds saved : {fold_new}")
    print(f"Already existing: {len(results) - fold_new}")
    print(f"Skipped (< {MIN_POINTS} pts or no data): {len(obj_ids) - len(results)}")

    if results:
        periods_df = pd.DataFrame([
            {k: r[k] for k in ("diaObjectId", "period_days", "ls_power", "bands_used", "n_points")}
            for r in results
        ]).sort_values("diaObjectId")
        periods_df.to_csv(PERIODS_XMATCH, index=False)
        print(f"Periods saved: {PERIODS_XMATCH}  ({len(periods_df)} objects)")

        print(f"\nls_power distribution:")
        for thr in [0.1, 0.2, 0.3, 0.5]:
            n = (periods_df["ls_power"] >= thr).sum()
            print(f"  >= {thr}: {n}  ({100*n/len(periods_df):.0f}%)")


if __name__ == "__main__":
    main()
