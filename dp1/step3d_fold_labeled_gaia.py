"""
STEP 3d — Phase-fold all labeled objects using Gaia periods.

Replaces LS-computed periods for labeled objects with the more accurate
Gaia periods.  Folds are saved to folded_labeled/ and override any
previously computed LS-based folds for these objects.

Prerequisites:
  - gaia_periods.csv downloaded from RSP (run export_gaia_periods.py first)
  - LCs must be available in lc_strict/, lc_gentle/, or lc_xmatch/

Output:
  folded_labeled/folded_<id>.npy   (6, 100) float32
  periods/periods_gaia_labeled.csv diaObjectId, source_id, label, period_days, lc_source
"""

import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from step3_lomb_scargle import BANDS, BAND_IDX, N_PHASE, phase_fold_band, normalize_band

BASE           = os.path.dirname(os.path.abspath(__file__))
FOLDED_DIR     = os.path.join(BASE, "folded_labeled")
PERIODS_OUT    = os.path.join(BASE, "periods", "periods_gaia_labeled.csv")
CATALOG_CSV    = os.path.join(BASE, "crossmatch", "catalogue_labeled.csv")
GAIA_PER_CSV   = os.path.join(BASE, "gaia_periods.csv")

LC_DIRS = {
    "strict": os.path.join(BASE, "lc_strict"),
    "gentle": os.path.join(BASE, "lc_gentle"),
    "xmatch": os.path.join(BASE, "lc_xmatch"),
}


def build_work_list():
    lab  = pd.read_csv(CATALOG_CSV, dtype={"diaObjectId": str, "source_id": str})
    gper = pd.read_csv(GAIA_PER_CSV, dtype={"source_id": str})

    merged = lab.merge(gper[["source_id", "period_days"]], on="source_id", how="inner")
    missing = len(lab) - len(merged)
    if missing:
        print(f"  Warning: {missing} labeled objects have no Gaia period — skipped.")

    return merged[["diaObjectId", "source_id", "label", "lc_source", "period_days"]]


def lc_path(row):
    src = row["lc_source"]
    if src not in LC_DIRS:
        return None
    p = os.path.join(LC_DIRS[src], f"lc_{row['diaObjectId']}.csv")
    return p if os.path.exists(p) else None


def process_one(args):
    obj_id, period, fpath = args
    out_npy = os.path.join(FOLDED_DIR, f"folded_{obj_id}.npy")

    try:
        df = pd.read_csv(fpath, usecols=["band", "psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df.dropna(subset=["psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df[df["psfFluxErr"] > 0]
    except Exception:
        return None

    if len(df) < 5:
        return None

    folded = np.full((6, N_PHASE), np.nan)
    bands_used = []
    for b in BANDS:
        sub = df[df["band"] == b]
        if len(sub) < 3:
            continue
        raw = phase_fold_band(sub["expMidptMJD"].values,
                              sub["psfFlux"].values,
                              sub["psfFluxErr"].values, period)
        norm = normalize_band(raw)
        if not np.all(np.isnan(norm)):
            folded[BAND_IDX[b]] = norm
            bands_used.append(b)

    if not bands_used:
        return None

    np.save(out_npy, folded.astype(np.float32))
    return obj_id


def main():
    os.makedirs(FOLDED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PERIODS_OUT), exist_ok=True)

    work = build_work_list()
    print(f"Labeled objects with Gaia period: {len(work)}")
    print(work["label"].value_counts().to_string())
    print()

    # Build task list: (diaObjectId, period, lc_path)
    tasks = []
    missing_lc = 0
    for _, row in work.iterrows():
        p = lc_path(row)
        if p is None:
            missing_lc += 1
            continue
        tasks.append((row["diaObjectId"], row["period_days"], p))

    print(f"Tasks ready: {len(tasks)}  (skipped {missing_lc} with no LC file)\n")

    n_workers = max(1, cpu_count() - 1)
    done = []
    with Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            if res is not None:
                done.append(res)

    print(f"\nFolds saved: {len(done)} / {len(tasks)}")

    # Save period table (only successfully folded objects)
    done_set = set(done)
    out_df = work[work["diaObjectId"].isin(done_set)].copy()
    out_df.to_csv(PERIODS_OUT, index=False)
    print(f"Periods table: {PERIODS_OUT}  ({len(out_df)} objects)")

    print("\nBy class:")
    print(out_df["label"].value_counts().to_string())
    print("\nBy lc_source:")
    print(out_df["lc_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
