"""
Build the unified period lookup table used by step4 and step5.

Priority per object:
  1. Gaia period  (accurate — from gaia_periods.csv)
  2. LS period    (fallback — from periods_strict / gentle_extra / xmatch CSVs)

Also resolves which folded directory to load for each object:
  - RR / ECL / Ceph labeled → folded_labeled/   (folded with Gaia period)
  - LPV labeled             → folded_strict/ or folded_gentle_extra/ or folded_xmatch/
  - Unlabeled               → folded_strict/

Output: periods/periods_final.csv
Columns: diaObjectId, period_days, period_source, label, lc_source, fold_dir
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load period sources ────────────────────────────────────────────────────────
gaia_per  = pd.read_csv(f"{BASE}/periods/periods_gaia_labeled.csv",
                        dtype={"diaObjectId": str})

ls_strict = pd.read_csv(f"{BASE}/periods/periods_strict.csv",
                        dtype={"diaObjectId": str})
ls_gentle = pd.read_csv(f"{BASE}/periods/periods_gentle_extra.csv",
                        dtype={"diaObjectId": str})
ls_xmatch = pd.read_csv(f"{BASE}/periods/periods_xmatch.csv",
                        dtype={"diaObjectId": str})

ls_all = pd.concat([ls_strict, ls_gentle, ls_xmatch], ignore_index=True)
ls_all = ls_all.drop_duplicates("diaObjectId", keep="first")
ls_power_map = dict(zip(ls_all["diaObjectId"], ls_all["ls_power"]))

# Also load full Gaia periods to supplement labeled objects missing from
# periods_gaia_labeled (e.g. LPV not folded due to long period)
lab    = pd.read_csv(f"{BASE}/crossmatch/catalogue_labeled.csv",
                     dtype={"diaObjectId": str, "source_id": str})
gaia_raw = pd.read_csv(f"{BASE}/gaia_periods.csv", dtype={"source_id": str})
lab_gaia = lab.merge(gaia_raw[["source_id", "period_days"]], on="source_id", how="left")
lab_gaia = lab_gaia[lab_gaia["period_days"].notna()][["diaObjectId", "period_days"]]

# ── Gaia-folded objects (RR/ECL/Ceph in folded_labeled/) ──────────────────────
FOLDED_LABELED_DIR = os.path.join(BASE, "folded_labeled")
gaia_folded_ids = {
    f.replace("folded_", "").replace(".npy", "")
    for f in os.listdir(FOLDED_LABELED_DIR)
    if f.endswith(".npy")
}

# ── Build unified table for labeled objects ───────────────────────────────────
records = []
for _, row in lab.iterrows():
    oid  = row["diaObjectId"]
    lbl  = row["label"]
    lcsrc = row.get("lc_source")

    # Period: always LS — consistent across all objects (fold + tabular feature).
    # Gaia periods are preserved in gaia_periods.csv for external reference.
    ls_row = ls_all[ls_all["diaObjectId"] == oid]
    if ls_row.empty:
        continue
    period = float(ls_row.iloc[0]["period_days"])

    if lcsrc == "strict":
        fold_dir = "folded_strict"
    elif lcsrc == "gentle":
        fold_dir = "folded_gentle_extra"
    elif lcsrc == "xmatch":
        fold_dir = "folded_xmatch"
    else:
        fold_dir = None

    records.append({
        "diaObjectId"  : oid,
        "period_days"  : period,
        "period_source": "ls",
        "ls_power"     : ls_power_map.get(oid, 0.0),
        "label"        : lbl,
        "lc_source"    : lcsrc,
        "fold_dir"     : fold_dir,
    })

labeled_final = pd.DataFrame(records)

# ── Append unlabeled (folded_strict, no label) ────────────────────────────────
labeled_ids = set(labeled_final["diaObjectId"])
unlabeled = ls_strict[~ls_strict["diaObjectId"].isin(labeled_ids)].copy()
unlabeled["period_source"] = "ls"
unlabeled["label"]         = None
unlabeled["lc_source"]     = "strict"
unlabeled["fold_dir"]      = "folded_strict"
unlabeled = unlabeled[["diaObjectId", "period_days", "period_source", "ls_power",
                        "label", "lc_source", "fold_dir"]]

# ── Add ALL gentle_extra objects (matches original training set) ──────────────
# The original pipeline included all 430 gentle_extra objects in the VAE training.
# 267 of them are not in catalogue_labeled but many carry anchor-class labels
# (AGN, SOLAR_LIKE, RS, YSO...) essential for SupCon repulsion between clusters.
labeled_gentle_ids = set(labeled_final[labeled_final["lc_source"] == "gentle"]["diaObjectId"])
gentle_unlabeled = ls_gentle[~ls_gentle["diaObjectId"].isin(labeled_gentle_ids)].copy()
gentle_unlabeled["period_source"] = "ls"
gentle_unlabeled["label"]         = None
gentle_unlabeled["lc_source"]     = "gentle"
gentle_unlabeled["fold_dir"]      = "folded_gentle_extra"
gentle_unlabeled = gentle_unlabeled[["diaObjectId", "period_days", "period_source", "ls_power",
                                     "label", "lc_source", "fold_dir"]]

final = pd.concat([labeled_final, unlabeled, gentle_unlabeled], ignore_index=True)
out   = f"{BASE}/periods/periods_final.csv"
final.to_csv(out, index=False)

# ── Report ────────────────────────────────────────────────────────────────────
lab_only = final[final["label"].notna()]
print(f"{'='*55}")
print(f"Unlabeled objects  : {final['label'].isna().sum():>6,}")
print(f"Labeled objects    : {len(lab_only):>6,}")
print(f"{'='*55}")
print("\nLabeled — period source:")
print(lab_only.groupby(["label","period_source"]).size().unstack(fill_value=0).to_string())
print("\nLabeled — fold directory:")
print(lab_only.groupby(["label","fold_dir"]).size().unstack(fill_value=0).to_string())
print(f"\nSaved → {out}  ({len(final):,} total rows)")
