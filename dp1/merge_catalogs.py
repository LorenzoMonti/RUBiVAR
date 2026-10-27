"""
Merge labeled catalogs into a single ground-truth file.

Sources (in priority order):
  1. catalogue_lsst_sos.csv      — Gaia SOS, most reliable
  2. gaia_lsst_xmatch.csv        — positional Gaia → DiaObject match
  3. catalogue_lsst_classifier.csv — Gaia classifier (lower confidence)

Output: crossmatch/catalogue_labeled.csv
Columns: diaObjectId, source_id, label, label_source, ra, dec
"""

import os
import pandas as pd

BASE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crossmatch")
OUT     = f"{BASE}/catalogue_labeled.csv"

# ── 1. Load ───────────────────────────────────────────────────────────────────
sos  = pd.read_csv(f"{BASE}/catalogue_lsst_sos.csv")
xmat = pd.read_csv(f"{BASE}/gaia_lsst_xmatch.csv")
clf  = pd.read_csv(f"{BASE}/catalogue_lsst_classifier.csv")

# ── 2. Normalise SOS → label ──────────────────────────────────────────────────
# Some objects carry multiple SOS flags; priority: RR > Ceph > ECL > LPV
SOS_COLS = {
    "in_vari_rrlyrae"            : "RR",
    "in_vari_cepheid"            : "Ceph",
    "in_vari_eclipsing_binary"   : "ECL",
    "in_vari_long_period_variable": "LPV",
}
for col in SOS_COLS:
    sos[col] = sos[col].astype(str).str.lower() == "true"

def sos_label(row):
    for col, lbl in SOS_COLS.items():
        if row[col]:
            return lbl
    return None

sos["label"] = sos.apply(sos_label, axis=1)
sos4 = (sos[sos["label"].notna()]
        [["lsst_id", "source_id", "label", "ra", "dec"]]
        .rename(columns={"lsst_id": "diaObjectId"})
        .copy())
sos4["label_source"] = "SOS"
print(f"SOS    : {len(sos4):3d} objects  {sos4['label'].value_counts().to_dict()}")

# ── 3. Normalise xmatch → label ───────────────────────────────────────────────
xmat4 = (xmat[["diaObjectId", "source_id", "label", "lsst_ra", "lsst_dec"]]
         .rename(columns={"lsst_ra": "ra", "lsst_dec": "dec"})
         .copy())
xmat4["label_source"] = "xmatch"
print(f"xmatch : {len(xmat4):3d} objects  {xmat4['label'].value_counts().to_dict()}")

# ── 4. Normalise classifier → label (4 classes only) ─────────────────────────
CLF_MAP = {"RR": "RR", "CEP": "Ceph", "LPV": "LPV", "ECL": "ECL"}
clf4 = (clf[clf["best_class_name"].isin(CLF_MAP)]
        .copy()
        .assign(label=lambda d: d["best_class_name"].map(CLF_MAP),
                label_source="classifier")
        [["lsst_id", "source_id", "label", "ra", "dec", "label_source",
          "best_class_score"]]
        .rename(columns={"lsst_id": "diaObjectId"}))
print(f"CLF    : {len(clf4):3d} objects  {clf4['label'].value_counts().to_dict()}")

# ── 5. Merge with priority SOS > xmatch > classifier ─────────────────────────
# Stack all three, then keep the first occurrence per diaObjectId
# (priority is guaranteed by concat order + drop_duplicates keep='first')
merged = pd.concat(
    [sos4[["diaObjectId","source_id","label","label_source","ra","dec"]],
     xmat4[["diaObjectId","source_id","label","label_source","ra","dec"]],
     clf4[["diaObjectId","source_id","label","label_source","ra","dec"]]],
    ignore_index=True
)

# Primary dedup key: diaObjectId
merged = merged.drop_duplicates("diaObjectId", keep="first")

# Secondary: some entries may share source_id but different diaObjectId
# (e.g. xmatch found a different DIA counterpart than SOS); keep SOS match
merged = merged.drop_duplicates("source_id", keep="first")

merged = merged.reset_index(drop=True)

# ── 6. Stats ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Final catalog: {len(merged)} unique objects")
print("\nBy class:")
print(merged["label"].value_counts().to_string())
print("\nBy source:")
print(merged["label_source"].value_counts().to_string())
print("\nClass × source:")
print(pd.crosstab(merged["label"], merged["label_source"]))

# ── 7. Save ───────────────────────────────────────────────────────────────────
merged.to_csv(OUT, index=False)
print(f"\nSaved → {OUT}")
