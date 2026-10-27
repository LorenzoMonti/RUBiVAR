"""
Export Gaia periods for the labeled objects.
Run on RSP, then download gaia_periods.csv to local machine.

Logic: for each object use pf (fundamental period) if available,
       fall back to p1_o (first overtone) otherwise.
"""

import pandas as pd

GAIA_DIR = "Gaia catalogues"

# period column(s) differ by catalog:
#   rrlyrae, cepheids → pf (fundamental) + p1_o (first overtone)
#   lpv, eb           → periods
CLASS_FILES = {
    "RR"  : ("rrlyrae.csv",   ["pf", "p1_o"]),
    "Ceph": ("cepheids.csv",  ["pf", "p1_o"]),
    "LPV" : ("lpv.csv",       ["periods"]),
    "ECL" : ("eb.csv",        ["periods"]),
}

dfs = []
for label, (fname, per_cols) in CLASS_FILES.items():
    df = pd.read_csv(f"{GAIA_DIR}/{fname}", usecols=["source_id"] + per_cols)
    df["label"] = label
    if "pf" in per_cols:
        df["period_days"] = df["pf"].where(df["pf"].notna() & (df["pf"] > 0), df["p1_o"])
    else:
        df["period_days"] = df["periods"]
    df = df[df["period_days"].notna() & (df["period_days"] > 0)]
    dfs.append(df[["source_id", "label", "period_days"]])
    print(f"  {label:4s}: {len(df):6,} with valid period  "
          f"(range: {df['period_days'].min():.3f}–{df['period_days'].max():.3f} d)")

gaia_periods = pd.concat(dfs, ignore_index=True).drop_duplicates("source_id")
gaia_periods.to_csv("gaia_periods.csv", index=False)
print(f"\nSaved gaia_periods.csv  ({len(gaia_periods):,} sources)")
