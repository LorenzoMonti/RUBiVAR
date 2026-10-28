"""
How often is the Lomb-Scargle period right? Labelled RR Lyrae against their Gaia DR3
period (gaiadr3.vari_rrlyrae: pf for RRab/RRd, p1_o for RRc), broken down by observing
nights — the reason the benchmark prefers representations that need no period.

The Gaia periods are public: they are fetched once from the Gaia archive (astroquery)
into C.GAIA_RR_PERIODS_CSV. A period is "right" within PERIOD_TOL, a half / double
alias within the same relative tolerance.
"""

import os

import numpy as np
import pandas as pd

import catalog
import config as C

PERIOD_TOL = 0.02
CHUNK = 400   # source_ids per ADQL query


def fetch():
    """Gaia DR3 RR Lyrae periods of the labelled RR, cached in C.GAIA_RR_PERIODS_CSV."""
    if os.path.exists(C.GAIA_RR_PERIODS_CSV):
        return pd.read_csv(C.GAIA_RR_PERIODS_CSV, dtype={"source_id": str})
    from astroquery.gaia import Gaia
    cat = catalog.load()
    sid = cat.loc[cat["label"] == C.POSITIVE_CLASS, "source_id"].dropna().astype("int64").unique()
    out = []
    for i in range(0, len(sid), CHUNK):
        ids = ",".join(map(str, sid[i:i + CHUNK]))
        job = Gaia.launch_job("SELECT source_id, pf, p1_o, best_classification "
                              f"FROM gaiadr3.vari_rrlyrae WHERE source_id IN ({ids})")
        out.append(job.get_results().to_pandas())
    g = pd.concat(out)
    g.columns = [c.lower() for c in g.columns]
    g.to_csv(C.GAIA_RR_PERIODS_CSV, index=False)
    return pd.read_csv(C.GAIA_RR_PERIODS_CSV, dtype={"source_id": str})


def run():
    from classify import load_dataset
    _, _, ids, labels, folds, nights = load_dataset()
    g = fetch().set_index("source_id")
    cat = catalog.load().set_index("diaObjectId").reindex(ids)
    per = pd.read_csv(C.PERIODS_CSV, dtype={"diaObjectId": str}).set_index("diaObjectId")
    p_gaia = cat["source_id"].astype("Int64").astype(str).map(g["pf"].fillna(g["p1_o"]))
    ratio = per.reindex(ids)["period_days"].to_numpy(float) / p_gaia.to_numpy(float)
    right = np.abs(ratio - 1) < PERIOD_TOL
    alias = (np.abs(ratio - 0.5) < PERIOD_TOL / 2) | (np.abs(ratio - 2) < 2 * PERIOD_TOL)
    rr = (labels == C.POSITIVE_CLASS) & (folds >= 0) & np.isfinite(ratio)
    edges = list(C.NIGHT_BINS) + [np.inf]
    rows = [("all", rr)] + [(f"{lo}-{hi - 1:g}" if np.isfinite(hi) else f">={lo}",
                             rr & (nights >= lo) & (nights < hi))
                            for lo, hi in zip(edges[:-1], edges[1:])]
    print(f"LS period of labelled RR vs Gaia DR3 (|P_LS / P_Gaia - 1| < {PERIOD_TOL:.0%})")
    print(f"  {'nights':<8} {'RR':>6} {'right':>7} {'half/double':>12}")
    for name, m in rows:
        print(f"  {name:<8} {m.sum():6d} {right[m].mean():7.0%} {alias[m].mean():12.0%}")
