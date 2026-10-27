"""
Benchmark report: every scored model under the same protocol (see config, "Benchmark").

  1. held-out fields: mean ± std over seeds; by fold, observing nights and points per
     night on the seed-averaged probabilities; paired field bootstrap vs the reference
  2. the same objects (>= C.DEG_MIN_NIGHTS nights) cut to their first n nights
  3. learning curve: fraction of the training rows → held-out RR AUC-PR / bal_acc
"""

import os

import numpy as np
import pandas as pd

import config as C
from bench import degrade, protocol, run

PPN_BINS = [0, 5, 10, 20, 40, np.inf]   # points per night


def _pair(m):
    return f"{m['RR AUC-PR']:.3f} / {m['bal_acc']:.3f}"


def _points_per_night(d):
    per = pd.read_csv(C.PERIODS_CSV, dtype={"diaObjectId": str}).set_index("diaObjectId")
    per = per.reindex(d.ids)
    return (per["n_points"] / per["n_nights"].clip(lower=1)).to_numpy()


def report(names, reference="hgb"):
    d = protocol.load()
    rows = np.where(d.lab)[0]
    y, cls = d.y, d.cls
    rr = cls.index(C.POSITIVE_CLASS)
    _, deg_rows, deg_n = degrade.load()
    oof = {n: run.load_oof(n, d, len(deg_rows)) for n in names}
    mean = {n: (np.mean([p for p, _ in o.values()], 0), np.mean([q for _, q in o.values()], 0))
            for n, o in oof.items()}
    lines = [f"Benchmark — {len(rows):,} labelled objects in {len(np.unique(d.fields[rows]))} "
             f"fields, scored on held-out fields", ""]

    # 1. held-out fields
    t = {}
    for n, o in oof.items():
        per_seed = [protocol.metrics(y[rows], p[rows], cls) for p, _ in o.values()]
        t[f"{n} ({len(o)} seeds)"] = {
            k: f"{np.mean([m[k] for m in per_seed]):.3f} ± {np.std([m[k] for m in per_seed]):.3f}"
            for k in per_seed[0]}
    lines += ["1. Held-out fields, mean ± std over seeds", pd.DataFrame(t).T.to_string(), ""]

    ppn = _points_per_night(d)
    edges = list(C.NIGHT_BINS) + [np.inf]
    splits = {
        "fold": [(f"fold {k}", rows[d.folds[rows] == k]) for k in range(C.N_FIELD_FOLDS)],
        "observing nights": [(f"{lo}-{hi - 1:g}" if np.isfinite(hi) else f">={lo}",
                              rows[(d.nights[rows] >= lo) & (d.nights[rows] < hi)])
                             for lo, hi in zip(edges[:-1], edges[1:])],
        "points per night": [(f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}",
                              rows[(ppn[rows] >= lo) & (ppn[rows] < hi)])
                             for lo, hi in zip(PPN_BINS[:-1], PPN_BINS[1:])]}
    for title, groups in splits.items():
        t = {g: {"n": len(r), "RR": int((y[r] == rr).sum()),
                 **{n: _pair(protocol.metrics(y[r], p[r], cls)) for n, (p, _) in mean.items()}}
             for g, r in groups if len(r)}
        lines += [f"   by {title} (RR AUC-PR / bal_acc)", pd.DataFrame(t).T.to_string(), ""]

    if reference in mean and len(mean) > 1:
        lines.append(f"   paired field bootstrap, metric(model) - metric({reference}):")
        for n in names:
            if n != reference:
                b = protocol.field_bootstrap(y[rows], mean[reference][0][rows],
                                             mean[n][0][rows], d.fields[rows], cls)
                lines += [f"     {n:<12} {k:<10} {m:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]"
                          for k, (m, lo, hi) in b.items()]
        lines.append("")

    # 2. first n nights
    base = np.unique(deg_rows)
    t = {"all nights": {n: _pair(protocol.metrics(y[base], p[base], cls))
                        for n, (p, _) in mean.items()}}
    for lev in sorted(C.DEG_LEVELS, reverse=True):
        i = np.where(deg_n == lev)[0]
        t[f"first {lev}"] = {n: _pair(protocol.metrics(y[deg_rows[i]], q[i], cls))
                             for n, (_, q) in mean.items()}
    lines += [f"2. The same {len(base):,} objects (>= {C.DEG_MIN_NIGHTS} nights, "
              f"RR {int((y[base] == rr).sum())}) cut to their first n nights "
              f"(RR AUC-PR / bal_acc)", pd.DataFrame(t).T.to_string(), ""]

    # 3. learning curve
    t = {}
    for n in names:
        for f, p in sorted(run.load_lc(n, d).items()):
            t.setdefault(f"{f:.0%}", {})[n] = _pair(protocol.metrics(y[rows], p[rows], cls))
    lines += ["3. Learning curve, seed 0 (RR AUC-PR / bal_acc)",
              pd.DataFrame(t).T.to_string(), ""]

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(C.BENCH_DIR, "report.txt"), "w") as fh:
        fh.write(text)
