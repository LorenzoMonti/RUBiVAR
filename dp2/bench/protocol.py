"""
The protocol every benchmarked model shares: data, splits, night cuts, metrics,
field bootstrap and run metadata. See config, "Benchmark".

Each model writes to C.BENCH_DIR/<model>/, one file per outer fold k
    oof_s<seed>_f<k>.npz  rows, prob: fold k's test rows and their probabilities;
                          deg_idx, deg_prob: the same on features_degraded.npz rows;
                          emb (seed 0, if the model has one): its representation
    lc_<frac>_f<k>.npz    rows, prob, training on a fraction of the training rows
so that report.py treats all models alike.
"""

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)

import catalog
import config as C
from classify import classes, encode_labels, load_dataset


@dataclass
class Data:
    x: np.ndarray        # (n, 460) classical features + dm-dt
    names: list
    ids: np.ndarray
    labels: np.ndarray   # original labels, NaN if unlabelled or untrusted
    y: np.ndarray        # class index, -1 unlabelled
    cls: list
    folds: np.ndarray    # outer field fold, -1 outside every fold
    fields: np.ndarray
    nights: np.ndarray
    lab: np.ndarray      # labelled and inside a fold: every row that is trained or scored


def load():
    x, names, ids, labels, folds, nights = load_dataset()
    cls = classes()
    y = encode_labels(labels, cls)
    fields = catalog.load().set_index("diaObjectId").reindex(ids)["field"].to_numpy()
    return Data(x, names, ids, labels, y, cls, folds, fields, nights, (y >= 0) & (folds >= 0))


def outer_split(d, k):
    """(train rows, test rows) for outer fold k."""
    return np.where(d.lab & (d.folds != k))[0], np.where(d.lab & (d.folds == k))[0]


def inner_splits(d, k):
    """
    C.BENCH_INNER (inner train, inner validation) splits of fold k's training rows. The
    fields of at most C.BENCH_VAL_MAX of those rows are dealt into C.BENCH_INNER
    groups balanced on labels (GroupKFold); each group is the validation side once, so
    every such field is validated exactly once. Larger fields (the bulge, ~half of a
    fold's training labels) always stay in training.
    """
    from sklearn.model_selection import GroupKFold
    rows = outer_split(d, k)[0]
    fields, size = np.unique(d.fields[rows], return_counts=True)
    small = np.isin(d.fields[rows], fields[size <= C.BENCH_VAL_MAX * len(rows)])
    cand = rows[small]
    return [(np.setdiff1d(rows, cand[va]), cand[va])
            for _, va in GroupKFold(C.BENCH_INNER).split(cand, groups=d.fields[cand])]


def inner_split(d, k):
    """The first inner split: early stopping for the final fit of sequence models."""
    return inner_splits(d, k)[0]


def night_of(t):
    """Observing night of an MJD: nights change at local noon in Chile, ~0.5 in MJD."""
    return np.floor(t - 0.5)


def first_nights(night, n_keep):
    """Mask keeping the curve's first n_keep observing nights."""
    un = np.unique(night)
    return night <= un[min(n_keep, len(un)) - 1] if len(un) else np.zeros(0, bool)


def metrics(y, p, cls):
    """RR AUC-PR / AUC-ROC, balanced accuracy and per-class F1 of probabilities p."""
    rr = cls.index(C.POSITIVE_CLASS)
    pred = p.argmax(1)
    f1 = f1_score(y, pred, labels=range(len(cls)), average=None, zero_division=0)
    pos = y == rr
    return {"RR AUC-PR": average_precision_score(pos, p[:, rr]),
            "RR AUC-ROC": roc_auc_score(pos, p[:, rr]) if 0 < pos.sum() < len(y) else np.nan,
            "bal_acc": balanced_accuracy_score(y, pred),
            **{f"F1 {c}": v for c, v in zip(cls, f1)}}


def field_bootstrap(y, p_a, p_b, fields, cls, n=1000, seed=C.RANDOM_SEED):
    """Paired bootstrap over whole fields of metric(b) - metric(a): {metric: (mean, lo, hi)}."""
    rng = np.random.default_rng(seed)
    uf = np.unique(fields)
    rows = {f: np.where(fields == f)[0] for f in uf}
    keys = ("RR AUC-PR", "bal_acc")
    diffs = {k: [] for k in keys}
    for _ in range(n):
        r = np.concatenate([rows[f] for f in rng.choice(uf, len(uf))])
        if not (y[r] == cls.index(C.POSITIVE_CLASS)).any():
            continue
        a, b = metrics(y[r], p_a[r], cls), metrics(y[r], p_b[r], cls)
        for k in keys:
            diffs[k].append(b[k] - a[k])
    return {k: (np.mean(v), *np.percentile(v, [2.5, 97.5])) for k, v in diffs.items()}


def model_dir(name):
    path = os.path.join(C.BENCH_DIR, name)
    C.ensure_dirs(path)
    return path


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_meta(**extra):
    """What is needed to rerun a result: code version, inputs, packages, settings."""
    import sklearn
    git = lambda *a: subprocess.run(["git", "-C", C.BASE, *a], capture_output=True,
                                    text=True).stdout.strip()
    meta = {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", ".")),
            "inputs": {os.path.basename(p): _sha(p) for p in
                       (C.FEATURES_NPZ, C.LABELED_CSV, C.FEATURES_DEG_NPZ)
                       if os.path.exists(p)},
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__}
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    except ImportError:
        pass
    return {**meta, **extra}


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
