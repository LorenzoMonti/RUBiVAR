"""
One classifier on the classical features (features.py), scored on held-out sky fields.

Classes: C.TARGET_CLASSES plus C.OTHER_CLASS, which collects every other trusted label
(see config, "OTHER"). Two products come out of the same model:

  pred_class — the most probable class, flagged low-confidence below C.CONF_KEEP
  is_rr      — P(RR) above a max-F1 threshold, the RR Lyrae candidate list

Evaluation protocol:

  1. whole sky fields are dealt into C.N_FIELD_FOLDS folds (catalog.assign_fields);
     each fold is predicted by a model trained on the other folds only. These
     out-of-fold predictions on unseen fields are the numbers to quote. Every
     labelled object is scored, OTHER included: that is what the real catalogue holds.
  2. the RR threshold for fold k is chosen (max F1) on the other folds' OOF
     predictions, so even P/R at threshold never use fold k's labels.
  3. the final model is fitted on every trusted label and applied to all objects.

A random split would keep every field on both sides and reward recognising the field:
see config, "Sky fields".
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)

import catalog
import config as C
import features


def load_nights(ids):
    """Observing nights per object, from the fold stage's periods.csv."""
    per = pd.read_csv(C.PERIODS_CSV, dtype={"diaObjectId": str})
    if "n_nights" not in per.columns:
        raise KeyError("periods.csv has no `n_nights` — rerun the fold stage (--force).")
    nmap = dict(zip(per["diaObjectId"], per["n_nights"]))
    return np.array([nmap.get(o, 0) for o in ids])


def _night_bins(nights):
    """[(label, mask), ...] over C.NIGHT_BINS."""
    edges = list(C.NIGHT_BINS) + [np.inf]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"{lo}-{int(hi) - 1}" if np.isfinite(hi) else f">={lo}"
        out.append((label, (nights >= lo) & (nights < hi)))
    return out


def _pr_at(probs, y, thr):
    pred = probs >= thr
    tp = float((pred & (y == 1)).sum())
    return tp / max(pred.sum(), 1), tp / max((y == 1).sum(), 1)


def binary_by_nights(y, probs, nights, thr):
    """Binary metrics per bin of observing nights. `thr` is a scalar or per-object array."""
    thr = np.broadcast_to(thr, probs.shape)
    rows = []
    for label, m in _night_bins(nights):
        if m.sum() == 0 or y[m].sum() == 0 or y[m].sum() == m.sum():
            continue
        p_at, r_at = _pr_at(probs[m] - thr[m], y[m], 0.0)
        rows.append({"nights": label, "n": int(m.sum()), C.POSITIVE_CLASS: int(y[m].sum()),
                     "AUC-ROC": roc_auc_score(y[m], probs[m]),
                     "AUC-PR": average_precision_score(y[m], probs[m]),
                     "AUC-PR random": y[m].mean(), "P@thr": p_at, "R@thr": r_at})
    return pd.DataFrame(rows).to_string(index=False, float_format="%.3f")


def multiclass_by_nights(y, pred, nights, classes):
    """Balanced accuracy and per-class F1 per bin of observing nights."""
    rows = []
    for label, m in _night_bins(nights):
        if m.sum() == 0:
            continue
        row = {"nights": label, "n": int(m.sum()),
               "bal_acc": balanced_accuracy_score(y[m], pred[m])}
        f1 = f1_score(y[m], pred[m], labels=range(len(classes)), average=None, zero_division=0)
        row.update({f"F1 {c}": v for c, v in zip(classes, f1)})
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False, float_format="%.3f")


def load_dataset():
    """
    Features joined to the catalog. Returns (X, names, ids, labels, folds, nights);
    labels rejected by catalog.trusted are blanked to NaN and take part in nothing.
    """
    ids, names, x = features.load()
    cat = catalog.load()
    if "field_fold" not in cat.columns:
        raise KeyError("catalog has no `field_fold` — rerun the catalog stage.")
    cat = cat.set_index("diaObjectId").reindex(ids)
    labels = cat["label"].to_numpy(dtype=object)
    ok = catalog.trusted(labels, cat["label_source"].to_numpy(dtype=object))
    labels = np.where(ok & pd.notna(labels), labels, np.nan).astype(object)
    folds = cat["field_fold"].fillna(-1).astype(int).to_numpy()
    return x, names, np.array(ids), labels, folds, load_nights(ids)


def _model(seed=C.RANDOM_SEED):
    return HistGradientBoostingClassifier(max_iter=C.HGB_MAX_ITER, learning_rate=C.HGB_LR,
                                          class_weight="balanced", random_state=seed)


def _field_cv(x, y, folds, mask):
    """Out-of-fold probabilities on held-out fields, plus the per-fold models."""
    n_out = len(np.unique(y[mask]))
    oof = np.full((len(y), n_out), np.nan)
    models = {}
    for k in range(C.N_FIELD_FOLDS):
        tr, te = mask & (folds != k), mask & (folds == k)
        if te.sum() == 0:
            continue
        m = _model().fit(x[tr], y[tr])
        # a fold can lack a class in training (no AGN in the Galactic-plane folds)
        oof[np.ix_(np.where(te)[0], m.classes_)] = m.predict_proba(x[te])
        oof[te] = np.nan_to_num(oof[te])
        models[k] = m
    return oof, models


def _max_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best = int(np.argmax(f1[:-1]))
    return float(thr[best]), prec[best], rec[best], f1[best]




def classes():
    return catalog.target_classes() + [C.OTHER_CLASS]


def encode_labels(labels, cls):
    """Class index per object; non-target labels → OTHER, unlabelled → -1."""
    return np.array([cls.index(l) if l in cls else (len(cls) - 1 if pd.notna(l) else -1)
                     for l in labels])


def run(no_plot=False):
    C.ensure_dirs(C.RESULTS_DIR, C.MODEL_DIR)
    cls = classes()
    rr = cls.index(C.POSITIVE_CLASS)
    x, names, ids, labels, folds, nights = load_dataset()
    y = encode_labels(labels, cls)
    lab = (y >= 0) & (folds >= 0)
    counts = "  ".join(f"{c}={int((y[lab] == i).sum()):,}" for i, c in enumerate(cls))
    print(f"Objects  : {len(ids):,} x {x.shape[1]} features")
    print(f"Labelled : {int(lab.sum()):,}  ({counts})\n")

    oof, _ = _field_cv(x, y, folds, lab)
    yl, pl, fl, nl = y[lab], oof[lab], folds[lab], nights[lab]

    # ── Classes: most probable class on held-out fields
    pred = pl.argmax(1)
    bacc = balanced_accuracy_score(yl, pred)
    report = classification_report(yl, pred, labels=range(len(cls)), target_names=cls,
                                   zero_division=0)
    ap_line = "  ".join(f"{c}={average_precision_score(yl == i, pl[:, i]):.3f}"
                        for i, c in enumerate(cls))
    cm = pd.DataFrame(confusion_matrix(yl, pred, labels=range(len(cls))),
                      index=cls, columns=cls).to_string()
    fold_lines = []
    for k in range(C.N_FIELD_FOLDS):
        m = fl == k
        n_k = "  ".join(f"{c}={int((yl[m] == i).sum())}" for i, c in enumerate(cls))
        fold_lines.append(f"  fold {k}: bal_acc={balanced_accuracy_score(yl[m], pred[m]):.3f}"
                          f"  ({n_k})")
    class_nights = multiclass_by_nights(yl, pred, nl, cls)

    # Where each labelled class goes, OTHER split back into its original labels.
    conf_thr = float(np.quantile(pl.max(1), 1 - C.CONF_KEEP))
    low = pl.max(1) < conf_thr
    orig = labels[lab]
    rows = []
    for c in list(cls[:-1]) + list(pd.Series(orig[yl == len(cls) - 1]).value_counts().index):
        m = orig == c
        row = {"label": c, "n": int(m.sum()), "low_conf": low[m].mean()}
        row.update({f"->{t}": int(((pred[m] == i) & ~low[m]).sum()) for i, t in enumerate(cls)})
        rows.append(row)
    where_table = pd.DataFrame(rows).to_string(index=False, float_format="%.3f")

    # ── RR candidates: P(RR) with a per-fold threshold chosen on the other folds
    y_rr, p_rr = (yl == rr).astype(int), pl[:, rr]
    thr_obj = np.full(len(yl), np.nan)
    rr_rows = []
    for k in range(C.N_FIELD_FOLDS):
        te, other = fl == k, fl != k
        thr_k = _max_f1_threshold(y_rr[other], p_rr[other])[0]
        thr_obj[te] = thr_k
        p_at, r_at = _pr_at(p_rr[te], y_rr[te], thr_k)
        rr_rows.append({"fold": k, "n": int(te.sum()), C.POSITIVE_CLASS: int(y_rr[te].sum()),
                        "AUC-PR": average_precision_score(y_rr[te], p_rr[te]),
                        "AUC-PR random": y_rr[te].mean(), "thr": thr_k,
                        "P@thr": p_at, "R@thr": r_at})
    rr_fold_table = pd.DataFrame(rr_rows).to_string(index=False, float_format="%.3f")
    auc, ap = roc_auc_score(y_rr, p_rr), average_precision_score(y_rr, p_rr)
    rr_p, rr_r = _pr_at(p_rr - thr_obj, y_rr, 0.0)
    rr_nights = binary_by_nights(y_rr, p_rr, nl, thr_obj)
    rr_thr, t_p, t_r, t_f1 = _max_f1_threshold(y_rr, p_rr)

    # ── Final model on every trusted label
    final = _model().fit(x[lab], y[lab])
    joblib.dump({"model": final, "classes": cls, "rr_threshold": rr_thr,
                 "conf_threshold": conf_thr, "names": names},
                os.path.join(C.MODEL_DIR, "classifier.joblib"))
    probs = final.predict_proba(x)
    pred_all = probs.argmax(1)
    low_all = probs.max(1) < conf_thr
    is_rr = probs[:, rr] >= rr_thr

    out = pd.DataFrame({"diaObjectId": ids, "label": labels, "field_fold": folds,
                        "n_nights": nights})
    for i, c in enumerate(cls):
        out[f"prob_{c}"] = probs[:, i]
    oof_rr = np.full(len(ids), np.nan)
    oof_rr[lab] = p_rr
    out[f"prob_{C.POSITIVE_CLASS}_oof"] = oof_rr
    out["pred_class"] = [cls[i] for i in pred_all]
    out["pred_prob"] = probs.max(1)
    out["low_conf"] = low_all.astype(int)
    out[f"is_{C.POSITIVE_CLASS.lower()}"] = is_rr.astype(int)
    out.to_csv(os.path.join(C.RESULTS_DIR, "clf_predictions.csv"), index=False)

    yield_lines = [f"  {'thr':>6} {'P_oof':>8} {'R_oof':>8} {'N pred (all)':>14}"]
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        p_t, r_t = _pr_at(p_rr, y_rr, t)
        yield_lines.append(f"  {t:>6.2f} {p_t:>8.3f} {r_t:>8.3f} "
                           f"{int((probs[:, rr] >= t).sum()):>14,}")
    pred_lines = [f"  {c:<8}: {int(((pred_all == i) & ~low_all).sum()):>9,}"
                  for i, c in enumerate(cls)]

    lines = [
        "DP2 — classifier on classical features + dm-dt (gradient-boosted trees)", "=" * 70, "",
        f"Objects  : {len(ids):,} x {x.shape[1]} features",
        f"Classes  : {', '.join(cls)}   ({C.OTHER_CLASS} = every other trusted label)",
        f"Labelled : {int(lab.sum()):,}  ({counts})", "",
        "HELD-OUT FIELDS — quote these. Every labelled object is scored, OTHER included.", "",
        f"── {C.POSITIVE_CLASS} candidates: P({C.POSITIVE_CLASS}) ≥ threshold",
        f"  AUC-ROC = {auc:.4f}   AUC-PR = {ap:.4f}   (random {y_rr.mean():.3f})",
        f"  at per-fold thresholds (chosen on the other folds): P={rr_p:.3f}  R={rr_r:.3f}", "",
        "By fold:", rr_fold_table, "",
        "By observing nights:", rr_nights, "",
        f"Final threshold (max F1 on all OOF): {rr_thr:.4f}  "
        f"OOF P={t_p:.3f} R={t_r:.3f} F1={t_f1:.3f}",
        "Precision / recall (OOF) and yield at fixed thresholds:", *yield_lines, "",
        "── Classes: most probable class",
        f"  balanced accuracy = {bacc:.4f}", f"  AUC-PR per class: {ap_line}", "", report,
        "Confusion matrix (rows true, cols predicted):", cm, "",
        "By fold:", *fold_lines, "",
        "By observing nights:", class_nights, "",
        f"Low-confidence: top probability < {conf_thr:.3f} "
        f"(keeps {C.CONF_KEEP:.0%} of labelled objects)",
        f"Where each label goes (OTHER split into its original labels; counts exclude "
        f"low-confidence):", where_table, "",
        f"── Final model, all {len(ids):,} objects",
        f"  {C.POSITIVE_CLASS} candidates (P ≥ {rr_thr:.3f}): {int(is_rr.sum()):,}",
        f"  low-confidence: {int(low_all.sum()):,}", "  most probable class:", *pred_lines,
    ]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(C.RESULTS_DIR, "clf_eval.txt"), "w") as fh:
        fh.write(text)
    print(f"\nReport → {os.path.join(C.RESULTS_DIR, 'clf_eval.txt')}")

    if not no_plot:
        _plot_rr(y_rr, p_rr, fl, probs[:, rr], rr_thr,
                 os.path.join(C.RESULTS_DIR, "clf_rr_diagnostics.png"))


def _plot_rr(y, p, folds, prob_all, thr, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for k in range(C.N_FIELD_FOLDS):
        m = folds == k
        if y[m].sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(y[m], p[m])
        ax.plot(rc, pr, lw=1, alpha=0.6,
                label=f"fold {k} (AP={average_precision_score(y[m], p[m]):.2f})")
    pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, "k", lw=2.5,
            label=f"all held-out fields (AP={average_precision_score(y, p):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{C.POSITIVE_CLASS}: precision-recall on held-out fields")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(prob_all, bins=60, color="#adb5bd", log=True, label="all objects (final model)")
    ax.hist(p[y == 1], bins=60, color=C.CLASS_COLORS.get(C.POSITIVE_CLASS, "#e63946"),
            alpha=0.85, log=True, label=f"{C.POSITIVE_CLASS}, out-of-fold")
    ax.axvline(thr, ls="--", c="k", lw=1, label=f"threshold {thr:.3f}")
    ax.set_xlabel(f"P({C.POSITIVE_CLASS})")
    ax.set_ylabel("count (log)")
    ax.set_title("Score distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {out_path}")
