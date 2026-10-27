"""
Ground truth for DP2, built from the two Gaia crossmatch tables.

1. Gaia SOS is authoritative: an object carrying at least one `in_vari_*` flag takes
   its class from there, resolved through config.FLAG_PRIORITY.
2. Objects whose SOS one-hot row is all-false fall back to the Gaia classifier.
3. Anything left over keeps label = NaN (`label_source = "none"`).

Output rows are ordered SOS block first, then classifier, then unlabelled; within each
block by descending r_var_significance (objects with no value sort last).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupKFold

import config as C


def load_class_definitions():
    """Parse class_definition.txt → {class_name: description}."""
    defs = {}
    with open(C.CLASS_DEF_TXT) as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            defs[parts[2]] = parts[3].strip().strip('"')
    return defs


def build(min_clf_score=None):
    """Merge SOS + classifier into the labelled catalog. Returns the DataFrame."""
    min_clf_score = C.MIN_CLF_SCORE if min_clf_score is None else min_clf_score

    sos = pd.read_csv(C.SOS_CSV, dtype={"diaObjectId": str, "source_id": str})
    clf = pd.read_csv(C.CLF_CSV, dtype={"diaObjectId": str, "source_id": str})
    print(f"SOS        : {len(sos):,} rows")
    print(f"Classifier : {len(clf):,} rows")

    flags = [f for f, _ in C.FLAG_PRIORITY]
    missing = set(flags) - set(sos.columns)
    if missing:
        raise KeyError(f"SOS table is missing expected flags: {sorted(missing)}")
    for f in flags:
        sos[f] = sos[f].astype(str).str.lower() == "true"

    # ── SOS label: first flag that fires, in priority order ───────────────────
    sos["n_sos_flags"] = sos[flags].sum(axis=1)
    label = pd.Series(pd.NA, index=sos.index, dtype="object")
    for flag, cls in C.FLAG_PRIORITY:
        label = label.mask(label.isna() & sos[flag], cls)
    sos["label"] = label

    # Traceability: keep every flag that fired, not just the winning one.
    short = {f: f.replace("in_vari_", "") for f in flags}
    flag_vals = sos[flags].to_numpy()
    sos["sos_flags"] = [
        "|".join(short[f] for f, on in zip(flags, row) if on) or pd.NA
        for row in flag_vals
    ]

    n_sos = int(sos["label"].notna().sum())
    print(f"\nLabelled by SOS       : {n_sos:,}  ({n_sos / len(sos) * 100:.1f}%)")
    print(f"  multi-flag resolved : {int((sos['n_sos_flags'] > 1).sum()):,}")

    # ── Classifier fallback ───────────────────────────────────────────────────
    cat = sos.merge(
        clf[["diaObjectId", "best_class_name", "best_class_score"]],
        on="diaObjectId", how="left",
    )

    usable_clf = cat["best_class_name"].notna()
    if min_clf_score is not None:
        usable_clf &= cat["best_class_score"] >= min_clf_score
        print(f"Classifier score cut  : >= {min_clf_score}")

    from_clf = cat["label"].isna() & usable_clf
    cat.loc[from_clf, "label"] = cat.loc[from_clf, "best_class_name"]

    cat["label_source"] = np.select(
        [cat["n_sos_flags"] > 0, from_clf], ["sos", "classifier"], default="none"
    )
    # A score is meaningful only for classifier labels; SOS membership carries none.
    cat["label_score"] = cat["best_class_score"].where(cat["label_source"] == "classifier")

    print(f"Labelled by classifier: {int(from_clf.sum()):,}")
    print(f"Unlabelled            : {int((cat['label_source'] == 'none').sum()):,}")

    # ── Sort: SOS block, classifier block, unlabelled; then by significance ───
    cat["_order"] = cat["label_source"].map(C.SOURCE_ORDER)
    cat = cat.sort_values(
        ["_order", "r_var_significance"], ascending=[True, False], kind="mergesort"
    ).drop(columns="_order")

    cat = assign_fields(cat)

    cols = [
        "diaObjectId", "source_id", "label", "label_source", "label_score",
        "field", "field_fold",
        "r_var_significance", "n_sos_flags", "sos_flags",
        "ra_lsst", "dec_lsst", "ra_gaia", "dec_gaia", "phot_variable_flag",
    ]
    cat = cat[[c for c in cols if c in cat.columns]]

    C.ensure_dirs(C.CATALOG_DIR)
    cat.to_csv(C.LABELED_CSV, index=False)

    print(f"\n{'=' * 62}")
    print(f"Saved → {C.LABELED_CSV}  ({len(cat):,} rows)")
    print(f"{'=' * 62}")
    print("\nLabel × source:")
    pivot = (cat.pivot_table(index="label", columns="label_source",
                             values="diaObjectId", aggfunc="count", fill_value=0)
                .assign(total=lambda d: d.sum(axis=1))
                .sort_values("total", ascending=False))
    print(pivot.to_string())
    print(f"\nTotal labelled: {int(cat['label'].notna().sum()):,} / {len(cat):,}")

    return cat


def assign_fields(cat):
    """
    Group objects into sky fields and deal whole fields into C.N_FIELD_FOLDS folds.

    Folds are balanced on trusted labels (GroupKFold); unlabelled objects inherit the
    fold of their field. DBSCAN noise (field -1) is one more group. Written into the
    catalog as `field` / `field_fold` so every stage holds out the same fields.
    """
    xy = np.radians(cat[["dec_lsst", "ra_lsst"]].to_numpy())
    field = DBSCAN(eps=np.radians(C.FIELD_EPS_DEG), min_samples=C.FIELD_MIN_SAMPLES,
                   metric="haversine").fit_predict(xy)

    lab = np.where(cat["label"].notna().to_numpy()
                   & trusted(cat["label"], cat["label_source"]))[0]
    fold_of_field = {}
    for k, (_, te) in enumerate(GroupKFold(C.N_FIELD_FOLDS).split(lab, groups=field[lab])):
        fold_of_field.update({f: k for f in np.unique(field[lab[te]])})
    cat["field"] = field
    cat["field_fold"] = [fold_of_field.get(f, -1) for f in field]

    print(f"\nSky fields: {len(set(field) - {-1})} (+ noise), "
          f"{C.N_FIELD_FOLDS} folds of whole fields")
    t = cat.iloc[lab]
    print(pd.crosstab(t["field_fold"], t["label"])
            .reindex(columns=target_classes(cat), fill_value=0).to_string())
    return cat


def load():
    """Read the labelled catalog built by `build()`."""
    return pd.read_csv(C.LABELED_CSV, dtype={"diaObjectId": str, "source_id": str,
                                             "sos_flags": str})


def trusted(labels, sources):
    """
    Boolean mask: is this label usable? Target-class labels must come from
    C.TARGET_SOURCES; every other label is kept whatever its source.
    """
    targets = set(target_classes())
    labels = np.asarray(labels, dtype=object)
    sources = np.asarray(sources, dtype=object)
    in_target = np.array([l in targets for l in labels])
    return ~in_target | np.isin(sources, C.TARGET_SOURCES)


def target_classes(cat=None):
    """
    Classes the multiclass head predicts: C.TARGET_CLASSES when pinned, otherwise
    every class with at least C.MIN_TARGET_SIZE labelled members.
    """
    if C.TARGET_CLASSES is not None:
        return list(C.TARGET_CLASSES)
    cat = load() if cat is None else cat
    counts = cat["label"].value_counts()
    keep = counts[counts >= C.MIN_TARGET_SIZE]
    # Target class first, then by population — keeps reports readable.
    ordered = [C.POSITIVE_CLASS] if C.POSITIVE_CLASS in keep.index else []
    ordered += [c for c in keep.index if c != C.POSITIVE_CLASS]
    return ordered
