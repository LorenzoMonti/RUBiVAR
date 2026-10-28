"""
Single source of truth for paths, constants and hyperparameters.

Every other module reads from here. In DP1 these values lived scattered across ten
scripts and drifted apart; keeping them in one place is what stops that happening again.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE        = os.path.dirname(os.path.abspath(__file__))
CATALOG_DIR = os.path.join(BASE, "catalogs")
LC_DIR      = os.path.join(BASE, "lightcurves")   # lc_<diaObjectId>.csv, from RSP
FOLDED_DIR  = os.path.join(BASE, "folded")        # folded_<diaObjectId>.npy
MODEL_DIR   = os.path.join(BASE, "models")        # fitted feature classifiers
RESULTS_DIR = os.path.join(BASE, "results")

SOS_CSV       = os.path.join(CATALOG_DIR, "lsst_dp2_gaia_sos.csv")
CLF_CSV       = os.path.join(CATALOG_DIR, "lsst_dp2_gaia_classifier_result.csv")
CLASS_DEF_TXT = os.path.join(CATALOG_DIR, "class_definition.txt")
LABELED_CSV   = os.path.join(CATALOG_DIR, "dp2_labeled_catalog.csv")
PERIODS_CSV   = os.path.join(CATALOG_DIR, "periods.csv")
GAIA_RR_PERIODS_CSV = os.path.join(CATALOG_DIR, "gaia_rr_periods.csv")   # period_check.py
FEATURES_NPZ  = os.path.join(CATALOG_DIR, "features.npz")

RANDOM_SEED = 42

# ── Sky fields: the evaluation unit ───────────────────────────────────────────
#
# Our objects sit in a handful of separate sky regions, each with its own cadence and
# class mix (two Galactic-plane fields hold almost every LPV, a few extragalactic
# ones almost every AGN). With a random split every field sits on both sides, and a model
# can score by recognising the field instead of the star: sampling-only features
# reached AUC-PR 0.41 for RR vs ECL under a random split and 0.13, below chance,
# once whole fields were held out.
#
# A field is a set of objects linked by shared visits: two objects observed in the
# same exposure are in the same pointing, and chains of shared visits join
# overlapping pointings (connected components of the object-visit graph). dp2.Visit
# carries no target or field name, so this is the closest thing to an official field.
# DBSCAN on positions (1 deg) agreed on every isolated field but cut one wide strip
# (RA 283-342, 3,518 objects, all sharing visits) into 7 pieces spread over 3 folds,
# leaking the same pointing into training and test. Whole fields are dealt into
# N_FIELD_FOLDS folds (GroupKFold over trusted labels). Every classifier is scored
# out-of-fold on fields it never saw: that is the number to quote.
N_FIELD_FOLDS     = 5


# ── Ground truth ──────────────────────────────────────────────────────────────

# SOS flag → class name, in priority order (first match wins).
# Names follow the Gaia classifier vocabulary in class_definition.txt so that SOS and
# classifier labels share one namespace.  COMPACT_COMPANION has no classifier
# counterpart and keeps its own name (the classifier calls all 74 of them ELL).
#
# `short_timescale` ranks last on purpose: it is Gaia's catch-all for fast variability
# and accounts for every one of the 898 multi-flag conflicts, routinely co-flagged with
# a genuine physical class.
FLAG_PRIORITY = [
    ("in_vari_rrlyrae",              "RR"),
    ("in_vari_cepheid",              "CEP"),
    ("in_vari_planetary_transit",    "EP"),
    ("in_vari_microlensing",         "MICROLENSING"),
    ("in_vari_compact_companion",    "COMPACT_COMPANION"),
    ("in_vari_ms_oscillator",        "DSCT|GDOR|SXPHE"),
    ("in_vari_agn",                  "AGN"),
    ("in_vari_eclipsing_binary",     "ECL"),
    ("in_vari_long_period_variable", "LPV"),
    ("in_vari_rotation_modulation",  "SOLAR_LIKE"),
    ("in_vari_short_timescale",      "S"),
]

SOURCE_ORDER = {"sos": 0, "classifier": 1, "none": 2}

# Minimum Gaia classifier score to trust a fallback label. None = keep all.
# The classifier block is ~26% of labels with median score 0.62; tighten once the
# light-curve quality cut is settled.
MIN_CLF_SCORE = None


# ── Light curves: Lomb-Scargle + phase-fold ───────────────────────────────────

BANDS     = ["u", "g", "r", "i", "z", "y"]
# Time column in the downloaded light curves. DP1 called it expMidptMJD; the DP2
# DiaSource schema may use another name, so the reader accepts any of these.
TIME_COLUMNS = ["expMidptMJD", "midpointMjdTai", "midPointTai", "midpointTai",
                "obsStartMJD", "expMidpt"]

# Which flux to fold. dp2.ForcedSourceOnDiaObject carries both:
#   psfFlux     — PSF fit forced on the calexp (science image): total stellar flux.
#                 What DP1 used, and what a light curve normally means.
#   psfDiffFlux — PSF fit forced on the difference image: variability relative to
#                 the template. Immune to blending from constant neighbours, but
#                 with no mean level.
# Each band is normalised to zero mean / unit variance before folding, so the two
# mainly differ in noise and blending behaviour — worth comparing on real data.
FLUX_COLUMN = "psfFlux"
BAND_IDX  = {b: i for i, b in enumerate(BANDS)}

# Phase bins per band. Measured on the real DP2 light curves: RR Lyrae carry a median
# of ~20 measurements per band, so 100 bins left 82% of them empty and the encoder
# read mostly holes. At 50 bins the median RR band fills ~33% of its bins, and the
# chance a bin has data in at least one of the six bands rises from 70% to 91%.
# Resolution is still ample — an RR ab rise spans ~0.15 in phase, i.e. 7-8 bins.
# Any value works: the network derives its internal lengths from this one.
N_PHASE   = 50

FREQ_MIN  = 1.0 / 100.0   # day⁻¹ — longest period searched: 100 d
FREQ_MAX  = 24.0          # day⁻¹ — shortest period searched: 1 h
N_FREQ    = 100_000

# Quality gates. MIN_POINTS is deliberately loose here: the right value is the one
# measured on the labelled set (see `main.py fold --min-points`), not a guess.
# Forced photometry (ForcedSourceOnDiaObject) returns a measurement at every visit,
# so counts run far higher than DP1's detection-only light curves. Set this from the
# distribution the download script prints, not from the DP1 value.
MIN_POINTS          = 20   # total measurements across all bands
MIN_POINTS_PER_BAND = 3    # a band below this is left as NaN in the fold


# ── Classical features ────────────────────────────────────────────────────────
#
# Hand-crafted light-curve features (Richards et al. 2011 / ALeRCE style) plus a
# coarse dm-dt map. Under the field hold-out they beat the SS-VAE latent outright
# (RR vs ECL AUC-PR 0.83 against 0.24) and the latent only hurt when added to them.
#
# What carries the signal, by grouped permutation importance: per-band amplitudes,
# the shape of the magnitude distribution, chi2 against a constant, the LS period and
# the colours. Fourier shape on the folded curve contributes almost nothing — with
# DP2 sampling the LS period is right for only ~30% of RR, which is also why the
# folded-curve latent does not generalise across fields.
#
# Sampling counts (points, nights, span) are deliberately NOT features: they encode
# the field, not the star.
FEAT_MIN_BAND_POINTS = 5     # per-band statistics need at least this many points
FOURIER_MIN_POINTS   = 10    # ... and a Fourier fit at least this many
N_HARMONICS          = 3

# dm-dt (Mahabal et al. 2017): every pair of points in a band lands in a (dt, dm)
# cell, each dt column normalised to a distribution. It needs no period, which is
# why it helps most where the period fails (RR vs rest at <= 8 nights: 0.47 -> 0.60).
# Fine dt bins are pooled into DMDT_DT_GROUPS — at fine resolution which columns are
# empty is the cadence's fingerprint and the map named the field 99% of the time.
# Columns still empty after pooling take the population mean.
DMDT_MAPS     = ["all", "g", "r", "i"]
DMDT_DT_EDGES = [0, 1/48, 1/24, 1/12, 1/6, 1/3, 0.75, 1.5, 3, 6, 12, 25, 50, 100, 250]  # days
DMDT_DM_EDGES = [-float("inf"), -1.5, -1.0, -0.6, -0.4, -0.25, -0.15, -0.08, -0.03,
                 0.03, 0.08, 0.15, 0.25, 0.4, 0.6, 1.0, 1.5, float("inf")]           # mag
# < 1 h, 1-8 h, 18 h-3 d, 3-12 d, > 12 d. The 8-18 h column is daytime and is dropped.
DMDT_DT_GROUPS = [[0, 1], [2, 3, 4], [6, 7], [8, 9], [10, 11, 12, 13]]


# ── VAE (bench/vae.py) ────────────────────────────────────────────────────────

WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
FREE_BITS    = 0.5   # per-dimension KL floor, guards against posterior collapse
# Weight on reconstructing the tabular vector. The curve contributes 6 x N_PHASE
# elements against a handful of tabular ones, so without a boost the tabular
# features carry no weight in the ELBO and vanish from the latent.
TAB_RECON_WEIGHT = 3.0
# Minimum labelled members for a class to take part in SupCon. Its job there is
# repulsion, so a modest cluster still earns its place.
MIN_CLASS_SIZE = 30


# ── Classifiers ───────────────────────────────────────────────────────────────

# Classes the multiclass head predicts. None = derive from the labelled counts,
# keeping every class with at least MIN_TARGET_SIZE members.
#
# DP1 was restricted to three classes because nothing else had enough labels. That
# constraint is gone: RS (3,246), SOLAR_LIKE (2,861), AGN (1,448), S (1,581),
# DSCT (795) and YSO (541) all have more labelled examples than DP1 had in total.
# Modelling them explicitly should also help RR purity — right now those objects
# have nowhere to go but into one of the three classes or into OOD.
#
# Pin the list to ["RR", "ECL", "LPV"] to reproduce the DP1 setup.
#
# Pinned to four. With all 13 classes in SupCon only LPV and AGN separated: RS,
# SOLAR_LIKE, S, DSCT and YSO come mostly or entirely from the Gaia classifier and
# have no distinctive folded shape, so pulling them apart cost the RR separation.
# Every other class goes into OTHER for the classifier.
TARGET_CLASSES  = ["RR", "ECL", "LPV", "AGN"]

# OTHER: every trusted label outside TARGET_CLASSES, as one more class (classify.py).
# Without it the 9,364 such objects had nowhere to go: 8,528 were called ECL and 236 RR,
# so on held-out fields ECL precision was 0.50 and RR 0.67 once they were counted.
# With OTHER: ECL 0.79, RR 0.73, at the cost of 16% of true ECL sent to OTHER
# (probably low-amplitude binaries, which look like RS / SOLAR_LIKE with few nights).
# P(RR) from this model also beats a dedicated RR-vs-rest binary: AUC-PR 0.798 vs
# 0.749, paired field bootstrap [+0.022, +0.077]. Giving every class with >= 100
# labels its own head instead (11 classes) loses RR (0.734): RS, YSO, ELL and CV
# labels come only from the Gaia classifier and are not separable with few nights.
OTHER_CLASS = "OTHER"

# Label sources trusted for TARGET_CLASSES. Restricting to SOS drops 156 RR, 762 LPV
# and 144 AGN labels from the classifier fallback. Non-target classes keep every
# source: they only ever land in OTHER, where a noisy label costs little.
TARGET_SOURCES  = ("sos",)
MIN_TARGET_SIZE = 500   # needs ~100 per class in the test split to mean anything
POSITIVE_CLASS  = "RR"  # the candidate list (is_rr) and the benchmark's objective

# Gradient boosting on the classical features.
HGB_MAX_ITER = 300
HGB_LR       = 0.05
# An object whose top probability falls below the level that keeps this fraction of
# out-of-fold labelled objects is flagged low-confidence, not forced.
CONF_KEEP = 0.95

# Test metrics are also reported per bin of observing nights. A single AUC averages
# two regimes: on the first 4-class run AUC-PR went from 0.43 at <= 8 nights to
# 0.76 at 18-23, because a fold of few nights is often folded on an alias.
# Edges are inclusive lower bounds; roughly the quartiles of the test split.
NIGHT_BINS = [0, 9, 18, 24]


# ── Benchmark: ML vs DL vs generative (bench/) ────────────────────────────────
#
# Every model is tuned and scored the same way. Per outer field fold k: Optuna
# (BENCH_TRIALS trials, seeded, the first one the model's untuned defaults) on
# BENCH_INNER inner splits of the other folds, whose validation sides are disjoint
# groups of whole FIELDS; objective = RR AUC-PR over the pooled validation
# predictions. A single validation split chose configurations that generalised worse
# than the defaults (transformer 0.47 tuned vs 0.59 untuned on held-out fields).
# Fold k is then predicted once per seed in BENCH_SEEDS. GPU models are not
# bit-reproducible (bf16, atomic adds), hence several seeds. No augmentation, so no
# model has seen a cut curve before the few-nights test.
#
# Three tests, all on held-out fields:
#   1. the models as tuned (also split by fold, nights and points per night)
#   2. few nights: labelled objects with >= DEG_MIN_NIGHTS nights cut to their FIRST
#      n nights, n in DEG_LEVELS, everything re-measured (LS period included) — what
#      early LSST data look like. Random nights instead would keep the full baseline.
#   3. learning curve: inner training subsampled to LC_FRACTIONS, best params, one seed
# Real few-night DP2 curves carry 30-75 points per night (vs ~7 for well-sampled
# ones), but points per night are tied to the field too, so neither axis alone
# measures the information in a curve.
BENCH_DIR        = os.path.join(RESULTS_DIR, "bench")
FEATURES_DEG_NPZ = os.path.join(CATALOG_DIR, "features_degraded.npz")
BENCH_INNER    = 3
BENCH_VAL_MAX  = 0.25   # fields above this share of the training rows (the bulge)
                        # are never validation
BENCH_TRIALS   = 30
BENCH_SEEDS    = [0, 1, 2]
DEG_LEVELS     = [3, 5, 10]
DEG_MIN_NIGHTS = 18
LC_FRACTIONS   = [0.25, 0.5]


# ── Plotting ──────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "RR": "#e63946", "ECL": "#2a9d8f", "LPV": "#e9c46a",
    "AGN": "#8338ec", "SOLAR_LIKE": "#118ab2", "YSO": "#06d6a0",
    "RS": "#ffd166", "S": "#ff9f1c", "CV": "#ff006e",
    "DSCT|GDOR|SXPHE": "#8ecae6", "CEP": "#fb8500", "ELL": "#adb5bd",
    "COMPACT_COMPANION": "#6a4c93",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
