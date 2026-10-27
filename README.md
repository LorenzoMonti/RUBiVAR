# RUBiVAR — Rubin variable-star classification in early LSST data

Classification of variable stars from Vera C. Rubin Observatory / LSST data, with
**RR Lyrae** as the priority class: what can and cannot be done with the few epochs the
early data releases provide.

Method: a 5-class gradient-boosted classifier on classical light-curve features, scored on
held-out sky fields, benchmarked against a transformer on raw light curves (DL) and a
semi-supervised VAE (generative) under one protocol (`dp2/bench/`).

---

## Repository layout

```
RUBiVAR/
├── dp1/        # DP1 — frozen archive (complete pipeline code)
├── dp2/        # DP2 — active work
├── requirements.txt
└── LICENSE
```

Each data release has its own self-contained tree. `BASE` in the scripts is derived from
`__file__`, so each folder runs in place without absolute paths.

---

## DP1 — archive (`dp1/`)

First complete iteration on LSST DP1: 13,505 `strict` light curves, Gaia DR3 crossmatch
(SOS + 2″ positional match + classifier), 392 labelled objects, a 32-dimensional SS-VAE,
and binary RR and multiclass classifiers on the latent space.

See `dp1/README.md` for the pipeline, the results and the exact state in which the work
stopped. The folder is read-only: do not modify it.

---

## DP2 — active work (`dp2/`)

A new iteration on DP2 data with a much larger ground truth: **~34,000 LSST stars
classified as variable by Gaia DR3**.

Input catalogues (in `dp2/catalogs/`, produced by `rsp/crossmatch_gaia.py`):

| File | Role |
|---|---|
| `lsst_dp2_gaia_sos.csv` | Gaia SOS — 34,553 objects, 11 `in_vari_*` flags; high-reliability classes, primary source |
| `lsst_dp2_gaia_classifier_result.csv` | Gaia classifier — 33,485 objects with `best_class_name` / `best_class_score`; fallback for objects without an SOS class |
| `class_definition.txt` | Legend of the 24 Gaia classifier classes |

Coverage: 25,584 objects have at least one SOS flag (898 have more than one, resolved by
priority); the remaining 8,969 can take their class from the classifier.

### Layout

```
dp2/
├── main.py            single entry point
├── config.py          paths and hyperparameters — the single source of truth
├── catalog.py         Gaia ground truth (SOS → classifier) + sky fields (shared visits) and folds
├── lightcurves.py     Lomb-Scargle + phase-fold
├── features.py        classical features + dm–dt maps
├── classify.py        5-class classifier + RR candidates, scored on held-out fields
├── bench/             benchmark: ML (HGB) / DL (transformer) / generative (SS-VAE)
├── slurm/             jobs for Leonardo (CINECA)
├── catalogs/          Gaia catalogues + ground truth + periods + features
└── rsp/               runs on the Rubin Science Platform
    ├── crossmatch_gaia.py    Gaia DR3 × DP2 crossmatch → catalogues in catalogs/
    ├── download_lc.py        light curves → lightcurves/
    └── download_visits.py    dp2.Visit metadata → catalogs/ (field descriptions)
```

```bash
python main.py catalog                  # Gaia ground truth
                                        # (then download the light curves on RSP)
python main.py fields                   # sky fields from shared visits + 5 field folds
python main.py fold --min-points 30     # Lomb-Scargle + phase-fold
python main.py features                 # classical features + dm–dt
python main.py classify                 # classifier, held-out fields
python main.py all                      # the five stages in order

# benchmark, same protocol for the three models (M = hgb | transformer | vae)
python main.py bench-cache                         # sequences + first-n-nights features
python main.py bench --model M --step tune         # Optuna on inner validation fields
python main.py bench --model M --step eval         # tests 1-2: held-out fields, first n nights
python main.py bench --model M --step lc           # test 3: learning curve
python main.py bench --step report                 # one table → results/bench/report.txt
```

Downloading the light curves is **not** a `main.py` stage: it runs on RSP, the only place
where `lsst.rsp` exists.

### Pipeline

```
catalog   ground truth: SOS first, classifier as fallback
  ↓
fields    objects linked by shared visits form a sky field (18 fields);
          whole fields dealt into 5 folds
  ↓
fold      multiband Lomb-Scargle + phase-fold
  ↓
features  120 classical features (amplitudes, distribution shape, Stetson, χ²,
          Fourier, colours, period) + coarse-Δt dm–dt maps
  ↓
classify  5-class gradient boosting (RR, ECL, LPV, AGN, OTHER = every other
          label), scored out-of-fold on unseen fields. Two products: the most
          probable class with a low-confidence flag, and RR candidates with
          P(RR) above a threshold chosen on the other folds; results are also
          broken down by number of observing nights
```

### Benchmark

Every model is tuned and scored the same way (see `config.py`, "Benchmark"): Optuna on
three inner splits whose validation sides are disjoint groups of whole sky fields,
objective RR AUC-PR on the pooled validation predictions, first trial = the untuned
defaults; each outer fold is then predicted once per seed. No augmentation. Three tests, all on held-out
fields:

1. the tuned models, also broken down by fold, observing nights and points per night;
2. well-sampled objects (≥ 18 nights) cut to their **first n nights** (3, 5, 10) and
   re-measured from scratch — what early LSST data look like;
3. a learning curve (25 / 50 / 100% of the training rows).

### Why held-out fields, and why classical features

DP2 fields differ widely in cadence and class mix (the two Galactic-plane fields hold
almost all LPVs, some extragalactic fields almost all AGN). With a random split every
field sits in both training and test, and a model can recognise the field instead of the
star: sampling-only features gave AUC-PR 0.41 for RR vs ECL under a random split and 0.13
(below chance) on held-out fields.

A field is a set of objects linked by shared visits (`dp2.Visit` has no target or field
name). DBSCAN on positions agreed on every isolated field but cut one wide strip
(RA 283–342°, 3,518 objects sharing visits) into 7 pieces spread over 3 folds.

On held-out fields (AUC-PR, RR vs ECL / RR vs rest):

| Input | RR vs ECL | RR vs rest |
|---|---|---|
| SS-VAE latent, retrained per fold (old branch, replaced by `bench/vae.py`) | 0.24 | 0.32 |
| dm–dt maps | 0.70 | 0.63 |
| classical features | 0.83 | 0.70 |
| **classical features + dm–dt** | **0.85** | **0.74** |

The OTHER class is needed because the real catalogue holds the other classes too: without
it they almost all ended up as ECL (ECL precision 0.50) and 236 as RR. With OTHER,
precision rises to 0.79 (ECL) and 0.73 (RR), and P(RR) from the 5-class model beats a
dedicated RR-vs-rest binary (AUC-PR 0.80 vs 0.75). One head per class with ≥ 100 labels
instead hurts RR (0.73): RS, YSO, ELL and CV labels come only from the Gaia classifier.

The latent depends on the folded curve, but with DP2 sampling the Lomb-Scargle period is
right for only ~30% of RR Lyrae: in fields with few nights the folded curve is noise and
the representation does not transfer. Classical and dm–dt features carry information that
does not need the right period.

## Setup

```bash
pip install -r requirements.txt
```

Training on NVIDIA GPUs (CUDA). Phase-folding and Lomb-Scargle use CPU multiprocessing.

---

## Data

No data are versioned (see `.gitignore`): light curves, folds, periods, features,
catalogues, models and results stay out of the repository. DP2 data are subject to Rubin
data rights, so "reproducible" means: rerunnable by anyone with DP2 access on the Rubin
Science Platform, in the order

    rsp/crossmatch_gaia.py → rsp/download_lc.py      (on RSP)
    main.py all → main.py bench-cache → main.py bench ...

Every benchmark result stores the git commit, fingerprints of its input files and the
package versions (`results/bench/<model>/meta_*.json`). `_admin/` holds material unrelated
to the project and is excluded from version control.
