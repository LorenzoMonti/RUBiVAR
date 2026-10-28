# RUBiVAR — Rubin variable-star classification in early LSST data

Classification of variable stars from Vera C. Rubin Observatory / LSST data, with
**RR Lyrae** as the priority class: what can and cannot be done with the few epochs the
early data releases provide.

Method: a 5-class gradient-boosted classifier on classical light-curve and dm–dt features
(ML), benchmarked against a transformer on raw light curves (DL) and a semi-supervised VAE
on dm–dt maps (generative) under one protocol, scored on held-out sky fields
(`dp2/bench/`).

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

Input catalogues (in `dp2/catalogs/`). The Gaia DR3 × DP2 crossmatch was run on RSP by a
collaborator and is described in the paper; `rsp/crossmatch_gaia.py` is a placeholder:

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
├── period_check.py    LS period of the RR vs their Gaia DR3 period
├── bench/             benchmark: ML (HGB) / DL (transformer) / generative (SS-VAE)
├── slurm/             jobs for Leonardo (CINECA)
├── catalogs/          Gaia catalogues + ground truth + periods + features
└── rsp/               runs on the Rubin Science Platform
    ├── crossmatch_gaia.py    placeholder: the Gaia DR3 × DP2 crossmatch (see the paper)
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
python main.py period-check             # LS period of the RR vs Gaia DR3, by nights

# benchmark, same protocol for every model (M = hgb | hgb_classic | hgb_dmdt | transformer | vae | vae_hgb | mlp_dmdt)
python main.py bench-cache                         # sequences + first-n-nights features
python main.py bench --model M --step tune         # Optuna on inner validation fields
python main.py bench --model M --step eval         # tests 1-2: held-out fields, first n nights
python main.py bench --model M --step lc           # test 3: learning curve
python main.py bench --step report  [--model M ...] [--out report.txt]   # tables + bootstrap
python main.py bench --step figures [--model M ...] [--out figures]      # paper figures
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

Every model is tuned and scored the same way (see `config.py`, "Benchmark"): Optuna (30
trials, the first one the untuned defaults) on three inner splits whose validation sides
are disjoint groups of whole sky fields, objective RR AUC-PR on the pooled validation
predictions; each outer fold is then predicted once per seed (3 seeds). No augmentation.
Three tests, all on held-out fields:

1. the tuned models, also broken down by fold, observing nights and points per night;
2. well-sampled objects (≥ 18 nights) cut to their **first n nights** (3, 5, 10) and
   re-measured from scratch, Lomb-Scargle period included — what early LSST data look like;
3. a learning curve (25 / 50 / 100% of the training rows, seed 0).

Differences between models come with a paired bootstrap over whole fields (95% intervals).

| `--model` | Role | Input |
|---|---|---|
| `hgb` | ML | 120 classical features + 340 dm–dt bins |
| `transformer` | DL | raw light curve (time, band, magnitude), no period |
| `vae` | generative | dm–dt maps + 6 tabular values; SS-VAE, linear head on the latent |
| `hgb_classic` | ablation | the 120 classical features only |
| `hgb_dmdt` | ablation | exactly the VAE input, to trees |
| `vae_hgb` | ablation | the VAE latent, tree head |
| `mlp_dmdt` | ablation | the VAE encoder without ELBO and unlabelled objects |

### Results (held-out fields, 33,491 objects, 18 fields)

| Model | RR AUC-PR | Balanced accuracy | RR AUC-PR, first 10 / 5 / 3 nights |
|---|---|---|---|
| ML: HGB | **0.797 ± 0.006** | **0.837 ± 0.002** | 0.70 / 0.28 / 0.25 |
| Generative: SS-VAE | 0.676 ± 0.015 | 0.807 ± 0.001 | 0.57 / 0.41 / 0.30 |
| DL: transformer | 0.599 ± 0.016 | 0.756 ± 0.012 | 0.57 / 0.39 / 0.31 |

Mean ± std over seeds; chance RR AUC-PR in the few-nights test is 0.064. Full tables in
`results/bench/report.txt` (three models) and `report_ablations.txt` (all seven).

- HGB beats both alternatives (VAE −0.11 [−0.13, −0.06], transformer −0.15 [−0.25, −0.08]).
  VAE vs transformer is not resolved (−0.04 [−0.14, +0.03]).
- Classical and dm–dt features are complementary: either alone loses to both together
  (classical only −0.05 [−0.10, −0.03]).
- At fixed input, trees and the VAE are not distinguishable (−0.02 [−0.06, +0.03]); the
  VAE's generative part does help: the same encoder without ELBO and unlabelled objects
  loses −0.05 [−0.09, −0.01], and more on the first 3–5 nights.
- With 3–5 nights no model is usable; the differences between models there are within
  the bootstrap intervals.
- The two bulge fields are hard for every model (HGB 0.53 / 0.62 vs ≥ 0.74 elsewhere).
- Learning curve: HGB is nearly flat (0.77 → 0.80), the transformer still climbing
  (0.38 → 0.60).

### Why held-out fields, and why period-free representations

DP2 fields differ widely in cadence and class mix (the two Galactic-plane fields hold
almost all LPVs, some extragalactic fields almost all AGN). With a random split every
field sits in both training and test, and a model can recognise the field instead of the
star; scoring on whole held-out fields prevents that.

A field is a set of objects linked by shared visits (`dp2.Visit` has no target or field
name). DBSCAN on positions agreed on every isolated field but cut one wide strip
(RA 283–342°, 3,518 objects sharing visits) into 7 pieces spread over 3 folds.

The OTHER class is needed because the real catalogue holds the other classes too: without
it they mostly end up as ECL or RR.

With DP2 sampling the Lomb-Scargle period matches Gaia DR3 (`vari_rrlyrae`, within 2%;
`main.py period-check`) for only 31% of RR Lyrae (17% with ≤ 8 nights, 42% with ≥ 18): an SS-VAE on the phase-folded curve reached RR AUC-PR
0.17, the same model on dm–dt maps 0.68. Classical statistics and dm–dt maps carry
information that does not need the right period.

### Limitations

- Labels are Gaia DR3 classes, so the benchmark covers stars bright enough for Gaia
  (G ≲ 20.7), not the depth LSST reaches.
- DP2 cadence is uneven and not that of the main survey; the few-nights test uses only
  objects with ≥ 18 nights: 10 of the 18 fields, none in bulge field 0 (fold 0).

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
