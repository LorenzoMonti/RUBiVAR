"""
Tuning and scoring, the same for every model.

A model module provides
    NAME                    output folder under C.BENCH_DIR
    DEFAULT                 untuned params, the first trial of every study
    USES_VAL                True if training needs the validation rows (early stopping);
                            otherwise the final fit takes all of fold k's training rows
    space(trial)            Optuna search space → params dict
    train(d, tr, va, params, seed)    → fitted model
    predict(model, d, rows)           → (len(rows), n_classes) probabilities
    predict_degraded(model, d, idx)   → same, on rows idx of features_degraded.npz
    embed(model, d, rows)             optional: the model's representation of the rows,
                                      saved for seed 0 (latent figures, period probe)

    tune(module, k)            Optuna on fold k's inner splits; resumable, one journal
                               per fold, so folds can run as parallel jobs
    evaluate(module)           tests 1-2: best params → fold k, once per C.BENCH_SEEDS
    learning_curve(module)     test 3
"""

import os
import time

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.metrics import average_precision_score

import config as C
from bench import degrade, protocol

optuna.logging.set_verbosity(optuna.logging.WARNING)   # trials are printed by tune()


def _study(module, k):
    path = os.path.join(protocol.model_dir(module.NAME), f"hpo_fold{k}.journal")
    return optuna.create_study(
        study_name=f"{module.NAME}_fold{k}", direction="maximize", load_if_exists=True,
        storage=JournalStorage(JournalFileBackend(path)),
        sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED + k))


def tune(module, k, n_trials=C.BENCH_TRIALS):
    """
    Run trials on fold k until n_trials are complete (earlier runs count). Each trial
    trains once per inner split; the objective is RR AUC-PR over the pooled validation
    predictions (every validation object predicted once). Trial 0 is module.DEFAULT,
    so tuning starts from the untuned model.
    """
    d = protocol.load()
    splits = protocol.inner_splits(d, k)
    rr = d.cls.index(C.POSITIVE_CLASS)
    va_all = np.concatenate([va for _, va in splits])
    study = _study(module, k)
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print(f"[{module.NAME} fold {k}] {len(splits)} inner splits, validation "
          + ", ".join(f"{len(va):,}" for _, va in splits)
          + f" (RR {int((d.y[va_all] == rr).sum())}, {len(np.unique(d.fields[va_all]))} fields)"
          + f"  trials done {done}/{n_trials}", flush=True)
    if not study.trials:
        study.enqueue_trial(module.DEFAULT)

    def objective(trial):
        t0 = time.time()
        params = module.space(trial)
        p = np.concatenate([module.predict(module.train(d, tr, va, params, C.RANDOM_SEED), d, va)
                            for tr, va in splits])
        ap = average_precision_score(d.y[va_all] == rr, p[:, rr])
        print(f"[{module.NAME} fold {k}] trial {trial.number:3d}  RR AP {ap:.3f}  "
              f"({time.time() - t0:.0f}s)  {trial.params}", flush=True)
        return ap

    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
    print(f"[{module.NAME} fold {k}] best RR AP {study.best_value:.3f}  {study.best_params}"
          f"  (default: {study.trials[0].value:.3f})")
    return study.best_params


def best_params(module, k):
    return _study(module, k).best_params


def _train_rows(module, d, k, frac=1.0):
    """
    (train, val) rows of outer fold k. Models without early stopping also train on the
    validation fields. frac < 1 keeps that fraction of each class's training rows.
    """
    tr, va = protocol.inner_split(d, k)
    if not module.USES_VAL:
        tr, va = np.concatenate([tr, va]), va[:0]
    if frac < 1:
        rng = np.random.default_rng(C.RANDOM_SEED + k)
        tr = np.sort(np.concatenate([
            rng.choice(r, max(1, round(frac * len(r))), replace=False)
            for c in np.unique(d.y[tr]) for r in [tr[d.y[tr] == c]]]))
    return tr, va


def _save(name, fname, **arrays):
    np.savez_compressed(os.path.join(protocol.model_dir(name), fname), **arrays)


def evaluate(module, seeds=C.BENCH_SEEDS, folds=range(C.N_FIELD_FOLDS)):
    """
    Tests 1 and 2: best params of fold k → predict its real curves and its first-n-night
    cuts. One file per (seed, fold), so folds can run as separate jobs.
    """
    d = protocol.load()
    _, deg_rows, _ = degrade.load()
    params = {k: best_params(module, k) for k in folds}
    for seed in seeds:
        for k in folds:
            t0 = time.time()
            tr, va = _train_rows(module, d, k)
            model = module.train(d, tr, va, params[k], seed)
            te = protocol.outer_split(d, k)[1]
            idx = np.where(d.folds[deg_rows] == k)[0]   # none in fold 0: no bulge object
            deg = (module.predict_degraded(model, d, idx) if len(idx)   # has 18 nights
                   else np.zeros((0, len(d.cls))))
            extra = ({"emb": module.embed(model, d, te)}
                     if seed == seeds[0] and hasattr(module, "embed") else {})
            _save(module.NAME, f"oof_s{seed}_f{k}.npz", rows=te,
                  prob=module.predict(model, d, te), deg_idx=idx, deg_prob=deg, **extra)
            print(f"[{module.NAME} seed {seed}] fold {k} done ({time.time() - t0:.0f}s)",
                  flush=True)
    _save_meta(module, "evaluate", seeds=list(seeds), folds=list(folds), params=params)


def learning_curve(module, fracs=C.LC_FRACTIONS, folds=range(C.N_FIELD_FOLDS)):
    """Test 3: training rows cut to each fraction, best params, seed 0."""
    d = protocol.load()
    for k in folds:
        params = best_params(module, k)
        te = protocol.outer_split(d, k)[1]
        for f in fracs:
            t0 = time.time()
            tr, va = _train_rows(module, d, k, f)
            model = module.train(d, tr, va, params, 0)
            _save(module.NAME, f"lc_{f:g}_f{k}.npz", rows=te, prob=module.predict(model, d, te))
            print(f"[{module.NAME} {f:.0%}] fold {k}: {len(tr):,} training rows "
                  f"({time.time() - t0:.0f}s)", flush=True)
    _save_meta(module, "learning_curve", fracs=list(fracs), folds=list(folds))


def _save_meta(module, step, **extra):
    protocol.save_json(os.path.join(protocol.model_dir(module.NAME), f"meta_{step}.json"),
                       protocol.run_meta(model=module.NAME, step=step,
                                         trials=C.BENCH_TRIALS, **extra))


def _merge(name, prefix, n_obj, n_cls, n_deg=0):
    """Join the per-fold files <prefix>_f<k>.npz; None if a fold is missing."""
    prob, deg = np.full((n_obj, n_cls), np.nan), np.full((n_deg, n_cls), np.nan)
    for k in range(C.N_FIELD_FOLDS):
        path = os.path.join(C.BENCH_DIR, name, f"{prefix}_f{k}.npz")
        if not os.path.exists(path):
            return None
        a = np.load(path)
        prob[a["rows"]] = a["prob"]
        if n_deg:     # the learning curve has no first-n-nights predictions
            deg[a["deg_idx"]] = a["deg_prob"]
    return prob, deg


def load_oof(name, d, n_deg):
    """{seed: (prob, deg_prob)} over every fold, for the seeds whose folds are all done."""
    res = {}
    for seed in C.BENCH_SEEDS:
        m = _merge(name, f"oof_s{seed}", len(d.y), len(d.cls), n_deg)
        if m is not None:
            res[seed] = m
    if not res:
        raise FileNotFoundError(f"{name}: no seed with every fold scored — run evaluate.")
    return res


def load_lc(name, d):
    """{fraction: prob} of the learning curve, 1.0 being seed 0 of evaluate."""
    res = {f: _merge(name, f"lc_{f:g}", len(d.y), len(d.cls)) for f in C.LC_FRACTIONS}
    res[1.0] = _merge(name, "oof_s0", len(d.y), len(d.cls))
    return {f: m[0] for f, m in res.items() if m is not None}
