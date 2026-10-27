"""
Gradient-boosted trees (sklearn HistGradientBoosting) on the 460 classical + dm-dt
features — the ML entry of the benchmark. The period comes in through the features
(LS period, Fourier fit on it); that is the point of comparison with the transformer.

`view` restricts the trees to another input, for the representation ablations
(hgb_classic.py, hgb_dmdt.py): a function of (feature rows, their TAB_COLUMNS).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from bench import degrade, vae

NAME = "hgb"
USES_VAL = False
_DEG = None   # few-nights features, loaded on first use
DEFAULT = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20,
           "l2_regularization": 1e-4, "max_features": 1.0}   # ~sklearn defaults, as classify.py


def space(trial):
    return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0)}


def _x(view, x, tab):
    return x if view is None else view(x, tab())


def train(d, tr, va, params, seed, view=None):
    x = _x(view, d.x[tr], lambda: vae._inputs(d)[2][tr])
    model = HistGradientBoostingClassifier(**params, class_weight="balanced",
                                           random_state=seed).fit(x, d.y[tr])
    model.n_cls, model.view = len(d.cls), view
    return model


def _proba(model, x):
    out = np.zeros((len(x), model.n_cls))
    out[:, model.classes_] = model.predict_proba(x)   # a fold can lack a class (AGN)
    return out


def predict(model, d, rows):
    return _proba(model, _x(model.view, d.x[rows], lambda: vae._inputs(d)[2][rows]))


def predict_degraded(model, d, idx):
    global _DEG
    if _DEG is None:
        _DEG = degrade.load()[0]
    return _proba(model, _x(model.view, _DEG[idx], lambda: degrade.load_folded()[1][idx]))
