"""
Gradient-boosted trees (sklearn HistGradientBoosting) on the 460 classical + dm-dt
features — the ML entry of the benchmark. The period comes in through the features
(LS period, Fourier fit on it); that is the point of comparison with the transformer.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from bench import degrade

NAME = "hgb"
USES_VAL = False
_DEG = None   # few-nights features, loaded on first use


def space(trial):
    return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0)}


def train(d, tr, va, params, seed):
    model = HistGradientBoostingClassifier(**params, class_weight="balanced",
                                           random_state=seed).fit(d.x[tr], d.y[tr])
    model.n_cls = len(d.cls)
    return model


def _proba(model, x):
    out = np.zeros((len(x), model.n_cls))
    out[:, model.classes_] = model.predict_proba(x)   # a fold can lack a class (AGN)
    return out


def predict(model, d, rows):
    return _proba(model, d.x[rows])


def predict_degraded(model, d, idx):
    global _DEG
    if _DEG is None:
        _DEG = degrade.load()[0]
    return _proba(model, _DEG[idx])
