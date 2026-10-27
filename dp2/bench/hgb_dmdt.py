"""
Model ablation at fixed input: the trees of hgb.py on exactly what the SS-VAE reads —
the dm-dt maps, which dt rows were sampled, and TAB_COLUMNS (raw: trees need neither
the log period nor z-scores, and handle the missing colours themselves).
"""

from functools import partial

import numpy as np

from bench import hgb, vae
from bench.hgb import DEFAULT, USES_VAL, predict, predict_degraded, space  # noqa: F401

NAME = "hgb_dmdt"


def _vae_input(x, tab):
    maps, mask = vae._maps(x)
    return np.hstack([maps.reshape(len(x), -1), mask, tab])


train = partial(hgb.train, view=_vae_input)
