"""
Representation ablation: the trees of hgb.py on the 120 classical features alone,
without the dm-dt maps.
"""

from functools import partial

from bench import hgb
from bench.hgb import DEFAULT, USES_VAL, predict, predict_degraded, space  # noqa: F401

NAME = "hgb_classic"
N_CLASSICAL = 120   # features.npz: classical features first, then dm-dt


def _classical(x, tab):
    return x[:, :N_CLASSICAL]


train = partial(hgb.train, view=_classical)
