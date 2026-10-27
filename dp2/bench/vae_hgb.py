"""
The SS-VAE of vae.py with gradient-boosted trees as the head instead of a logistic
regression: what the latent holds rather than what a linear model reads from it.
Same tuned parameters (the vae study); nothing is tuned here.
"""

from functools import partial

from bench import vae
from bench.vae import DEFAULT, USES_VAL, embed, predict, predict_degraded  # noqa: F401

NAME = "vae_hgb"
STUDY = "vae"
train = partial(vae.train, head="hgb")
