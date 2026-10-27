"""
Generative ablation: the SS-VAE encoder of vae.py without its generative part (no
decoder, no KL, no unlabelled objects) — SupCon and the same linear head on the same
dm-dt input. vae vs mlp_dmdt isolates what the ELBO and the unlabelled data add;
mlp_dmdt vs hgb_dmdt, deep vs trees at fixed input.
"""

from functools import partial

from bench import vae
from bench.vae import USES_VAL, embed, predict, predict_degraded  # noqa: F401

NAME = "mlp_dmdt"
DEFAULT = {k: vae.DEFAULT[k] for k in ("latent_dim", "lr", "temp", "C")}


def space(trial):   # vae.space without beta_max and lambda_max, which no longer act
    return {"latent_dim": trial.suggest_categorical("latent_dim", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "temp": trial.suggest_categorical("temp", [0.05, 0.07, 0.1, 0.2]),
            "C": trial.suggest_float("C", 1e-2, 10.0, log=True)}


train = partial(vae.train, generative=False)
