#!/bin/bash
# End-to-end check before the long runs: one trial per model on fold 2, then scoring
# fold 2 with seed 0. Results go to a scratch directory, not results/bench.
#SBATCH --job-name=rubivar-smoke
#SBATCH --account=iscrc_tranrr
#SBATCH --partition=boost_usr_prod --qos=boost_qos_dbg
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
source slurm/env.sh
python - <<'PY'
import config as C
C.BENCH_DIR = "/leonardo_work/IscrC_TranRR/variae/smoke"
from bench import hgb, run, transformer, vae
transformer.EPOCHS, vae.EPOCHS = 2, 5
for m in (hgb, vae, transformer):
    run.tune(m, 2, n_trials=1)
    run.evaluate(m, seeds=[0], folds=[2])
    run.learning_curve(m, fracs=[0.25], folds=[2])
print("SMOKE OK")
PY
