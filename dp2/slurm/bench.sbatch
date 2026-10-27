#!/bin/bash
# One benchmark step for one model; the array index is the outer fold.
#   sbatch --array=0-4 slurm/bench.sbatch transformer tune
#   sbatch --array=0-4 slurm/bench.sbatch transformer eval
#   sbatch --array=0-4 slurm/bench.sbatch transformer lc
#SBATCH --job-name=rubivar-bench
#SBATCH --account=iscrc_tranrr
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
source slurm/env.sh
python main.py bench --model "$1" --step "$2" --folds "$SLURM_ARRAY_TASK_ID"
