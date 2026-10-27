#!/bin/bash
# Token sequences + first-n-nights features (CPU only; a booster node is the only
# partition this account can use, so it takes one GPU it does not need).
#SBATCH --job-name=rubivar-cache
#SBATCH --account=iscrc_tranrr
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:1 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
source slurm/env.sh
python main.py bench-cache
