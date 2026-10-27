# Sourced by every job: the project's own venv, never the module Python (its
# PYTHONPATH breaks other interpreters). Compute nodes have no internet.
module purge
unset PYTHONPATH
WORK=/leonardo_work/IscrC_TranRR/variae
source "$WORK/env/bin/activate"
cd "$WORK/repo/dp2"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
