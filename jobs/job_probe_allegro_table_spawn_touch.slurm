#!/bin/bash
#SBATCH --job-name=akt_tablespawn_touch_probe
#SBATCH --output=slurm/probe_allegro_table_spawn_touch_%j.log
#SBATCH --error=slurm/probe_allegro_table_spawn_touch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# One-env confirm: cube on desk, IK palm touching it. Then the 300M job.
#   sbatch jobs/job_probe_allegro_table_spawn_touch.slurm

unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  probe table_spawn touch"
nvidia-smi || true
python -u scripts/probe_allegro_table_spawn_touch.py
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
