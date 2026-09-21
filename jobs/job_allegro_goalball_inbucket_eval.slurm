#!/bin/bash
#SBATCH --job-name=akt_goalball_inbucket_eval
#SBATCH --output=slurm/allegro_goalball_inbucket_eval_%j.log
#SBATCH --error=slurm/allegro_goalball_inbucket_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Stochastic in-bucket eval of every 7.5cm (goal_ball) Allegro throw run.
# Re-runnable: skips ckpt iters already in the per-run CSV.
#   sbatch jobs/job_allegro_goalball_inbucket_eval.slurm
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

mkdir -p slurm figs/allegro_kuka_throw/goalball_inbucket_eval
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "eval: train=7.5cm goal_ball; score=ever in bucket (cylinder)"
nvidia-smi || true

python -u scripts/eval_allegro_goalball_inbucket.py

echo "done"
ls -lh figs/allegro_kuka_throw/goalball_inbucket_eval \
  figs/allegro_kuka_throw/akt_live_queue_goalball_inbucket_eval.png || true
