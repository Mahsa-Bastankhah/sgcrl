#!/bin/bash
#SBATCH --job-name=bb_eval_c3t1
#SBATCH --output=slurm/watch_bb_c3t1_pd_new_eval_%j.log
#SBATCH --error=slurm/watch_bb_c3t1_pd_new_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=seas

# Incremental checkpoint eval for creative-3-task1 PD (tau=0) new run.
# Watches logs/ppo_builderbench_creative3_task1_e1024_pd_new/ and updates:
#   figs/builderbench/creative3_task1_pd_new_checkpoint_success.png
#   figs/builderbench/checkpoint_eval/ppo_builderbench_creative_3_task1_0_checkpoint_success.csv
unset LD_PRELOAD

module purge
try_cuda() {
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -n "${CUDA_HOME:-}" ] || return 1
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || true

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p slurm figs/builderbench/checkpoint_eval

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "watching logs/ppo_builderbench_creative3_task1_e1024_pd_new/ (interval=300s)"
echo "training job: ppo_bb_c3t1_pd (sbatch jobs/job_ppo_builderbench_creative3_task1_e1024_pd_tau.slurm)"

python -u scripts/watch_builderbench_checkpoint_eval.py \
  --log_dir=logs/ppo_builderbench_creative3_task1_e1024_pd_new \
  --env=builderbench_creative_3_task1 \
  --plot_tag=creative3_task1_pd_new \
  --num_eval_episodes=20 \
  --eval_seed=0 \
  --watch_interval=300
