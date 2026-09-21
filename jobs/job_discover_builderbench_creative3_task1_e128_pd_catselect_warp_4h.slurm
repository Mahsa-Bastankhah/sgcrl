#!/bin/bash
#SBATCH --job-name=discover_bb_c3t1_cs_warp4h
#SBATCH --output=slurm/discover_builderbench_creative3_task1_e128_pd_catselect_warp_4h_%A_%a.log
#SBATCH --error=slurm/discover_builderbench_creative3_task1_e128_pd_catselect_warp_4h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# DISCOVER (TD3+HER+UCB) on BuilderBench creative-3-task1.
# Paper UCB hyps (ensemble=6, σ_ini=1, μ_tar=1, HER future=0.7, transform_ucb).
# PD + catselect (3 cubes). E=128 matches DISCOVER arm recipe (not PPO E=1024).
# Warp 4h cap: E=128 is ~1k env steps/s, so this run will not reach 200M.
unset LD_PRELOAD

module purge

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

CUDA_LOADED=0
try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    if [ -z "${CUDA_HOME:-}" ]; then
      for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
        if [ -d "${_cand}/lib64" ]; then
          export CUDA_HOME="${_cand}"
          break
        fi
      done
    fi
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || \
  try_cuda cudatoolkit/12.5 || \
  try_cuda cudatoolkit/12.4 || true

if [ "$CUDA_LOADED" != 1 ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1
      echo "Fallback CUDA_HOME=${CUDA_HOME} (module load failed)"
      break
    fi
  done
fi

if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit and no /usr/local/cuda* found."
  module avail cudatoolkit 2>&1 || true
  ls -d /usr/local/cuda* 2>&1 || true
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench

source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=warp
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV=${ENV:-builderbench_creative_3_task1}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
NUM_ENVS=${NUM_ENVS:-128}
LOG_DIR="${LOG_DIR:-logs/discover_builderbench_creative3_task1_e128_pd_catselect_warp_4h/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS}"
echo "log_dir=${LOG_DIR} num_envs=${NUM_ENVS}"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "discover: TD3+HER+UCB ensemble=6 ucb_std_ini=1 ucb_mean_tar=1 her_future=0.7"
echo "builderbench_use_pd=true pd_duration=5 permute_start_boxes=false"
echo "categorical_select=true (3 cubes) pd_macro_ep_len=50"
echo "discover_eval_interval=10 discover_checkpoint_interval=10 discover_video_interval=10"
echo "goal_reach_thresh=0.1 (DISCOVER sparse ball); logs also track BB success@2cm"

python -u -c "
from envs.builderbench_utils import default_fixed_target_goal, num_creative_tasks
import numpy as np
print('creative-3 n_tasks=', num_creative_tasks(3))
g = default_fixed_target_goal(3, 0)
print('hard_goal (flat)=', g.tolist())
print('hard_goal (cubes)=')
print(np.asarray(g).reshape(-1, 3))
"
nvidia-smi || true
python -u -c "import os, sgcrl_jax_acme_compat, warp, jax; print('warp', warp.__version__); print('BUILDERBENCH_MJX_IMPL', os.environ.get('BUILDERBENCH_MJX_IMPL')); print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u baseline-agents/discover/train.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num-timesteps="${NUM_STEPS}" \
  --num-envs="${NUM_ENVS}" \
  --num-eval-envs="${NUM_ENVS}" \
  --log-dir="${LOG_DIR}" \
  --eval-interval=10 \
  --checkpoint-interval=10 \
  --video-interval=10 \
  --categorical-select
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
