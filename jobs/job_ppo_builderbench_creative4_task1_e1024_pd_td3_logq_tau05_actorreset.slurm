#!/bin/bash
#SBATCH --job-name=ppo_bb_c4t1_td3_ar
#SBATCH --output=slurm/ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_actorreset_%A_%a.log
#SBATCH --error=slurm/ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_actorreset_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=seas
#SBATCH --array=0-1

# 4-cube creative-4-task1, PD + default TD3 (cross_batch_goals off,
# mlp Q, no bilinear). Actor-reset guard is always on for BuilderBench
# JAX (reinit policy if ep_length_mean < 80% of nominal).
# PPO reward = log((1-γ)·Q1) with EMA Q1 tau=0.5.
# Critic still uses online Q + Polyak targets (tau=0.005).
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
      break
    fi
  done
fi

if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: no usable CUDA installation found"
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV=${ENV:-builderbench_creative_4_task1}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-10000000}
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2

LOG_DIR="${LOG_DIR:-logs/ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_actorreset/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=td3"
echo "td3 reward=log((1-γ)Q1)  td3_reward_tau=0.5"
echo "log_dir=${LOG_DIR} num_envs=${PPO_NUM_ENVS} max_replay_size=${MAX_REPLAY_SIZE}"
echo "builderbench_use_pd=true pd_duration=5 permute_start_boxes=true"
echo "actor_reset=true (ep_length_mean < 0.8 * nominal; default for BuilderBench JAX)"
echo "td3 defaults otherwise: cross_batch_goals=false, mlp Q, no bilinear; polyak_tau=0.005"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_checkpoint_interval=300 \
  --builderbench_use_pd=true \
  --builderbench_pd_duration=5 \
  --builderbench_permute_start_boxes=true \
  --ppo_rollout_length=50 \
  --ppo_crl_steps_per_iter=10 \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=0 \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --ppo_repr_mode=td3 \
  --noppo_td3_cross_batch_goals \
  --ppo_td3_log_reward \
  --ppo_td3_reward_tau=0.5 \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
