#!/bin/bash
#SBATCH --job-name=c5t2_td3_cat_e1
#SBATCH --output=slurm/ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_actorreset_nopermute_normobs_evalvid_catselect_extrew1_%A_%a.log
#SBATCH --error=slurm/ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_actorreset_nopermute_normobs_evalvid_catselect_extrew1_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# 5-cube creative-5-task2, PD + TD3 logQ tau=0.5 + normobs + nopermute + catselect.
# External hard-success bonus +1 (AFTER return-norm). Eval every 200 / video every 400.
# Same knobs as c5t2_td3_cat, plus extrew1 (matches c5t2_crl_cat_e1).
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
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV=${ENV:-builderbench_creative_5_task2}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-10000000}
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
EXT_REW_SCALE=${EXT_REW_SCALE:-1}

LOG_DIR="${LOG_DIR:-logs/ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_actorreset_nopermute_normobs_evalvid_catselect_extrew1/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=td3"
echo "td3 reward=log((1-γ)Q1)  td3_reward_tau=0.5"
echo "log_dir=${LOG_DIR} num_envs=${PPO_NUM_ENVS}"
echo "max_replay_size=${MAX_REPLAY_SIZE}"
echo "builderbench_use_pd=true pd_duration=5 permute_start_boxes=false goals=MID+offsets(default)"
echo "actor_reset=true (ep_length_mean < 0.8 * nominal)"
echo "ppo_norm_obs=true"
echo "ppo_categorical_select=true (hybrid Gaussian + categorical select)"
echo "ppo_use_external_reward=true ppo_external_reward_scale=${EXT_REW_SCALE} (AFTER return-norm)"
echo "ppo_eval_interval=200 ppo_video_interval=400 (deterministic + live obs_rms)"
echo "td3 defaults otherwise: cross_batch_goals=false, mlp Q, no bilinear; polyak_tau=0.005"
echo "steps_per_iter=$((PPO_NUM_ENVS * 50))"
echo "num_iterations=$((NUM_STEPS / (PPO_NUM_ENVS * 50)))"
python -u -c "
from envs.builderbench_utils import default_fixed_target_goal
import numpy as np
g = default_fixed_target_goal(5, 1)
print('hard_goal (flat)=', g.tolist())
print('hard_goal (cubes)=')
print(np.asarray(g).reshape(-1, 3))
"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

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
  --builderbench_permute_start_boxes=false \
  --ppo_rollout_length=50 \
  --ppo_crl_steps_per_iter=25 \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=200 \
  --ppo_eval_episodes=5 \
  --ppo_video_interval=400 \
  --ppo_video_fps=10 \
  --ppo_skip_first_video=true \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --ppo_repr_mode=td3 \
  --noppo_td3_cross_batch_goals \
  --ppo_td3_log_reward \
  --ppo_td3_reward_tau=0.5 \
  --ppo_norm_obs \
  --ppo_use_external_reward \
  --ppo_external_reward_scale="${EXT_REW_SCALE}" \
  --ppo_categorical_select \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
