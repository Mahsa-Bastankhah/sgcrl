#!/bin/bash
#SBATCH --job-name=c4t2_crl10
#SBATCH --output=slurm/final_runs/other_density_estimators/c4t2_crl_crl10_%A_%a.log
#SBATCH --error=slurm/final_runs/other_density_estimators/c4t2_crl_crl10_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --account=seas
#SBATCH --array=0-1

# Reproduce the successful c4t2 state-only CRL setup on Warp, changing only
# eval cadence (10) and this ablation's estimator updates/iteration (10).
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
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || \
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
[ "$CUDA_LOADED" != 1 ] && echo "ERROR: CUDA unavailable" && exit 1

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=warp
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV=builderbench_creative_4_task2
SEED=${SLURM_ARRAY_TASK_ID}
NUM_STEPS=200000000
PPO_NUM_ENVS=1024
MAX_REPLAY_SIZE=10000000
CRL_STEPS=10
LOG_STEM=ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_catselect_stateonly_extrew1_ep50_200m_crl10_eval10_warp_2h30
LOG_DIR="logs/final_runs/other_density_estimators/${LOG_STEM}/"
mkdir -p slurm/final_runs/other_density_estimators "${LOG_DIR}"

echo "env=${ENV} seed=${SEED} steps=${NUM_STEPS} repr=crl state_only=true"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL} ppo_crl_steps_per_iter=${CRL_STEPS}"
echo "ppo_eval_interval=10 log_dir=${LOG_DIR}"
nvidia-smi || true
python -u -c "import os, sgcrl_jax_acme_compat, warp, jax; print('warp', warp.__version__); print('BUILDERBENCH_MJX_IMPL', os.environ.get('BUILDERBENCH_MJX_IMPL')); print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --ppo_ent_coef=0.05 \
  --ppo_anneal_ent_coef=false \
  --ppo_actor_min_std=0.01 \
  --ppo_discount=0.99 \
  --ppo_clip_coef=0.2 \
  --ppo_checkpoint_interval=100 \
  --builderbench_use_pd=true \
  --builderbench_pd_duration=5 \
  --builderbench_permute_start_boxes=true \
  --builderbench_mj_episode_length=250 \
  --ppo_rollout_length=50 \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=10 \
  --ppo_eval_episodes=5 \
  --ppo_video_interval=0 \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --ppo_repr_mode=crl \
  --crl_state_only=true \
  --ppo_crl_repr_tau=0.5 \
  --ppo_norm_obs=false \
  --ppo_use_external_reward=true \
  --ppo_external_reward_scale=1 \
  --ppo_categorical_select=true \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
