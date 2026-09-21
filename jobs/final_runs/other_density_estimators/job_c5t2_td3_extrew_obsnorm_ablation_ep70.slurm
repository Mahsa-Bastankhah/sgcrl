#!/bin/bash
#SBATCH --job-name=bb_c5t2_td3_ablate
#SBATCH --output=slurm/final_runs/other_density_estimators/bb_c5t2_td3_ablate_%A_%a.log
#SBATCH --error=slurm/final_runs/other_density_estimators/bb_c5t2_td3_ablate_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas
#SBATCH --array=0-5

# C5T2 TD3 external-reward/observation-normalization ablation.
# Configs: {25 updates + norm obs, 25 updates + no norm obs,
#           45 updates + no norm obs} x new seeds {2, 3}.
# PD duration 5 and MJ horizon 350 yield T=70 policy actions.
unset -f python python3 nvidia-smi mkdir pkill 2>/dev/null || true
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

CONFIG=$((SLURM_ARRAY_TASK_ID / 2))
SEED=$((SLURM_ARRAY_TASK_ID % 2 + 2))
ENV=builderbench_creative_5_task2
NORM_OBS=false

case "${CONFIG}" in
  0)
    CRL_STEPS=25
    NORM_OBS=true
    NORM_TOKEN=normobs
    ;;
  1)
    CRL_STEPS=25
    NORM_TOKEN=nonormobs
    ;;
  2)
    CRL_STEPS=45
    NORM_TOKEN=nonormobs
    ;;
  *)
    echo "ERROR: invalid CONFIG=${CONFIG}"
    exit 2
    ;;
esac

RUN_BASE="ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_actorreset_nopermute_${NORM_TOKEN}_evalvid_catselect_extrew1"
LOG_STEM="${RUN_BASE}_ep70_200m_crl${CRL_STEPS}_eval10_warp_4h"
LOG_DIR="logs/final_runs/other_density_estimators/${LOG_STEM}/"
mkdir -p slurm/final_runs/other_density_estimators "${LOG_DIR}"

METHOD_ARGS=(
  --ppo_repr_mode=td3
  --ppo_td3_cross_batch_goals=false
  --ppo_td3_bilinear=false
  --ppo_td3_fb_loss=false
  --ppo_td3_tau=0.005
  --ppo_td3_log_reward=true
  --ppo_td3_reward_tau=0.5
)

echo "array=${SLURM_ARRAY_TASK_ID} config=${CONFIG} task=creative5_task2 method=td3 seed=${SEED}"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL} crl_steps=${CRL_STEPS} ppo_eval_interval=10"
echo "mj_episode_length=350 pd_duration=5 rollout_length=50"
echo "permute=false fixed_start_x=unset norm_obs=${NORM_OBS}"
echo "min_std=0.01 extrew=true external_reward_scale=1 video_interval=400"
echo "log_dir=${LOG_DIR}"
nvidia-smi || true
python -u -c "import os, sgcrl_jax_acme_compat, warp, jax; print('warp', warp.__version__); print('BUILDERBENCH_MJX_IMPL', os.environ.get('BUILDERBENCH_MJX_IMPL')); print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps=200000000 \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs=1024 \
  --ppo_ent_coef=0.05 \
  --ppo_anneal_ent_coef=false \
  --ppo_anneal_lr=true \
  --ppo_actor_min_std=0.01 \
  --ppo_discount=0.99 \
  --ppo_clip_coef=0.2 \
  --ppo_checkpoint_interval=100 \
  --builderbench_use_pd=true \
  --builderbench_pd_duration=5 \
  --builderbench_mj_episode_length=350 \
  --builderbench_permute_start_boxes=false \
  --ppo_rollout_length=50 \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=10 \
  --ppo_eval_episodes=5 \
  --ppo_video_interval=400 \
  --ppo_video_fps=10 \
  --ppo_skip_first_video=true \
  --max_replay_size=10000000 \
  "${METHOD_ARGS[@]}" \
  --ppo_norm_obs="${NORM_OBS}" \
  --ppo_use_external_reward=true \
  --ppo_external_reward_scale=1 \
  --ppo_categorical_select=true \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
