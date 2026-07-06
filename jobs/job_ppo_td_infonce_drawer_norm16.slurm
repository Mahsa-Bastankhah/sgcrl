#!/bin/bash
#SBATCH --job-name=td_infonce_drawer
#SBATCH --output=slurm/ppo_td_infonce_drawer_norm16_%j.out
#SBATCH --error=slurm/ppo_td_infonce_drawer_norm16_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=seas

# TD InfoNCE drawer: repr_norm, repr_dim=16, target_tau=0.995,
# actor_learning_rate=5e-5, no LR anneal,
# random warmup → one CRL block @ min replay → π + PPO,
# a' action histogram logging (100 bins/dim).

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
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || true
if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit."
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

ENV=sawyer_drawer_open
SEED=${SEED:-0}
NUM_STEPS=${NUM_STEPS:-10000000}
LOG_DIR="${LOG_DIR:-logs/ppo_td_infonce_drawer_norm16/}"
RUN_DIR="${LOG_DIR%/}/ppo_${ENV}_${SEED}"

# Always start from scratch (no checkpoint resume).
if [ -d "${RUN_DIR}" ]; then
  echo "Fresh start: removing ${RUN_DIR}"
  rm -rf "${RUN_DIR}"
fi
PPO_NUM_ENVS=${PPO_NUM_ENVS:-4}
CRL_STEPS=${CRL_STEPS:-128}
TD_TARGET_TAU=${TD_TARGET_TAU:-0.995}
REPR_TAU=0.0
REPR_DIM=16
ACTOR_LR=5e-5
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
ACTION_BINS=${ACTION_BINS:-100}

mkdir -p slurm

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "env=${ENV} seed=${SEED} td_infonce target_tau=${TD_TARGET_TAU} repr_tau=${REPR_TAU} repr_dim=${REPR_DIM} repr_norm=true actor_lr=${ACTOR_LR} anneal_lr=false random_warmup=true action_bins=${ACTION_BINS}"
echo "log_dir=${LOG_DIR} num_steps=${NUM_STEPS} num_envs=${PPO_NUM_ENVS} crl_steps=${CRL_STEPS}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_crl_loss_direction=td_infonce \
  --ppo_crl_repr_tau="${REPR_TAU}" \
  --ppo_crl_td_target_tau="${TD_TARGET_TAU}" \
  --repr_norm \
  --repr_dim="${REPR_DIM}" \
  --actor_learning_rate="${ACTOR_LR}" \
  --noppo_anneal_lr \
  --ppo_random_warmup_before_policy \
  --td_infonce_action_bins="${ACTION_BINS}" \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
