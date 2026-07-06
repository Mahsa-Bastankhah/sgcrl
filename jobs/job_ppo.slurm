#!/bin/bash
#SBATCH --job-name=ppo_eightrooms
#SBATCH --output=slurm/ppo_eightrooms_%A_%a.out
#SBATCH --error=slurm/ppo_eightrooms_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=seas
#SBATCH --array=0-3

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
source .venv/bin/activate

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210

ENV=${ENV:-point_EightRooms}
NUM_STEPS=${NUM_STEPS:-10000000}
SEED_OFFSET=${SEED_OFFSET:-120}
SEED=$(( SEED_OFFSET + ${SLURM_ARRAY_TASK_ID:-0} ))

CRL_STEPS=64
ENT_COEF=0.05
MIN_STD=0.01
LOG_DIR="${LOG_DIR:-logs/ppo_eightrooms_kde_dirac/}"
REWARD_MODE="${REWARD_MODE-kde_dirac}"
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} variant=crl64_ent0p05_minstd0p1"
echo "overrides: crl_steps=${CRL_STEPS} ent_coef=${ENT_COEF} min_std=${MIN_STD}"
echo "log_dir=${LOG_DIR} reward_mode=${REWARD_MODE}"
nvidia-smi
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

REWARD_MODE_ARGS=()
if [ -n "${REWARD_MODE}" ]; then
  REWARD_MODE_ARGS=(--ppo_reward_mode="${REWARD_MODE}")
fi

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --uniform_sampling=True \
  --hidden_layer_sizes="256,256,256,256,256,256" \
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
  "${REWARD_MODE_ARGS[@]}"
