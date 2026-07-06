#!/bin/bash
#SBATCH --job-name=wbc_eightrooms
#SBATCH --output=slurm/wbc_eightrooms_%A_%a.log
#SBATCH --error=slurm/wbc_eightrooms_%A_%a.log
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

ENV="${ENV:-point_EightRooms}"
NUM_STEPS="${NUM_STEPS:-10000000}"
SEED_OFFSET="${SEED_OFFSET:-120}"
SEED=$(( SEED_OFFSET + ${SLURM_ARRAY_TASK_ID:-0} ))
LOG_DIR="${LOG_DIR:-logs/wbc_eightrooms_v2/}"

CRL_STEPS=64
ENT_COEF=0.1
MIN_STD=0.05
PPO_NUM_EPOCHS=1
DISCOUNT_FACTOR=0.99

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  seed=${SEED}  log_dir=${LOG_DIR}"
echo "hyperparams: epochs=${PPO_NUM_EPOCHS} ent_coef=${ENT_COEF} min_std=${MIN_STD}"
nvidia-smi
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u wbc_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_num_epochs="${PPO_NUM_EPOCHS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --discount="${DISCOUNT_FACTOR}" \
  --uniform_sampling=True \
  --hidden_layer_sizes="256,256,256,256,256,256"
