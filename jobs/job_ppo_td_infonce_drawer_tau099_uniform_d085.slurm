#!/bin/bash
#SBATCH --job-name=td_inf_drawer
#SBATCH --output=slurm/ppo_td_infonce_drawer_tau099_uniform_d085_%j.out
#SBATCH --error=slurm/ppo_td_infonce_drawer_tau099_uniform_d085_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=seas

# TD InfoNCE sawyer_drawer_open: target_tau=0.99, uniform-random warmup init,
# discount=0.85 for CRL + PPO; all other knobs at codebase defaults.

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

ENV=sawyer_drawer_open
SEED=${SEED:-0}
NUM_STEPS=${NUM_STEPS:-8000000}
LOG_DIR="${LOG_DIR:-logs/ppo_td_infonce_drawer_tau099_uniform_d085/}"
TD_TARGET_TAU=0.99
DISCOUNT_FACTOR=0.85

mkdir -p slurm

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "env=${ENV} seed=${SEED} td_infonce target_tau=${TD_TARGET_TAU} discount=${DISCOUNT_FACTOR} uniform_warmup=true"
echo "log_dir=${LOG_DIR} num_steps=${NUM_STEPS}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --discount="${DISCOUNT_FACTOR}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_crl_loss_direction=td_infonce \
  --ppo_crl_td_target_tau="${TD_TARGET_TAU}" \
  --ppo_random_warmup_before_policy
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
