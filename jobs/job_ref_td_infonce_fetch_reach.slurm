#!/bin/bash
#SBATCH --job-name=ref_fetch_reach
#SBATCH --output=slurm/ref_td_infonce_fetch_reach_%j.out
#SBATCH --error=slurm/ref_td_infonce_fetch_reach_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=seas

# Official upstream TD-InfoNCE (README): fetch_reach, 500k actor steps, local_mp.
# NOT adapted to sawyer / PPO tasks — uses external/td_infonce envs only.

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
export TF_FORCE_GPU_ALLOW_GROWTH=true

ENV_NAME="${ENV_NAME:-fetch_reach}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-500000}"
LOG_DIR="${LOG_DIR:-/n/fs/mislresearch/sgcrl/logs/ref_td_infonce_${ENV_NAME}}"

mkdir -p slurm

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "official ref TD-InfoNCE env=${ENV_NAME} seed=${SEED} max_steps=${MAX_STEPS}"
echo "log_dir=${LOG_DIR}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

bash scripts/run_ref_td_infonce_official.sh
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
