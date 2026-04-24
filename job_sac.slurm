#!/bin/bash
#SBATCH --job-name=q_sac
#SBATCH --output=slurm/q_sac_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --account=seas
#SBATCH --array=0-0          # 4 seeds; change range to run more

module purge

# Drop any broken inherited CUDA paths from the submit host.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

# Load CUDA — same logic as job.slurm.
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
  echo "ERROR: Could not load cudatoolkit. Run 'module avail cudatoolkit' and edit this script."
  exit 1
fi

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
cd /n/fs/mislresearch/sgcrl
source .venv/bin/activate

# LD_LIBRARY_PATH — identical order to job.slurm.
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mb6458/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210

# ---------------------------------------------------------------------------
# Config — edit to taste
# ---------------------------------------------------------------------------
ENV=sawyer_bin
# ALG=q_sac
ALG=contrastive_cpc
NUM_STEPS=8000000
LOG_DIR=logs/${ALG}/

SEED_OFFSET=${SEED_OFFSET:-19}
SEED=$(( ${SLURM_ARRAY_TASK_ID:-0} + SEED_OFFSET ))

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  alg=${ALG}  seed=${SEED}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_HOME: ${CUDA_HOME}"
nvidia-smi

# Verify JAX sees the GPU (mirrors job.slurm diagnostic).
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python -u lp_contrastive.py \
    --lp_launch_type=local_mt \
    --env="${ENV}" \
    --alg="${ALG}" \
    --seed="${SEED}" \
    --num_steps="${NUM_STEPS}" \
    --log_dir_path="${LOG_DIR}" \
