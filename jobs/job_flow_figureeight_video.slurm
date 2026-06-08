#!/bin/bash
#SBATCH --job-name=flow_video
#SBATCH --output=slurm/flow_video_%A_%a.out
#SBATCH --error=slurm/flow_video_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

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

# SUMO (figure-eight traffic sim)
export SUMO_HOME=/u/mb6458/sumo_binaries/bin
export PATH="${SUMO_HOME}:${PATH}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp

# Config — override via --export
ENV=${ENV:-flow_figureeight}
SEED_OFFSET=${SEED_OFFSET:-0}
SEED=$(( SEED_OFFSET + ${SLURM_ARRAY_TASK_ID:-0} ))
STEPS=${STEPS:-300}
FPS=${FPS:-20}
FRAME_SKIP=${FRAME_SKIP:-1}
LOG_DIR=${LOG_DIR:-logs/ppo}

CHECKPOINT=${CHECKPOINT:-"${LOG_DIR}/ppo_${ENV}_${SEED}/checkpoints/latest.pkl"}
OUTPUT="videos/${ENV}_${SEED}.gif"

mkdir -p slurm videos

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  seed=${SEED}  steps=${STEPS}"
echo "checkpoint=${CHECKPOINT}"
echo "output=${OUTPUT}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend())"

python -u scripts/flow_figureeight_video.py \
  --env="${ENV}" \
  --checkpoint="${CHECKPOINT}" \
  --steps="${STEPS}" \
  --fps="${FPS}" \
  --frame_skip="${FRAME_SKIP}" \
  --output="${OUTPUT}"

echo "Done. Output:"
ls -lh "${OUTPUT}" 2>/dev/null || echo "(file not found)"
