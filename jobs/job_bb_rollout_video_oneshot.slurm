#!/bin/bash
#SBATCH --job-name=bb_rollout_vid
#SBATCH --output=slurm/bb_rollout_vid_%j.log
#SBATCH --error=slurm/bb_rollout_vid_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# One-shot BuilderBench rollout videos for a run's checkpoints.
# Required env vars: RUN_DIR, ENV_NAME, RUN_TAG
# Optional: SEED (default 0), OUT_DIR, CHECKPOINT (default RUN_DIR/checkpoints)
unset LD_PRELOAD
module purge

try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -n "${CUDA_HOME:-}" ] && return 0
    for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
      [ -d "${_cand}/lib64" ] && export CUDA_HOME="${_cand}" && return 0
    done
    return 1
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || try_cuda cudatoolkit/12.4 || true
if [ -z "${CUDA_HOME:-}" ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
# Override script setdefault('JAX_PLATFORMS','cpu') so GPU is used.
export JAX_PLATFORMS=cuda

: "${RUN_DIR:?RUN_DIR required}"
: "${ENV_NAME:?ENV_NAME required}"
: "${RUN_TAG:?RUN_TAG required}"
SEED="${SEED:-0}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints}"
LOG_BASENAME="$(basename "$(dirname "${RUN_DIR}")")"
OUT_DIR="${OUT_DIR:-videos/builderbench/${LOG_BASENAME}}"
mkdir -p slurm "${OUT_DIR}"

echo "RUN_DIR=${RUN_DIR}"
echo "ENV_NAME=${ENV_NAME}  RUN_TAG=${RUN_TAG}  SEED=${SEED}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUT_DIR=${OUT_DIR}"
ls "${CHECKPOINT}"/ckpt_iter_*.pkl 2>/dev/null | wc -l | awk '{print "n_ckpts="$1}'
nvidia-smi || true

python -u scripts/ppo_builderbench_rollout_video.py \
  --checkpoint="${CHECKPOINT}" \
  --env="${ENV_NAME}" \
  --output="${OUT_DIR}/" \
  --run_tag="${RUN_TAG}" \
  --seed="${SEED}" \
  --skip_existing

PY_EXIT=$?
ls -lh "${OUT_DIR}/${RUN_TAG}"_iter_*.mp4 2>&1 | tail -20 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
