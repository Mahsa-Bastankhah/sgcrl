#!/bin/bash
#SBATCH --job-name=mpo_bb_vid
#SBATCH --output=slurm/mpo_bb_vid_new_%j.log
#SBATCH --error=slurm/mpo_bb_vid_new_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# Short one-shot: render ONLY MPO checkpoints listed in CKPT_LIST_FILE.
# Required: ENV_NAME, RUN_TAG, OUT_DIR, CKPT_LIST_FILE
# Optional: SEED (default 0). Walltime overridden by sbatch --time from the watcher.
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
export JAX_PLATFORMS=cuda

: "${ENV_NAME:?ENV_NAME required}"
: "${RUN_TAG:?RUN_TAG required}"
: "${OUT_DIR:?OUT_DIR required}"
: "${CKPT_LIST_FILE:?CKPT_LIST_FILE required (one checkpoint path per line)}"
SEED="${SEED:-0}"
mkdir -p slurm "${OUT_DIR}"

echo "ENV_NAME=${ENV_NAME}  RUN_TAG=${RUN_TAG}  SEED=${SEED}"
echo "OUT_DIR=${OUT_DIR}"
echo "CKPT_LIST_FILE=${CKPT_LIST_FILE}"
nvidia-smi || true

if [[ ! -f "${CKPT_LIST_FILE}" ]]; then
  echo "[mpo_bb_vid_new] missing list file: ${CKPT_LIST_FILE}"
  exit 1
fi

PY_EXIT=0
while IFS= read -r ckpt || [[ -n "${ckpt}" ]]; do
  [[ -z "${ckpt}" || "${ckpt}" =~ ^# ]] && continue
  if [[ ! -f "${ckpt}" ]]; then
    echo "[mpo_bb_vid_new] missing ckpt: ${ckpt}"
    PY_EXIT=1
    continue
  fi
  echo "[mpo_bb_vid_new] rendering ${ckpt}"
  python -u scripts/mpo_builderbench_rollout_video.py \
    --checkpoint="${ckpt}" \
    --env="${ENV_NAME}" \
    --output="${OUT_DIR}/" \
    --run_tag="${RUN_TAG}" \
    --seed="${SEED}" \
    --skip_existing || PY_EXIT=$?
done < "${CKPT_LIST_FILE}"

ls -lh "${OUT_DIR}/${RUN_TAG}"_*.mp4 2>&1 | tail -30 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
