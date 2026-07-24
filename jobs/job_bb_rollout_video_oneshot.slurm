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
# Optional: SEED (default 0), OUT_DIR, CHECKPOINT (default RUN_DIR/checkpoints),
#           CKPT_STRIDE (default 1; e.g. 3 ≈ every 3rd / ~1/3 of ckpts,
#           4 ≈ every 4th / ~1/4; always keeps first + last).
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
CKPT_STRIDE="${CKPT_STRIDE:-1}"
LOG_BASENAME="$(basename "$(dirname "${RUN_DIR}")")"
OUT_DIR="${OUT_DIR:-videos/builderbench/${LOG_BASENAME}}"
mkdir -p slurm "${OUT_DIR}"

# Optional subsample: symlink every CKPT_STRIDE-th milestone (+ last) so we
# do not render the full checkpoint timeline.
RENDER_CKPT="${CHECKPOINT}"
STAGE_DIR=""
if [ "${CKPT_STRIDE}" -gt 1 ] 2>/dev/null; then
  STAGE_DIR="${TMPDIR:-/tmp}/bb_vid_stride_${SLURM_JOB_ID:-$$}_${RUN_TAG}"
  mkdir -p "${STAGE_DIR}"
  python -u - <<PY
import glob, os, re, sys
src = "${CHECKPOINT}"
stage = "${STAGE_DIR}"
stride = max(1, int("${CKPT_STRIDE}"))
ckpts = sorted(
    glob.glob(os.path.join(src, "ckpt_iter_*.pkl")),
    key=lambda p: int(re.search(r"ckpt_iter_(\d+)", p).group(1)),
)
if not ckpts:
    print(f"ERROR: no ckpt_iter_*.pkl in {src}", file=sys.stderr)
    sys.exit(2)
idxs = list(range(0, len(ckpts), stride))
if idxs[-1] != len(ckpts) - 1:
    idxs.append(len(ckpts) - 1)
picks = [ckpts[i] for i in idxs]
for p in picks:
    dst = os.path.join(stage, os.path.basename(p))
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(p), dst)
    print("pick", os.path.basename(p))
print(f"stride={stride}: {len(ckpts)} -> {len(picks)} staged in {stage}")
PY
  RENDER_CKPT="${STAGE_DIR}"
fi

echo "RUN_DIR=${RUN_DIR}"
echo "ENV_NAME=${ENV_NAME}  RUN_TAG=${RUN_TAG}  SEED=${SEED}"
echo "CHECKPOINT=${CHECKPOINT}  CKPT_STRIDE=${CKPT_STRIDE}"
echo "RENDER_CKPT=${RENDER_CKPT}"
echo "OUT_DIR=${OUT_DIR}"
ls "${RENDER_CKPT}"/ckpt_iter_*.pkl 2>/dev/null | wc -l | awk '{print "n_ckpts="$1}'
nvidia-smi || true

python -u scripts/ppo_builderbench_rollout_video.py \
  --checkpoint="${RENDER_CKPT}" \
  --env="${ENV_NAME}" \
  --output="${OUT_DIR}/" \
  --run_tag="${RUN_TAG}" \
  --seed="${SEED}" \
  --skip_existing

PY_EXIT=$?
[ -n "${STAGE_DIR}" ] && rm -rf "${STAGE_DIR}"
ls -lh "${OUT_DIR}/${RUN_TAG}"_iter_*.mp4 2>&1 | tail -20 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
