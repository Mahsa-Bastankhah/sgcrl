#!/bin/bash
#SBATCH --job-name=ppo_video
#SBATCH --output=slurm/video_%A_%a.log
#SBATCH --error=slurm/video_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=0:45:00
#SBATCH --account=seas
#SBATCH --array=0-5          # matches the training seeds; edit range as needed

# ---------------------------------------------------------------------------
# What this does
#   For each SLURM_ARRAY_TASK_ID (= seed), loads *every* checkpoint under
#   logs/$LOG_SUBDIR/ppo_${ENV}_${SEED}/checkpoints (ckpt_iter_*.pkl +
#   latest.pkl) and writes one mp4 per checkpoint into
#   videos/${ENV}_${SEED}/.  Videos use deterministic actions by default;
#   set STOCHASTIC=1 to sample instead.  Rotation defaults to 180° which
#   is what the metaworld sawyer scene needs to appear upright.
#
# To render a single specific checkpoint instead, set $CHECKPOINT to a
# .pkl file:
#   sbatch --export=ALL,CHECKPOINT=logs/ppo/ppo_sawyer_bin_0/checkpoints/ckpt_iter_5000.pkl \
#          job_rollout_video.slurm
# ---------------------------------------------------------------------------

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

# Load CUDA — same logic as job_ppo.slurm.  Rendering itself is handled by
# mujoco_py (OSMesa), so the GPU is only used by JAX for the policy forward
# pass.  A single forward-per-step is fine on CPU too; if allocating a GPU
# is slow, drop `--gres=gpu:1` above and set `JAX_PLATFORMS=cpu` below.
CUDA_LOADED=0
try_cuda() {
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || true

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
cd /n/fs/mislresearch/sgcrl
source .venv/bin/activate

export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:}${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mb6458/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2     # only needs policy forward pass
export PYTHONUNBUFFERED=1

# OSMesa = software offscreen rendering.  Works on any node without requiring
# a functioning EGL/GLX stack — slower than EGL but robust.  Switch to 'egl'
# if you know the node has working nvidia EGL libraries.
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210

# ---------------------------------------------------------------------------
# Config — edit to taste
# ---------------------------------------------------------------------------
ENV=${ENV:-sawyer_bin}
LOG_SUBDIR=${LOG_SUBDIR:-ppo}                 # logs live under logs/$LOG_SUBDIR/
VIDEO_DIR=${VIDEO_DIR:-videos}
WIDTH=${WIDTH:-640}
HEIGHT=${HEIGHT:-480}
FPS=${FPS:-30}
CAMERA=${CAMERA:-corner}
ROTATE=${ROTATE:-180}                         # N*90 degrees applied per frame
STOCHASTIC=${STOCHASTIC:-0}                   # set to 1 to sample instead of mode

# Offset the array-id so `--array=0-2` maps to seeds 3, 4, 5 — matching the
# retrained PPO runs that wrote logs/ppo/ppo_${ENV}_{3,4,5}/.  Override via
# `sbatch --export=ALL,SEED_OFFSET=0 job_rollout_video.slurm` to restore the
# original 0-indexed mapping.
SEED_OFFSET=${SEED_OFFSET:-3}
SEED=$(( ${SLURM_ARRAY_TASK_ID:-0} + SEED_OFFSET ))
RUN_DIR="logs/${LOG_SUBDIR}/ppo_${ENV}_${SEED}"

# Checkpoint selection:
#   - If $CHECKPOINT is set explicitly (a single .pkl), render just that one.
#   - Otherwise point at the whole checkpoints/ directory so the python
#     script iterates every ckpt_iter_*.pkl + latest.pkl.
CKPT=${CHECKPOINT:-"${RUN_DIR}/checkpoints"}

# Output target:
#   - If rendering a single checkpoint, write videos/<env>_<seed>.mp4.
#   - Otherwise write into videos/<env>_<seed>/ (one mp4 per checkpoint).
if [ -f "${CKPT}" ]; then
  OUT="${VIDEO_DIR}/${ENV}_${SEED}.mp4"
  mkdir -p "${VIDEO_DIR}"
else
  OUT="${VIDEO_DIR}/${ENV}_${SEED}/"
  mkdir -p "${OUT}"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  seed=${SEED}"
echo "checkpoint=${CKPT}"
echo "output=${OUT}"
echo "rotate=${ROTATE} deg"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"

if [ ! -e "${CKPT}" ]; then
  echo "ERROR: checkpoint path not found: ${CKPT}"
  echo "Run dir contents (if any):"
  ls -la "${RUN_DIR}/checkpoints/" 2>/dev/null || echo "  (no such dir)"
  exit 2
fi
if [ -d "${CKPT}" ] && ! ls "${CKPT}"/ckpt_iter_*.pkl "${CKPT}"/latest.pkl \
      >/dev/null 2>&1; then
  echo "ERROR: no ckpt_iter_*.pkl or latest.pkl inside ${CKPT}"
  ls -la "${CKPT}" || true
  exit 2
fi

# ---------------------------------------------------------------------------
# Flag assembly — optional --stochastic, optional --camera override.
# ---------------------------------------------------------------------------
EXTRA_FLAGS=()
if [ "${STOCHASTIC}" = "1" ]; then
  EXTRA_FLAGS+=(--stochastic)
fi
if [ -n "${CAMERA}" ]; then
  EXTRA_FLAGS+=(--camera "${CAMERA}")
fi

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
python -u scripts/ppo_rollout_video.py \
    --checkpoint="${CKPT}" \
    --env="${ENV}" \
    --output="${OUT}" \
    --width="${WIDTH}" \
    --height="${HEIGHT}" \
    --fps="${FPS}" \
    --rotate="${ROTATE}" \
    --seed="${SEED}" \
    "${EXTRA_FLAGS[@]}"

echo "Done. Output:"
if [ -d "${OUT}" ]; then
  ls -lh "${OUT}"
else
  ls -lh "${OUT}"
fi
