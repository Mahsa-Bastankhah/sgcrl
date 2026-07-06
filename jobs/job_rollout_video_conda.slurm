#!/bin/bash
#SBATCH --job-name=ppo_video_conda
#SBATCH --output=slurm/video_conda_%A_%a.out
#SBATCH --error=slurm/video_conda_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
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
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || true

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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export TMPDIR=/n/fs/mislresearch/mb6458-tmp

# --- config (override via sbatch --export=ALL,ENV=...) ---
ENV=${ENV:-sawyer_bin}
LOG_SUBDIR=${LOG_SUBDIR:-ppo_bin}
SEED_OFFSET=${SEED_OFFSET:-0}
SEED=$(( ${SLURM_ARRAY_TASK_ID:-0} + SEED_OFFSET ))
VIDEO_DIR=${VIDEO_DIR:-videos}
WIDTH=${WIDTH:-640}
HEIGHT=${HEIGHT:-480}
FPS=${FPS:-30}
CAMERA=${CAMERA:-corner}
ROTATE=${ROTATE:-180}
STOCHASTIC=${STOCHASTIC:-0}

RUN_DIR="logs/${LOG_SUBDIR}/ppo_${ENV}_${SEED}"
CKPT=${CHECKPOINT:-"${RUN_DIR}/checkpoints"}

if [ -f "${CKPT}" ]; then
  OUT="${VIDEO_DIR}/${ENV}_${SEED}.mp4"
  mkdir -p "${VIDEO_DIR}"
else
  OUT="${VIDEO_DIR}/${LOG_SUBDIR}_${ENV}_${SEED}/"
  mkdir -p "${OUT}"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  seed=${SEED}  log_subdir=${LOG_SUBDIR}"
echo "checkpoint=${CKPT}"
echo "output=${OUT}"

if [ ! -e "${CKPT}" ]; then
  echo "ERROR: checkpoint path not found: ${CKPT}"; exit 2
fi

EXTRA_FLAGS=()
[ "${STOCHASTIC}" = "1" ] && EXTRA_FLAGS+=(--stochastic)
[ -n "${CAMERA}" ]        && EXTRA_FLAGS+=(--camera "${CAMERA}")

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
ls -lh "${OUT}"
