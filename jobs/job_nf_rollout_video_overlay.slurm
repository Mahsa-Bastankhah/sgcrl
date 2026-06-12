#!/bin/bash
#SBATCH --job-name=nf_vid_ov
#SBATCH --output=slurm/nf_video_overlay_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas
#SBATCH --array=0-1

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
export TMPDIR=/n/fs/mislresearch/mb6458-tmp

SEED=0
SAMPLE_SPAN=${SAMPLE_SPAN:-0.06}
SAMPLE_GRID=${SAMPLE_GRID:-3}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ENV=sawyer_reach
    LOG_SUBDIR=ppo_nf_reach_debug
    ;;
  1)
    ENV=sawyer_drawer_open
    LOG_SUBDIR=ppo_nf_drawer
    ;;
  *)
    echo "Unknown array id ${SLURM_ARRAY_TASK_ID}"; exit 1
    ;;
esac

CKPT="logs/${LOG_SUBDIR}/ppo_${ENV}_${SEED}/checkpoints"
OUT="videos/${LOG_SUBDIR}_overlay_${ENV}_${SEED}/"

echo "env=${ENV}  seed=${SEED}  ckpt=${CKPT}  out=${OUT}"

python -u scripts/ppo_nf_rollout_video_overlay.py \
  --checkpoint="${CKPT}" \
  --env="${ENV}" \
  --output="${OUT}" \
  --seed="${SEED}" \
  --sample_span="${SAMPLE_SPAN}" \
  --sample_grid="${SAMPLE_GRID}"

echo "Done:"
ls -lh "${OUT}" || true
