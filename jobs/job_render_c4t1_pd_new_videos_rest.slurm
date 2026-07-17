#!/bin/bash
#SBATCH --job-name=bb_vid_c4t1_rest
#SBATCH --output=slurm/render_c4t1_pd_new_videos_rest_%j.log
#SBATCH --error=slurm/render_c4t1_pd_new_videos_rest_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=seas

unset LD_PRELOAD

module purge
try_cuda() {
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -n "${CUDA_HOME:-}" ] || return 1
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || true

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CKPT_DIR=logs/ppo_builderbench_creative4_task1_e1024_pd_new/ppo_builderbench_creative_4_task1_1/checkpoints
OUT_DIR=videos/builderbench/ppo_builderbench_creative4_task1_e1024_pd_new/

mkdir -p slurm "$OUT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "rendering missing creative4 task1 pd_new videos"

for ITER in 4050 4200 4350 4500 4650 4800 4881; do
  CKPT=$(printf "${CKPT_DIR}/ckpt_iter_%07d.pkl" "$ITER")
  echo "=== rendering ${CKPT} ==="
  python -u scripts/ppo_builderbench_rollout_video.py \
    --checkpoint="${CKPT}" \
    --env=builderbench_creative_4_task1 \
    --output="${OUT_DIR}" \
    --run_tag=c4t1_pd_new \
    --seed=1
done

echo "=== rendering latest ==="
python -u scripts/ppo_builderbench_rollout_video.py \
  --checkpoint="${CKPT_DIR}/latest.pkl" \
  --env=builderbench_creative_4_task1 \
  --output="${OUT_DIR}" \
  --run_tag=c4t1_pd_new \
  --seed=1
