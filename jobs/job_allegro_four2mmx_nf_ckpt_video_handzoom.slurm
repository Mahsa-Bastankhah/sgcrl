#!/bin/bash
#SBATCH --job-name=allegro_mmx_handzoom
#SBATCH --output=slurm/allegro_four2mmx_nf_ckpt_video_handzoom_%j.log
#SBATCH --error=slurm/allegro_four2mmx_nf_ckpt_video_handzoom_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --account=seas

# Re-render control-sanity ckpt videos with hand close-up + GOAL-on-top stack.
# Usage:
#   sbatch jobs/job_allegro_four2mmx_nf_ckpt_video_handzoom.slurm \
#     /path/to/ckpt_iter_0000100.pkl [/path/to/out.mp4]

unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

CKPT="${1:?pass checkpoint path as first arg}"
OUT="${2:-}"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi
if [ -z "${OUT}" ]; then
  RUN_DIR="$(dirname "$(dirname "${CKPT}")")"
  ITER="$(basename "${CKPT}" .pkl)"
  OUT="${RUN_DIR}/videos/${ITER}_stoch_handzoom_goal.mp4"
fi
mkdir -p "$(dirname "${OUT}")" slurm

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CKPT=${CKPT}"
echo "OUT=${OUT}"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_ckpt_video.py \
  --checkpoint="${CKPT}" \
  --output="${OUT}" \
  --episodes=1 \
  --seed=0 \
  --pipeline=gpu

ls -lh "${OUT}"
