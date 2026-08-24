#!/bin/bash
#SBATCH --job-name=akt_ckpt_vid
#SBATCH --output=slurm/allegro_kuka_throw_ckpt_video_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_ckpt_video_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# Policy rollout video from an AllegroKukaThrow PPO checkpoint.
#   CKPT=.../ckpt_iter_0000100.pkl OUT=videos/....mp4 \
#     sbatch jobs/job_allegro_kuka_throw_ckpt_video.slurm
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

CKPT="${CKPT:-}"
OUT="${OUT:-videos/allegro_kuka_throw_ckpt.mp4}"
SEED="${SEED:-0}"
if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "CKPT missing or not a file: ${CKPT}"
  exit 1
fi
mkdir -p slurm "$(dirname "${OUT}")"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  ckpt=${CKPT}  out=${OUT}"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_ckpt_video.py \
  --checkpoint="${CKPT}" \
  --output="${OUT}" \
  --num-steps=300 \
  --fps=30 \
  --seed="${SEED}" \
  --pipeline=gpu
PY_EXIT=$?
ls -lh "${OUT}" || true
exit $PY_EXIT
