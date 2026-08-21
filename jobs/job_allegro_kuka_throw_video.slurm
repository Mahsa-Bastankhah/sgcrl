#!/bin/bash
#SBATCH --job-name=akt_vid
#SBATCH --output=slurm/allegro_kuka_throw_video_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_video_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# One-episode preview video of AllegroKukaThrow (random policy, GPU camera).
unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Headless Vulkan ICD for Isaac Gym cameras on Slurm nodes.
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

OUT="${OUT:-videos/allegro_kuka_throw_preview.mp4}"
mkdir -p slurm videos

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  writing ${OUT}"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_video.py \
  --output="${OUT}" \
  --num-steps=300 \
  --fps=30 \
  --pipeline=gpu
PY_EXIT=$?
ls -lh "${OUT}" || true
exit $PY_EXIT
