#!/bin/bash
#SBATCH --job-name=allegro_idxthumb_validate
#SBATCH --output=slurm/allegro_index_thumb_straight_validation_%j.log
#SBATCH --error=slurm/allegro_index_thumb_straight_validation_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --account=seas

set -euo pipefail

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

OUT_DIR="figs/allegro_kuka_throw/index_thumb_straight_validation"
mkdir -p slurm "${OUT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} mode=index_thumb_straight"
echo "Validation: randomized curled reset, straight index/thumb goal, trim_sa dimensions 16/8/8"
nvidia-smi || true

python -u scripts/allegro_kuka_control_sanity_viz.py \
  --mode=index_thumb_straight \
  --pipeline=gpu \
  --trim-sa \
  --randomize-init \
  --out-dir="${OUT_DIR}"

ls -lh "${OUT_DIR}"/index_thumb_straight_start.png \
  "${OUT_DIR}"/index_thumb_straight_goal.png

python -u scripts/allegro_kuka_control_sanity_viz.py \
  --mode=index_thumb_straight \
  --pipeline=gpu \
  --trim-sa \
  --randomize-init \
  --oracle-action \
  --num-envs=1024 \
  --steps=150
