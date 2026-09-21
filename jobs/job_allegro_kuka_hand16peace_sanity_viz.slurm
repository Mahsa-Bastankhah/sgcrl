#!/bin/bash
#SBATCH --job-name=allegro_hand16peace_viz
#SBATCH --output=slurm/allegro_hand16peace_sanity_viz_%j.log
#SBATCH --error=slurm/allegro_hand16peace_sanity_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# Object-free 16-DOF V-sign / peace goal stills. Does NOT train.
unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

OUT_DIR="figs/allegro_kuka_throw/control_sanity"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} mode=hand16peace object_free=true"
echo "goal: V-sign index+middle open, ring curled, thumb tucked"
nvidia-smi || true

python -u scripts/allegro_kuka_control_sanity_viz.py \
  --mode=hand16peace --pipeline=gpu --out-dir="${OUT_DIR}"
ls -lh "${OUT_DIR}"/hand16peace_*.png
