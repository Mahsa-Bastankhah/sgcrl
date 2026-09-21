#!/bin/bash
#SBATCH --job-name=akt_armrand15_initviz
#SBATCH --output=slurm/allegro_kuka_throw_armrand15_init_grid_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_armrand15_init_grid_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# Stills of the hover + finger±10% + arm±15% reset distribution. Not a train.
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

OUT_DIR="${OUT_DIR:-figs/allegro_kuka_throw/armrand15_inits}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  writing ${OUT_DIR}"
echo "8 resets  hover 12cm  curl=0.15  finger_noise=0.10  arm_noise=0.15"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_init_grid_viz.py \
  --out-dir="${OUT_DIR}" --pipeline=gpu \
  --n=8 --curl=0.15 --finger-noise=0.10 --arm-noise=0.15
PY=$?
ls -lh "${OUT_DIR}"/init_*.png "${OUT_DIR}"/init_grid.png || true
exit $PY
