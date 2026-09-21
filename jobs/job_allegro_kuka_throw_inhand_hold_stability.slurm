#!/bin/bash
#SBATCH --job-name=akt_keeparm_hold_stable
#SBATCH --output=slurm/allegro_kuka_throw_inhand_hold_stability_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_inhand_hold_stability_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# Probe: cube must stay in the palm for 30 hold-pose steps. Not a train.
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

OUT_DIR="${OUT_DIR:-figs/allegro_kuka_throw/inhand_keep_arm/hold_stable}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  keep-arm hold stability  NOT training"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_inhand_hold_stability.py \
  --out-dir="${OUT_DIR}" --pipeline=gpu --steps=30
PY=$?
ls -lh "${OUT_DIR}"/hold_t*.png || true
exit $PY
