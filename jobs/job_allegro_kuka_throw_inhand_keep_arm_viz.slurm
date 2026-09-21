#!/bin/bash
#SBATCH --job-name=akt_inhand_keeparm_viz
#SBATCH --output=slurm/allegro_kuka_throw_inhand_keep_arm_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_inhand_keep_arm_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# Pose v1 arm + wrist roll + curl + cube in palm, large table. No train.
#   sbatch jobs/job_allegro_kuka_throw_inhand_keep_arm_viz.slurm
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

OUT_DIR="${OUT_DIR:-figs/allegro_kuka_throw/inhand_keep_arm}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  writing ${OUT_DIR}"
echo "pose v1 + wrist -pi (palm-up) ±0.1  curl=1  in-hand  large_table  NOT training"
echo "object goal IN_HAND_KEEP_ARM_GOAL_XYZ=(0.00, -0.22, 0.555)  on table ~26cm in front"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_inhand_keep_arm_viz.py \
  --out-dir="${OUT_DIR}" --pipeline=gpu --n=4 --seed=0
PY=$?
ls -lh "${OUT_DIR}"/keep_arm_*.png || true
exit $PY
