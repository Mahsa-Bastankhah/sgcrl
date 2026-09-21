#!/bin/bash
#SBATCH --job-name=akt_goal_joint_viz
#SBATCH --output=slurm/allegro_kuka_throw_goal_joint_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_goal_joint_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# IK + stills of the curled-paddle hand-joint goal at table center.
#   sbatch jobs/job_allegro_kuka_throw_goal_joint_viz.slurm
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

OUT_DIR="${OUT_DIR:-figs/allegro_kuka_throw}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  writing ${OUT_DIR}"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_goal_joint_viz.py \
  --variant=tight10 --goal=center --large-table \
  --out-dir="${OUT_DIR}" --pipeline=gpu
ls -lh "${OUT_DIR}"/goal_joint_center_large_table_tight10_*.png \
  "${OUT_DIR}"/goal_joint_center_large_table_tight10_q.txt
