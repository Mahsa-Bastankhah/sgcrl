#!/bin/bash
#SBATCH --job-name=akt_palm_viz
#SBATCH --output=slurm/allegro_kuka_throw_palm_goal_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_palm_goal_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# Render slide palm stills: current (0.17) vs lip (0.22) vs over-gap (0.25).
#   sbatch jobs/job_allegro_kuka_throw_palm_goal_viz.slurm
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

echo "slide_compare palms: (0.17,0.08,0.57) (0.22,0.08,0.57) (0.25,0.08,0.55)"
python -u scripts/allegro_kuka_throw_palm_goal_viz.py \
  --output-dir="${OUT_DIR}" \
  --pipeline=gpu \
  --scene=slide_compare \
  --ik-iters=120
PY_EXIT=$?
ls -lh "${OUT_DIR}"/palm_goal_slide_compare.png "${OUT_DIR}"/palm_goal_slide.png || true
exit $PY_EXIT
