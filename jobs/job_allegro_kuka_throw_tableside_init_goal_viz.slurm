#!/bin/bash
#SBATCH --job-name=akt_tside_init_goal_viz
#SBATCH --output=slurm/allegro_kuka_throw_tableside_init_goal_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_tableside_init_goal_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# Short stills only: tableside table_spawn grasp init vs TABLE_SIDE_GOAL.
# T=50. v1 object-only + v2 palm+object. Does NOT train NF.
#   sbatch jobs/job_allegro_kuka_throw_tableside_init_goal_viz.slurm
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

OUT_DIR="${OUT_DIR:-figs/allegro_kuka_throw/tableside_init_goal}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  writing ${OUT_DIR}"
echo "T=50  table_push + table_spawn grasp  goal=(0.20,-0.15,0.555)"
echo "v1 object-only state49/goal3   v2 palm+object state52/goal6"
echo "NOT launching NF training"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_tableside_init_goal_viz.py \
  --variant=object --mode=grasp --out-dir="${OUT_DIR}" --pipeline=gpu --video
PY1=$?
python -u scripts/allegro_kuka_throw_tableside_init_goal_viz.py \
  --variant=palm --mode=grasp --out-dir="${OUT_DIR}" --pipeline=gpu --video
PY2=$?
ls -lh "${OUT_DIR}"/tableside_init_goal_grasp_*.png \
       "${OUT_DIR}"/tableside_init_goal_grasp_*.mp4 || true
if [ "${PY1}" -ne 0 ] || [ "${PY2}" -ne 0 ]; then
  echo "viz failed object=${PY1} palm=${PY2}"
  exit 1
fi
echo "viz-only complete; NF trains were NOT submitted"
