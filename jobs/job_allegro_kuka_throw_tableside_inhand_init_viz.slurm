#!/bin/bash
#SBATCH --job-name=akt_tside_inhand_init_viz
#SBATCH --output=slurm/allegro_kuka_throw_tableside_inhand_init_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_tableside_inhand_init_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# PhysX stills only: cube starts in a curled palm, 4 random inits.
# Does NOT train NF.
#   sbatch jobs/job_allegro_kuka_throw_tableside_inhand_init_viz.slurm
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

OUT_DIR="${OUT_DIR:-figs/allegro_kuka_throw/tableside_inhand_init}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  writing ${OUT_DIR}"
echo "in-hand tableside init  curl=1.0  above=0.07  hand xy ±3cm  obj ±1.2cm"
echo "goal=(0.20,-0.15,0.555)  large_table=1.5x1.5  NOT launching NF"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_tableside_inhand_init_viz.py \
  --out-dir="${OUT_DIR}" --pipeline=gpu --n=4 --seed=0
PY=$?
ls -lh "${OUT_DIR}"/inhand_init_*_side.png \
       "${OUT_DIR}"/inhand_init_*_three.png \
       "${OUT_DIR}"/inhand_init_grid_side.png || true
if [ "${PY}" -ne 0 ]; then
  echo "viz failed py=${PY}"
  exit 1
fi
echo "viz-only complete; NF trains were NOT submitted"
