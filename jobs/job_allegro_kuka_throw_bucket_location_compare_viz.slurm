#!/bin/bash
#SBATCH --job-name=akt_bucket_loc_viz
#SBATCH --output=slurm/allegro_kuka_throw_bucket_location_compare_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_bucket_location_compare_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# Stills only: current NVIDIA bucket vs two proposed harder xyz. Does NOT train.
#   sbatch jobs/job_allegro_kuka_throw_bucket_location_compare_viz.slurm
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

OUT_DIR="figs/allegro_kuka_throw/bucket_location_compare"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "keep-arm pad-hold  NVIDIA table  buckets: (0.50,-0.30) (0.45,-0.60) (0.00,-0.80)"
echo "NOT training"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_bucket_location_compare_viz.py \
  --out-dir="${OUT_DIR}" --pipeline=gpu
PY_EXIT=$?
ls -lh "${OUT_DIR}"/*.png || true
exit $PY_EXIT
