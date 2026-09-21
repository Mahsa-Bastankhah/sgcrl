#!/bin/bash
#SBATCH --job-name=akt_bucket_mid_viz
#SBATCH --output=slurm/allegro_kuka_throw_bucket_mid_compare_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_bucket_mid_compare_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# Stills only: NVIDIA / mid-y / true-mid / hard buckets. Does NOT train.
#   sbatch jobs/job_allegro_kuka_throw_bucket_mid_compare_viz.slurm
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

OUT_DIR="figs/allegro_kuka_throw/bucket_mid_compare"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "keep-arm pad-hold  NVIDIA table"
echo "buckets: (0.50,-0.30) (0.50,-0.45) (0.48,-0.45) (0.45,-0.60)"
echo "NOT training"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_bucket_location_compare_viz.py \
  --out-dir="${OUT_DIR}" --pipeline=gpu \
  --location "nvidia_050030:0.50,-0.30,0.40:NVIDIA  (0.50, -0.30, 0.40)" \
  --location "mid_nvx_050045:0.50,-0.45,0.40:mid NVIDIA-x  (0.50, -0.45, 0.40)" \
  --location "mid_true_048045:0.48,-0.45,0.40:true mid  (0.48, -0.45, 0.40)" \
  --location "hard_045060:0.45,-0.60,0.40:hard  (0.45, -0.60, 0.40)"
PY_EXIT=$?
ls -lh "${OUT_DIR}"/*.png || true
exit $PY_EXIT
