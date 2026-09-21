#!/bin/bash
#SBATCH --job-name=akt_bucket_feas
#SBATCH --output=slurm/allegro_kuka_throw_bucket_feasibility_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_bucket_feasibility_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# IK hover + scripted swing/release for three bucket xyz. Does NOT train.
#   sbatch jobs/job_allegro_kuka_throw_bucket_feasibility.slurm
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

OUT_DIR="figs/allegro_kuka_throw/bucket_feasibility"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "feasibility: IK hover + drop + scripted throw  (control + two proposed)"
echo "NOT training"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_bucket_feasibility.py \
  --out-dir="${OUT_DIR}" --pipeline=gpu
PY_EXIT=$?
echo '--- summary ---'
cat "${OUT_DIR}/feasibility.json" 2>/dev/null || true
ls -lh "${OUT_DIR}" || true
exit $PY_EXIT
