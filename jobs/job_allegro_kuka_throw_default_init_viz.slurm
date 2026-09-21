#!/bin/bash
#SBATCH --job-name=akt_throw_default_init_viz
#SBATCH --output=slurm/allegro_kuka_throw_default_init_viz_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_default_init_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --account=seas

# PhysX stills of NVIDIA throw pose v1 init + cube-in-bucket goal. Does NOT train.
#   sbatch jobs/job_allegro_kuka_throw_default_init_viz.slurm
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

OUT_DIR="${OUT_DIR:-figs/allegro_kuka_throw/throw_default_init_goal}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  writing ${OUT_DIR}"
echo "NVIDIA throw default  init=cube table center  goal=cube in bucket  NOT training"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_default_init_viz.py \
  --out-dir="${OUT_DIR}" --pipeline=gpu
PY=$?
ls -lh "${OUT_DIR}"/throw_default_*.png || true
exit $PY
