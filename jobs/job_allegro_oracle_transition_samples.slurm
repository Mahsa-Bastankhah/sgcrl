#!/bin/bash
#SBATCH --job-name=allegro_oracle_sas
#SBATCH --output=slurm/allegro_oracle_transition_samples_%j.log
#SBATCH --error=slurm/allegro_oracle_transition_samples_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# Dump index_thumb_straight trim_sa oracle (s,a,s') samples + 1-step
# oracle vs random Δ. No training.
set -euo pipefail

unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

OUT_DIR="figs/allegro_kuka_throw/oracle_transition_samples"
mkdir -p slurm "${OUT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "mode=index_thumb_straight trim_sa full_range num_envs=8"
nvidia-smi || true

python -u scripts/allegro_oracle_transition_samples.py \
  --num-envs=8 \
  --seed=0 \
  --trim-init-mode=full_range \
  --trim-init-range-frac=0.50 \
  --pipeline=gpu \
  --horizon=20 \
  --out-dir="${OUT_DIR}"

ls -lh "${OUT_DIR}"/its_full_range_oracle_vs_random_1step.csv \
  "${OUT_DIR}"/its_full_range_oracle_transitions.txt
