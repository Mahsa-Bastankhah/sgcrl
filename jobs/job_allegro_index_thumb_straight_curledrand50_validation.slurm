#!/bin/bash
#SBATCH --job-name=its_curledrand50_val
#SBATCH --output=slurm/allegro_index_thumb_straight_curledrand50_validation_%j.log
#SBATCH --error=slurm/allegro_index_thumb_straight_curledrand50_validation_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

echo "validate state/action/goal=16/8/8, reset=50%, initial success=0, oracle>=99%"
nvidia-smi || true
python -u scripts/allegro_kuka_control_sanity_viz.py \
  --mode=index_thumb_straight \
  --pipeline=gpu \
  --trim-sa \
  --trim-init-range-frac=0.50 \
  --randomize-init \
  --oracle-action \
  --num-envs=1024 \
  --steps=150
