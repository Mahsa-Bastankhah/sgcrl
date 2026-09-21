#!/bin/bash
#SBATCH --job-name=allegro_coord_val
#SBATCH --output=slurm/allegro_coordinate_modes_validation_%j.log
#SBATCH --error=slurm/allegro_coordinate_modes_validation_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

OUT_DIR="figs/allegro_kuka_throw/coordinate_mode_validation"
mkdir -p slurm "${OUT_DIR}"
echo "modes=mixed,physical,fully_scaled E=128 ep=150 seed=0"
echo "expected dims=state16/action8/goal8; oracle>=99%"
nvidia-smi || true
python -u scripts/validate_allegro_coordinate_modes.py \
  --output-dir="${OUT_DIR}" \
  --num-envs=128 \
  --steps=150 \
  --seed=0 \
  --pipeline=gpu
