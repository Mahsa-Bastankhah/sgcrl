#!/bin/bash
#SBATCH --job-name=its_fsqg_validate
#SBATCH --output=slurm/allegro_its_fullyscaled_qdgoal_validate_%j.log
#SBATCH --error=slurm/allegro_its_fullyscaled_qdgoal_validate_%j.log
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
mkdir -p slurm
nvidia-smi || true
python -u scripts/validate_allegro_qd_goal.py \
  --num-envs=128 \
  --trials=4 \
  --episode-length=150
