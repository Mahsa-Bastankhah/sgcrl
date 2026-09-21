#!/bin/bash
#SBATCH --job-name=allegro_mmx_oracle
#SBATCH --output=slurm/allegro_four2mmx_trim_randinit_oracle_sanity_%j.log
#SBATCH --error=slurm/allegro_four2mmx_trim_randinit_oracle_sanity_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=seas

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

echo "Oracle sanity: four2mmx, trim_sa, randomized init, E=1024"
echo "Action at every step is the exact normalized physical joint goal"
nvidia-smi || true

python -u scripts/allegro_kuka_control_sanity_viz.py \
  --mode=four2mmx \
  --pipeline=gpu \
  --trim-sa \
  --randomize-init \
  --oracle-action \
  --num-envs=1024 \
  --steps=150
