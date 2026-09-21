#!/bin/bash
#SBATCH --job-name=allegro_policy_oracle_a
#SBATCH --output=slurm/allegro_four2mmx_policy_vs_oracle_actions_%j.log
#SBATCH --error=slurm/allegro_four2mmx_policy_vs_oracle_actions_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export MPLBACKEND=Agg

CKPT="${1:?pass checkpoint path as first argument}"
OUT_DIR="figs/allegro_kuka_throw/policy_vs_oracle_actions"
mkdir -p slurm "${OUT_DIR}" "${MPLCONFIGDIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "checkpoint=${CKPT}"
echo "comparison=deterministic policy vs normalized absolute-target oracle"
echo "joints=1(index proximal),7(thumb proximal)"

python -u scripts/plot_allegro_policy_vs_oracle_actions.py \
  --checkpoint="${CKPT}" \
  --out-dir="${OUT_DIR}" \
  --seed=0 \
  --joints=1,7 \
  --pipeline=gpu
