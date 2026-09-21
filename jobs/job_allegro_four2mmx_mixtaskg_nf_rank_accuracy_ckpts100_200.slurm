#!/bin/bash
#SBATCH --job-name=mmx_nf_rank_100_200
#SBATCH --output=slurm/allegro_four2mmx_mixtaskg_nf_rank_accuracy_ckpts100_200_%j.log
#SBATCH --error=slurm/allegro_four2mmx_mixtaskg_nf_rank_accuracy_ckpts100_200_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
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

CKPT_DIR="logs/ppo_allegro_four2mmx_trim_sa_randinit_normgoal_e1024_nf_compact_small_200m_ep150_minstd1e4_ent01to001_mixtaskg_4h_jfrac/ppo_allegro_kuka_throw_0/checkpoints"
OUT_STEM="figs/allegro_kuka_throw/four2mmx_mixtaskg_nf_rank_accuracy_ckpts100_200"

mkdir -p slurm "$(dirname "${OUT_STEM}")"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "checkpoint_dir=${CKPT_DIR}"
echo "iterations=100,200"
echo "positive=same-trajectory truncated-geometric future q"
echo "negative=random q from other trajectory"
echo "expected trim_sa achieved-goal indices=state[0:8] (normalized q, not qd)"
nvidia-smi || true

python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --iterations=100,200 \
  --future-discount=0.99 \
  --output-csv="${OUT_STEM}.csv" \
  --output-plot="${OUT_STEM}.png" \
  --seed=0 \
  --min-samples=256 \
  --pipeline=gpu

ls -lh "${OUT_STEM}.csv" "${OUT_STEM}.png" \
  "${OUT_STEM}_meta.json"
