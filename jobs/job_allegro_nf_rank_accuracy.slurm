#!/bin/bash
#SBATCH --job-name=akt_nf_rank_acc
#SBATCH --output=slurm/allegro_nf_rank_accuracy_%j.log
#SBATCH --error=slurm/allegro_nf_rank_accuracy_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Two-rollout NF positive-vs-negative ranking accuracy for every checkpoint.
# Override CKPT_DIR / OUT_STEM to evaluate another Allegro NF run.
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

CKPT_DIR="${CKPT_DIR:-logs/ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_noshape_mixtaskg_mix50_extrew1_4h/ppo_allegro_kuka_throw_0/checkpoints}"
OUT_STEM="${OUT_STEM:-figs/allegro_kuka_throw/akt_tablespawn_grasp_nf_rank_accuracy}"

mkdir -p slurm "$(dirname "${OUT_STEM}")"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "checkpoint_dir=${CKPT_DIR}"
echo "output_stem=${OUT_STEM}"
nvidia-smi || true

python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --output-csv="${OUT_STEM}.csv" \
  --output-plot="${OUT_STEM}.png" \
  --seed=0 \
  --min-samples=256 \
  --pipeline=gpu
