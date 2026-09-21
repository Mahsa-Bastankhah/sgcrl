#!/bin/bash
#SBATCH --job-name=its_rank50
#SBATCH --output=slurm/allegro_index_thumb_straight_curledrand50_nf_rank_accuracy_%A_%a.log
#SBATCH --error=slurm/allegro_index_thumb_straight_curledrand50_nf_rank_accuracy_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --account=seas
#SBATCH --array=0-2

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

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    VARIANT="compactsmall_mix50"
    RUN="ppo_allegro_index_thumb_straight_curledrand50_normgoal_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h"
    ;;
  1)
    VARIANT="tiny_mix50"
    RUN="ppo_allegro_index_thumb_straight_curledrand50_normgoal_e1024_nf_tiny_sa2x128_r32_b4_w128_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h"
    ;;
  2)
    VARIANT="compactsmall_nomix"
    RUN="ppo_allegro_index_thumb_straight_curledrand50_normgoal_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_ent01fixed_nomix_extrew1_seed0_4h"
    ;;
  *)
    echo "invalid array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

CKPT_DIR="logs/${RUN}/ppo_allegro_kuka_throw_0/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/index_thumb_straight_curledrand50_rank_accuracy"
OUT_STEM="${OUT_DIR}/${VARIANT}_ckpts100_500_1000_1300"
mkdir -p slurm "${OUT_DIR}"
echo "variant=${VARIANT} checkpoint_dir=${CKPT_DIR}"
echo "iterations=100,500,1000,1300 seed=0 min_samples=256"
nvidia-smi || true

python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --iterations=100,500,1000,1300 \
  --future-discount=0.99 \
  --output-csv="${OUT_STEM}.csv" \
  --output-plot="${OUT_STEM}.png" \
  --seed=0 \
  --min-samples=256 \
  --pipeline=gpu
