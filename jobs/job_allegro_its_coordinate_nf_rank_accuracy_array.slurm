#!/bin/bash
#SBATCH --job-name=its_coord_rank
#SBATCH --output=slurm/allegro_its_coordinate_nf_rank_%A_%a.log
#SBATCH --error=slurm/allegro_its_coordinate_nf_rank_%A_%a.log
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
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

MODES=(physical fully_scaled mixed)
ITERS=(100 500 1000 1300)
MODE="${MODES[$((SLURM_ARRAY_TASK_ID / 4))]}"
ITER="${ITERS[$((SLURM_ARRAY_TASK_ID % 4))]}"
case "${MODE}" in
  physical)
    RUN="logs/ppo_allegro_its_fullrange_physical_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0"
    ;;
  fully_scaled)
    RUN="logs/ppo_allegro_its_fullrange_fully_scaled_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0"
    ;;
  mixed)
    RUN="logs/ppo_allegro_index_thumb_straight_fullrange_normgoal_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0"
    ;;
esac
CKPT_DIR="${RUN}/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/coordinate_mode_nf_rank_accuracy/${MODE}/parts"
OUT_STEM="${OUT_DIR}/iter_${ITER}"
mkdir -p slurm "${OUT_DIR}"
echo "mode=${MODE} iteration=${ITER} protocol=gamma0.99 two-stochastic-trajectories other-trajectory-negatives seed0"
test -f "${CKPT_DIR}/ckpt_iter_$(printf '%07d' "${ITER}").pkl"
nvidia-smi || true
python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --iterations="${ITER}" \
  --future-discount=0.99 \
  --output-csv="${OUT_STEM}.csv" \
  --output-plot="${OUT_STEM}.png" \
  --seed=0 \
  --min-samples=256 \
  --pipeline=gpu
