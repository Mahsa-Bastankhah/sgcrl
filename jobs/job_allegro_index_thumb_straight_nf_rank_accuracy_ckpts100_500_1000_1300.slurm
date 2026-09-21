#!/bin/bash
#SBATCH --job-name=its_nf_rank_100_1300
#SBATCH --output=slurm/allegro_index_thumb_straight_nf_rank_accuracy_ckpts100_500_1000_1300_%j.log
#SBATCH --error=slurm/allegro_index_thumb_straight_nf_rank_accuracy_ckpts100_500_1000_1300_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --account=seas

# NF positive-vs-negative rank accuracy for index_thumb_straight trim_sa run.
# Reuses scripts/eval_allegro_nf_rank_accuracy.py (trim_sa → goal = state[0:8]).
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

CKPT_DIR="logs/ppo_allegro_index_thumb_straight_trim_sa_curledrand_normgoal_balmean015_max030_e1024_nf_compact_small_200m_ep150_minstd1e4_ent05fixed_mixtaskg_4h_jfrac/ppo_allegro_kuka_throw_0/checkpoints"
OUT_STEM="figs/allegro_kuka_throw/index_thumb_straight_nf_rank_accuracy_ckpts100_500_1000_1300"

mkdir -p slurm "$(dirname "${OUT_STEM}")"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "checkpoint_dir=${CKPT_DIR}"
echo "iterations=100,500,1000,1300"
echo "mode=index_thumb_straight trim_sa=true curledrand=true normgoal=true"
echo "expected dims: state=16 action=8 goal=8; achieved-goal indices=state[0:8]"
echo "positive=same-trajectory truncated-geometric future q"
echo "negative=random q from other trajectory"
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

ls -lh "${OUT_STEM}.csv" "${OUT_STEM}.png" \
  "${OUT_STEM}_meta.json"
