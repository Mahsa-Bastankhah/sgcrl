#!/bin/bash
#SBATCH --job-name=its_fullrange_rank
#SBATCH --output=slurm/allegro_index_thumb_straight_fullrange_nf_rank_accuracy_%j.log
#SBATCH --error=slurm/allegro_index_thumb_straight_fullrange_nf_rank_accuracy_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --account=seas

# NF positive-vs-negative binary/rank accuracy for full_range init ITS run
# (train job 3832416). Same protocol as curledrand50 compactsmall_mix50 probes:
# seed=0, geometric positives, other-trajectory negatives, chance=50%.
# Early training: probe existing immutable milestones 100/200/300/400/500
# (preferred 1000/1300 not yet written).
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

CKPT_DIR="logs/ppo_allegro_index_thumb_straight_fullrange_normgoal_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/index_thumb_straight_fullrange_rank_accuracy"
OUT_STEM="${OUT_DIR}/compactsmall_mix50_ckpts100_200_300_400_500"
RUN_CFG="logs/ppo_allegro_index_thumb_straight_fullrange_normgoal_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0/run_config.json"

mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "checkpoint_dir=${CKPT_DIR}"
echo "iterations=100,200,300,400,500 seed=0 min_samples=256"
echo "positive=same-trajectory truncated-geometric future q"
echo "negative=random q from other trajectory"
echo "chance=50%"
python - <<'PY'
import json
cfg = json.load(open("logs/ppo_allegro_index_thumb_straight_fullrange_normgoal_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0/run_config.json"))
flags = cfg.get("flags") or cfg.get("resolved_config") or {}
mode = flags.get("isaacgym_control_sanity_trim_init_mode")
print(f"run_config trim_init_mode={mode!r} (expect full_range)")
if str(mode).strip().lower() != "full_range":
  raise SystemExit(f"unexpected trim_init_mode={mode!r}")
PY
ls -1 "${CKPT_DIR}"/ckpt_iter_0000{100,200,300,400,500}.pkl
nvidia-smi || true

python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --iterations=100,200,300,400,500 \
  --future-discount=0.99 \
  --output-csv="${OUT_STEM}.csv" \
  --output-plot="${OUT_STEM}.png" \
  --seed=0 \
  --min-samples=256 \
  --pipeline=gpu

ls -lh "${OUT_STEM}.csv" "${OUT_STEM}.png" \
  "${OUT_STEM}_meta.json"
