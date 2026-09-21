#!/bin/bash
#SBATCH --job-name=its_fsqg_rank
#SBATCH --output=slurm/allegro_its_fullyscaled_qdgoal_rank_%j.log
#SBATCH --error=slurm/allegro_its_fullyscaled_qdgoal_rank_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --account=seas

set -euo pipefail
: "${ITER:?submit with --export=ALL,ITER=<checkpoint>}"
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

RUN_ROOT="logs/ppo_allegro_its_fullyscaled_qdgoal_fullrange_ent01fixed_compactsmall_mix50_e1024_nf_sa3x192_r64_b6_w192_200m_ep150_minstd1e4_extrew1_seed0_4h/ppo_allegro_kuka_throw_0"
CKPT_DIR="${RUN_ROOT}/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/index_thumb_straight_fullyscaled_qdgoal_fullrange_rank_accuracy"
OUT_STEM="${OUT_DIR}/compactsmall_mix50_ckpt${ITER}_seed0"
mkdir -p slurm "${OUT_DIR}"

echo "checkpoint=${ITER} seed=0 trajectories=2 future_discount=0.99 min_samples=256"
echo "positive=same-trajectory geometric future [q_norm,qd_scaled]"
echo "negative=random [q_norm,qd_scaled] from other trajectory"
python - "${RUN_ROOT}/run_config.json" <<'PY'
import json, sys
flags = json.load(open(sys.argv[1]))["flags"]
assert flags["isaacgym_control_sanity_goal_include_qd"] is True
assert flags["isaacgym_coordinate_mode"] == "fully_scaled"
assert flags["isaacgym_control_sanity_trim_init_mode"] == "full_range"
print("run_config fully-scaled qd-goal/full_range verified")
PY

python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --iterations="${ITER}" \
  --future-discount=0.99 \
  --output-csv="${OUT_STEM}.csv" \
  --output-plot="${OUT_STEM}.png" \
  --seed=0 \
  --min-samples=256 \
  --pipeline=gpu
