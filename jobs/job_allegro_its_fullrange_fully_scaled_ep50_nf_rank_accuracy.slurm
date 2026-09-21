#!/bin/bash
#SBATCH --job-name=its_ep50_nf_rank
#SBATCH --output=slurm/allegro_its_fullrange_fully_scaled_ep50_nf_rank_accuracy_%j.log
#SBATCH --error=slurm/allegro_its_fullrange_fully_scaled_ep50_nf_rank_accuracy_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=seas

# One-shot NF binary rank accuracy for the T=50 fully_scaled ITS run
# (train job 3857446). Same protocol as eval_allegro_nf_rank_accuracy.py:
# two stochastic rollouts, truncated-geometric same-traj future g+,
# other-traj g-, chance=50%. Horizon is training T=50 (~98 pairs).
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

RUN="logs/ppo_allegro_its_fullrange_fully_scaled_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep50_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0"
CKPT_DIR="${RUN}/checkpoints"
OUT_DIR="figs/allegro_kuka_throw"
OUT_STEM="${OUT_DIR}/its_fullrange_fully_scaled_nf_ep50_nf_rank_accuracy"
mkdir -p slurm "${OUT_DIR}"

ITERS=$(python - <<'PY'
import glob, os, re
d = "logs/ppo_allegro_its_fullrange_fully_scaled_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep50_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0/checkpoints"
iters = []
for p in glob.glob(os.path.join(d, "ckpt_iter_*.pkl")):
    m = re.search(r"ckpt_iter_(\d+)\.pkl$", os.path.basename(p))
    if m:
        it = int(m.group(1))
        if it % 100 == 0:
            iters.append(it)
print(",".join(str(i) for i in sorted(set(iters))))
PY
)
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "checkpoint_dir=${CKPT_DIR}"
echo "iterations=${ITERS} seed=0 min_samples=64 num_steps=training_T=50"
echo "positive=same-trajectory truncated-geometric future q_norm"
echo "negative=random q_norm from other trajectory"
echo "chance=50%"
python - <<'PY'
import json
cfg = json.load(open(
    "logs/ppo_allegro_its_fullrange_fully_scaled_e1024_nf_compactsmall_sa3x192_r64_b6_w192_200m_ep50_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0/run_config.json"))
flags = cfg.get("flags") or cfg.get("resolved_config") or {}
mode = flags.get("isaacgym_coordinate_mode")
init = flags.get("isaacgym_control_sanity_trim_init_mode")
ep = flags.get("ppo_rollout_length")
print(f"run_config coordinate_mode={mode!r} init={init!r} T={ep}")
if str(mode).strip() != "fully_scaled":
    raise SystemExit(f"unexpected coordinate_mode={mode!r}")
if str(init).strip().lower() != "full_range":
    raise SystemExit(f"unexpected trim_init_mode={init!r}")
PY
nvidia-smi || true

python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --iterations="${ITERS}" \
  --future-discount=0.99 \
  --output-csv="${OUT_STEM}.csv" \
  --output-plot="${OUT_STEM}.png" \
  --seed=0 \
  --min-samples=64 \
  --pipeline=gpu

ls -lh "${OUT_STEM}.csv" "${OUT_STEM}.png" \
  "${OUT_STEM}_meta.json"
