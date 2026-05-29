#!/bin/bash
# Plot all checkpoints for PPO point-maze runs (old style).
#
# Defaults: LOG_ROOT=ppo_fourrooms, ENV=point_Impossible, SEEDS=400 401 402
#
# Examples:
#   LOG_ROOT=ppo_impossible ENV=point_Impossible SEEDS="123 124 125" \
#     OUT_BASE=plots/ppo_impossible_all_ckpts_sub25_traj5 \
#     bash scripts/plot_impossible_seeds_400_402.sh
#
#   LOG_ROOT=ppo_spiral11x11 ENV=point_Spiral11x11 SEEDS="123 124 125 126 127 128" \
#     OUT_BASE=plots/ppo_spiral11x11_sub25_traj5 \
#     bash scripts/plot_impossible_seeds_400_402.sh
set -euo pipefail

cd /n/fs/mislresearch/sgcrl
source .venv/bin/activate
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

LOG_ROOT="${LOG_ROOT:-ppo_fourrooms}"
ENV="${ENV:-point_Impossible}"
OUT_BASE="${OUT_BASE:-plots/ppo_fourrooms_impossible_sub25_traj5}"
HEATMAP_SUBCELLS=25
NUM_TRAJ=5
FIG_SCALE=2.0
read -ra SEEDS <<< "${SEEDS:-400 401 402}"

plot_run() {
  local seed="$1"
  local run_name="ppo_${ENV}_${seed}"
  local ckpt_dir="logs/${LOG_ROOT}/${run_name}/checkpoints"
  local out_dir="${OUT_BASE}/${run_name}/"

  if [ ! -d "${ckpt_dir}" ]; then
    echo "[plot] skip: missing ${ckpt_dir}"
    return 0
  fi
  shopt -s nullglob
  local pkls=( "${ckpt_dir}"/*.pkl )
  shopt -u nullglob
  if [ "${#pkls[@]}" -eq 0 ]; then
    echo "[plot] skip: no .pkl in ${ckpt_dir}"
    return 0
  fi
  echo "[plot] found ${#pkls[@]} checkpoint(s) in ${ckpt_dir}"

  mkdir -p "${out_dir}"
  echo "[plot] env=${ENV} checkpoint=${ckpt_dir}"
  echo "[plot] output=${out_dir}"
  python -u ppo_rollout_maze.py \
      --checkpoint="${ckpt_dir}" \
      --env="${ENV}" \
      --output="${out_dir}" \
      --seed="${seed}" \
      --num_trajectories="${NUM_TRAJ}" \
      --heatmap_subcells="${HEATMAP_SUBCELLS}" \
      --fig_scale="${FIG_SCALE}"
}

echo "[plot] LOG_ROOT=${LOG_ROOT}  ENV=${ENV}  OUT_BASE=${OUT_BASE}  seeds=${SEEDS[*]}"
for seed in "${SEEDS[@]}"; do
  plot_run "${seed}"
done

echo "[plot] done."
