#!/bin/bash
#SBATCH --job-name=ppo_maze
#SBATCH --output=slurm/maze_%A_%a.out
#SBATCH --error=slurm/maze_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:20:00
#SBATCH --account=seas
#SBATCH --array=0-1          # matches the training seeds; edit range as needed

# ---------------------------------------------------------------------------
# What this does
#   For each SLURM_ARRAY_TASK_ID (= array index), loads *every* checkpoint
#   under logs/$LOG_SUBDIR/ppo_${ENV}_${SEED}/checkpoints (ckpt_iter_*.pkl
#   + latest.pkl) and writes one .png trajectory plot per checkpoint into
#   plots/${ENV}_${SEED}/.  Uses deterministic actions by default; set
#   STOCHASTIC=1 to sample instead.
#
#   Unlike the sawyer/mujoco video job, this one has NO GPU / MuJoCo
#   dependency — maze envs are pure numpy, rendering is pure matplotlib,
#   so the script runs on any CPU node.
#
# To render a single specific checkpoint instead, set $CHECKPOINT to a
# .pkl file:
#   sbatch --export=ALL,CHECKPOINT=logs/ppo/ppo_point_FourRooms_0/checkpoints/ckpt_iter_5000.pkl \
#          job_rollout_maze.slurm
# ---------------------------------------------------------------------------

module purge

# Drop any broken inherited CUDA paths from the submit host so a stray
# /usr/local/cuda-xx path doesn't sabotage dlopen() inside jaxlib.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
cd /n/fs/mislresearch/sgcrl
source .venv/bin/activate

export PYTHONUNBUFFERED=1
# No GPU requested — force CPU so JAX doesn't probe for CUDA.
export JAX_PLATFORMS=cpu
# Headless matplotlib.
export MPLBACKEND=Agg

# ---------------------------------------------------------------------------
# Config — edit to taste
# ---------------------------------------------------------------------------
ENV=${ENV:-point_Impossible}
LOG_SUBDIR=${LOG_SUBDIR:-ppo_fourrooms}       # logs live under logs/$LOG_SUBDIR/
PLOT_DIR=${PLOT_DIR:-plots}
STOCHASTIC=${STOCHASTIC:-0}                   # 1 = sample from policy
MAX_STEPS=${MAX_STEPS:--1}                    # -1 = env default

# Offset the array-id so `--array=0-2` maps to seeds 3, 4, 5 — matching the
# retrained PPO runs.  Override via
# `sbatch --export=ALL,SEED_OFFSET=0 job_rollout_maze.slurm` to restore the
# 0-indexed mapping.
SEED_OFFSET=${SEED_OFFSET:-400}
SEED=$(( ${SLURM_ARRAY_TASK_ID:-0} + SEED_OFFSET ))
RUN_DIR="logs/${LOG_SUBDIR}/ppo_${ENV}_${SEED}"

# Checkpoint selection (mirrors job_rollout_video_conda.slurm):
#   - If $CHECKPOINT is set explicitly (a single .pkl), render just that one.
#   - Otherwise point at the whole checkpoints/ directory so the python
#     script iterates every ckpt_iter_*.pkl + latest.pkl.
CKPT=${CHECKPOINT:-"${RUN_DIR}/checkpoints"}

# Output target:
#   - Single checkpoint  -> plots/<env>_<seed>.png
#   - Directory of ckpts -> plots/<env>_<seed>/ (one png per checkpoint)
if [ -f "${CKPT}" ]; then
  OUT="${PLOT_DIR}/${ENV}_${SEED}.png"
  mkdir -p "${PLOT_DIR}"
else
  OUT="${PLOT_DIR}/${ENV}_${SEED}/"
  mkdir -p "${OUT}"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  seed=${SEED}"
echo "checkpoint=${CKPT}"
echo "output=${OUT}"

if [ ! -e "${CKPT}" ]; then
  echo "ERROR: checkpoint path not found: ${CKPT}"
  echo "Run dir contents (if any):"
  ls -la "${RUN_DIR}/checkpoints/" 2>/dev/null || echo "  (no such dir)"
  exit 2
fi
if [ -d "${CKPT}" ] && ! ls "${CKPT}"/ckpt_iter_*.pkl "${CKPT}"/latest.pkl \
      >/dev/null 2>&1; then
  echo "ERROR: no ckpt_iter_*.pkl or latest.pkl inside ${CKPT}"
  ls -la "${CKPT}" || true
  exit 2
fi

# ---------------------------------------------------------------------------
# Flag assembly — optional --stochastic, optional --max_steps override.
# ---------------------------------------------------------------------------
EXTRA_FLAGS=()
if [ "${STOCHASTIC}" = "1" ]; then
  EXTRA_FLAGS+=(--stochastic)
fi
if [ "${MAX_STEPS}" -ge 0 ] 2>/dev/null; then
  EXTRA_FLAGS+=(--max_steps "${MAX_STEPS}")
fi

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
python -u scripts/ppo_rollout_maze.py \
    --checkpoint="${CKPT}" \
    --env="${ENV}" \
    --output="${OUT}" \
    --seed="${SEED}" \
    --num_trajectories=5 \
    --heatmap_subcells=25 \
    --fig_scale=2.0 \
    "${EXTRA_FLAGS[@]}"

echo "Done. Output:"
if [ -d "${OUT}" ]; then
  ls -lh "${OUT}"
else
  ls -lh "${OUT}"
fi
