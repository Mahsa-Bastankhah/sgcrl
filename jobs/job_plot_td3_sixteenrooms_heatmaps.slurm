#!/bin/bash
#SBATCH --job-name=td3_maze_hm
#SBATCH --output=slurm/td3_sixteenrooms_heatmaps_%A_%a.log
#SBATCH --error=slurm/td3_sixteenrooms_heatmaps_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:45:00
#SBATCH --account=seas
#SBATCH --array=0-2

# Plot Q1(s,a,g) + V(s,g) maze heatmaps for ppo_td3_sixteenrooms_tau05 seeds 0-2.
# CPU-only (no GPU). Uses scripts/ppo_rollout_maze.py TD3 overlay mode.
#
#   sbatch jobs/job_plot_td3_sixteenrooms_heatmaps.slurm

module purge

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

cd /n/fs/mislresearch/sgcrl
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

ENV=${ENV:-point_SixteenRooms}
LOG_SUBDIR=${LOG_SUBDIR:-ppo_td3_sixteenrooms_tau05}
PLOT_DIR=${PLOT_DIR:-plots/ppo_td3_sixteenrooms_tau05_heatmaps}
SEED=${SLURM_ARRAY_TASK_ID:-0}
RUN_DIR="logs/${LOG_SUBDIR}/ppo_${ENV}_${SEED}"
CKPT=${CHECKPOINT:-"${RUN_DIR}/checkpoints"}
OUT="${PLOT_DIR}/${ENV}_${SEED}/"

mkdir -p "${OUT}" slurm

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  seed=${SEED}"
echo "checkpoint=${CKPT}"
echo "output=${OUT}"

if [ ! -e "${CKPT}" ]; then
  echo "ERROR: checkpoint path not found: ${CKPT}"
  ls -la "${RUN_DIR}/checkpoints/" 2>/dev/null || echo "  (no such dir)"
  exit 2
fi

python -u scripts/ppo_rollout_maze.py \
    --checkpoint="${CKPT}" \
    --env="${ENV}" \
    --output="${OUT}" \
    --seed="${SEED}" \
    --num_trajectories=5 \
    --heatmap_subcells=25 \
    --fig_scale=2.0

echo "Done. Output:"
ls -lh "${OUT}"
