#!/usr/bin/env bash
# run_all_pca.sh — run PCA once per seed with full list of checkpoints

# ------------------------------------------------------------------
# 1) Activate the conda env
# ------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_rl

# ------------------------------------------------------------------
# 2) Update LD_LIBRARY_PATH
# ------------------------------------------------------------------
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# ------------------------------------------------------------------
# 3) Configuration
# ------------------------------------------------------------------
ENV="point_Wall11x11"
LOG_DIR="logs"
ACTION_MODE="actor_max"
GRID_WIDTH="0.01"
NUM_EPISODES=1
export CUDA_VISIBLE_DEVICES=""  # Force CPU

# List of seeds
SEEDS=(101)

# List of checkpoints
CKPTS=(2 5 10 20 30 40 50 60)
CKPT_ARGS="${CKPTS[@]}"  # Expand into --ckpts list

# Create log folder
mkdir -p nohup_logs

set -euo pipefail

# ------------------------------------------------------------------
# 4) Launch one job per seed (sequentially, wait for each to finish)
# ------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
  LOGFILE="nohup_logs/pca_env${ENV}_seed${SEED}.out"
  echo "▶ Running seed=${SEED}, logs → ${LOGFILE}"

  # Run synchronously, log output immediately
  python -u -m experiments.plot_PCA_repr \
    --env "$ENV" \
    --log_dir "$LOG_DIR" \
    --ckpts $CKPT_ARGS \
    --seed "$SEED" \
    --action_mode "$ACTION_MODE" \
    --grid_width "$GRID_WIDTH" \
    --num_episodes "$NUM_EPISODES" \
    > "$LOGFILE" 2>&1

  echo "✓ Finished seed=$SEED"
done

echo 