#!/bin/bash

# Configuration
LOG_ROOT="/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"
export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench

function render_from_cmd() {
    local CMD="$1"
    
    # 1. Extract parameters from the provided command string
    # Assumes CMD format: python ppo_contrastive.py --env=... --log_dir_path=...
    local ENV_ID=$(echo "$CMD" | grep -oP '--env=\K\S+')
    local LOG_PATH=$(echo "$CMD" | grep -oP '--log_dir_path=\K\S+')
    
    # In your training script, the actual path ends with ppo_${ENV_ID}_${seed}
    # We find the run directory by looking for the matching folder inside the log path
    local RUN_DIR=$(find "$LOG_PATH" -maxdepth 1 -name "ppo_${ENV_ID}_*" | head -n 1)
    
    if [ -z "$RUN_DIR" ]; then
        echo "Error: Could not find run directory for env $ENV_ID in $LOG_PATH"
        return 1
    fi

    echo "Found run directory: $RUN_DIR"
    
    # 2. Run the video generator
    # We look for the latest checkpoint
    local LATEST_CKPT="$RUN_DIR/checkpoints/latest.pkl"
    local OUTPUT_DIR="$RUN_DIR/videos"
    
    if [ ! -f "$LATEST_CKPT" ]; then
        echo "Error: latest.pkl not found in $RUN_DIR/checkpoints/"
        return 1
    fi
    
    echo "Rendering video for $ENV_ID..."
    python scripts/ppo_builderbench_rollout_video.py \
        --checkpoint="$LATEST_CKPT" \
        --env="$ENV_ID" \
        --output="$OUTPUT_DIR"
    echo "Video saved to $OUTPUT_DIR"
}

# --- Standalone Usage Example ---
# You can specify the command exactly as you did in your training loop
MY_CMD="$1"

render_from_cmd "$MY_CMD"
