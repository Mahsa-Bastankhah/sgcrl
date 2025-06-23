#!/bin/bash
# run_psi_stats.sh  – batch-run psi_stats.py for one or many checkpoints
#
# Usage:
#   ./run_psi_stats.sh        # uses the values in the CUSTOMISE ME block
#   or edit variables below.

# ────────────────── CUSTOMISE ME ──────────────────────────────
ENV_NAME="point_Wall11x11"
LOG_DIR="logs"
######################################### WALL #########################################
# list of seeds and matching success flags (1 = success, 0 = fail)
SEEDS=(2  3  4  5  21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36)
###### 0  0  1  1  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  1
SUCCESS=(0  0  1  1  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  1)
# list of seeds and matching success flags (1 = success, 0 = fail)
# SEEDS=(2 )
# SUCCESS=(0)

## 6 layers exp: 110 , 111, 112, 113, 114, 115, 116, 117, 118


######################################### SPIRAL #########################################
SEEDS=(2  3  4  5  21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36)
###### 0  0  1  1  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  1
SUCCESS=(0  0  1  1  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  1)
# checkpoints you want to analyse
CKPTS=(1 2 3 4 5 8 10)

SEEDS=(26 1100 1101 1102 1103 1104 1105 1106)
SUCCESS=(1 1     1   1    1    1    1    0  )


# SEEDS=(36 800 801 802 803 804 805 806 807)
# SUCCESS=(1 0   1   0   1   1    0   1   0 )

# set PROJECTED=true to use projected ψ; false = raw ψ
PROJECTED=false
# ──────────────────────────────────────────────────────────────

# --- conda env setup (adjust path if needed) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_rl_nn
echo "CONDA_PREFIX = $CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# ▲ Disable all GPUs for everything launched in this script
export CUDA_VISIBLE_DEVICES=""

# (optional extra guard for JAX; harmless for others)
export JAX_PLATFORM_NAME="cpu"   

# flag for psi_stats
PROJECT_FLAG=""
$PROJECTED && PROJECT_FLAG="--projected"

# iterate over checkpoints
for CKPT in "${CKPTS[@]}"; do
  echo "▶ analysing ckpt $CKPT  (projected = $PROJECTED)"
  python -m experiments.report_similarity_stat \
    --env "$ENV_NAME" \
    --log_dir "$LOG_DIR" \
    --ckpt "$CKPT" \
    --seeds  "${SEEDS[@]}" \
    --success "${SUCCESS[@]}" \
    --metric "psi_similarity" \
    ${PROJECT_FLAG}

  echo "✓ done ckpt $CKPT"
done

echo "🚀  All psi-stats jobs finished."
