#!/bin/bash
# run_lp_contrastive.sh
# Launch lp_contrastive.py four times (seeds 12‑15) on GPUs 0‑3.
# Wait until all specified GPUs are free

# ------------------------------------------------------------------
# 1) Activate the conda env
# ------------------------------------------------------------------
# (Replace ~/miniconda3 with your own Miniconda/Anaconda path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_rl_nn            # ➜ env: contrastive_rl_nn

# ------------------------------------------------------------------
# 2) Show CONDA_PREFIX and update LD_LIBRARY_PATH
# ------------------------------------------------------------------
echo "$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"



# ------------------------------------------------------------------
# 3) Launch one run per seed / GPU
# ------------------------------------------------------------------
############## WALL
##seeds=(6 7 8 48 49 50 51 52 53 54 55) Relu after psi # List of seeds to run
## I think wall 40-47 is for relu after psi too
## 110 111 112 113 114 115 116 117  default sgcrl but with 6 layer networks
#default sgcrl with 2 layers that I did for histograms (2  3  4  5  21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36)
############## SPIRAL
## spiral 30-37 default sgcrl but with 6 layers


seeds=(1106)
devices=(0)




for idx in "${!seeds[@]}"; do
  SEED=${seeds[$idx]}
  DEV=${devices[$idx]}
  LOG="lp_contrastive_seed${SEED}.out"

  echo "▶ Launching seed $SEED on GPU $DEV  →  $LOG"
  CUDA_VISIBLE_DEVICES=$DEV \
    nohup python lp_contrastive.py \
      --env point_Wall11x11 \
      --seed "$SEED" \
      --num_steps 4000000 \
      > "$LOG" 2>&1 &
done




wait
echo "✅ All jobs launched (logs in lp_contrastive_seed*.out)"

