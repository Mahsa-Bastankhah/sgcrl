#!/bin/bash
# run_lp_contrastive.sh
# Launch lp_contrastive.py four times (seeds 12‑15) on GPUs 0‑3.
# Wait until all specified GPUs are free

# ------------------------------------------------------------------
# 1) Activate the conda env
# ------------------------------------------------------------------
# (Replace ~/miniconda3 with your own Miniconda/Anaconda path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_rl            # ➜ env: contrastive_rl_nn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
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


# seeds=(4000)
# devices=(7)

seeds=(456789)
devices=(0 1 2 3 4 5 6 7)
# ➊  Add this block just after you define the seeds / devices
hidden_sizes=(256 256 256 256 256 256)  # 12 layers
hidden_flags=()
for h in "${hidden_sizes[@]}"; do
  hidden_flags+=(--hidden_layer_sizes "$h")
done

for idx in "${!seeds[@]}"; do
  SEED=${seeds[$idx]}
  DEV=${devices[$idx]}
  LOG="lp_contrastive_seed${SEED}.out"

  echo "▶ Launching seed $SEED on GPU $DEV  →  $LOG"
    CUDA_VISIBLE_DEVICES=$DEV \
    nohup python -u lp_contrastive.py \
      --time_delta_minutes 15 \
      --env point_FourRooms \
      --region_bounds="0,6:8,11;  0,1:4,6; 7,0:11,4" \
      --seed "$SEED" \
      --num_steps 100000 \
      "${hidden_flags[@]}" \
      > "$LOG" 2>&1 &   # redirection now belongs to the nohup command
done


      # --init_weight "logs/contrastive_cpc_point_Wall11x11_26/checkpoints/learner" \


echo "✅ All jobs launched (logs in lp_contrastive_seed*.out)"

      # --goal_neg_actor_steps 50000 \
            # --cold_q_init \
            # --cold_q_scale 1e-12 \
                  # --perturbed_negatives_goal_num 5 \
      #                   --goal_neg_actor_steps 50000 \
      # --goal_pos_actor_steps 50000 \
      # --goal_pos_frac 0.05 \