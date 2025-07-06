#!/bin/bash


# ------------------------------------------------------------------
# 1) Activate the conda env
# ------------------------------------------------------------------
# (Replace ~/miniconda3 with your own Miniconda/Anaconda path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_rl            # ➜ env: contrastive_rl_nn

# ------------------------------------------------------------------
# 2) Show CONDA_PREFIX and update LD_LIBRARY_PATH
# ------------------------------------------------------------------
echo "$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"
#devices=(0 1 2 3 4 5 6 7)   

#seeds=(2 3 4 5) # List of seeds to run
seeds=(780 781 782 783 784 785 786 787)
###### 0  0  1  1  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  1
devices=(7)        # GPU index for each run

for idx in "${!seeds[@]}"; do
  SEED=${seeds[$idx]}
  DEV=${devices[$idx]}
  echo "Running for seed $SEED"
  CUDA_VISIBLE_DEVICES="" \
  nohup python -m experiments.similarity_posterior_exp \
    --alpha 0.1 \
    --env_name point_Impossible \
    --seed $SEED \
    --log_dir logs \
    --alg contrastive_cpc \
    --ckpt_list 2 4 6 8 10 12 14 16\
    --NUM_AXES 2 \
    --NUM_EPISODES 5 \
    --plot_psi \
    --action_mode "actor_sample"\
    > "similarity_seed${SEED}.log" 2>&1 &
done
