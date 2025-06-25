#!/bin/bash


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
#devices=(0 1 2 3 4 5 6 7)   

#seeds=(2 3 4 5) # List of seeds to run
seeds=(5500 5501 5502 5503 5504 5505 5506 5507)
###### 0  0  1  1  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  1
devices=(7)        # GPU index for each run

for idx in "${!seeds[@]}"; do
  SEED=${seeds[$idx]}
  DEV=${devices[$idx]}
  echo "Running for seed $SEED"
  CUDA_VISIBLE_DEVICES="" \
  nohup python -m experiments.similarity_posterior_exp \
    --alpha 0.1 \
    --env_name point_Wall11x11 \
    --seed $SEED \
    --log_dir logs \
    --alg contrastive_cpc \
    --ckpt_list 1 2 3 4 5 6 7 8 9 10 11 12 14 \
    --NUM_AXES 2 \
    --NUM_EPISODES 5 \
    --plot_psi \
    --plot_phi_psi \
    --action_mode "actor_max"\
    > "similarity_seed${SEED}.log" 2>&1 &
done
