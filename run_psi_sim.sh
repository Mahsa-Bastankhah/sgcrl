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
seeds=(200 201 202 203 204 205 206 207)
seeds=(308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323)
seeds=(550 551 552 553 554 555 556 557)
seeds=(60)
#seeds=(228 229 230 231 232 233 234 235)
#seeds=(330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345)
#seeds=(924 925 926 927 928 929 930 931)
###### 0  0  1  1  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  1
devices=(7)        # GPU index for each run
#actor_sample
#q_max
for idx in "${!seeds[@]}"; do
  SEED=${seeds[$idx]}
  DEV=${devices[$idx]}
  echo "Running for seed $SEED"
  CUDA_VISIBLE_DEVICES="" \
  nohup python -m experiments.similarity_posterior_exp \
    --alpha 0.1 \
    --env_name point_Spiral11x11\
    --seed $SEED \
    --log_dir logs \
    --alg contrastive_cpc \
    --ckpt_list 2 3 4 5 6 7 8 9 10 12\
    --NUM_AXES 2 \
    --NUM_EPISODES 5 \
    --plot_psi \
    --action_mode "actor_sample"\
    > "similarity_seed${SEED}.log" 2>&1 &
done