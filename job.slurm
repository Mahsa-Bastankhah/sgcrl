#!/bin/bash
#SBATCH --job-name=contrastive_gpu     
#SBATCH --array=0-7%8                            
#SBATCH --cpus-per-task=1         
#SBATCH --mem-per-cpu=16G               
#SBATCH --gres=gpu:1         
#SBATCH --time=5:00:00                                   
#SBATCH --mail-user=mb6458@princeton.edu
#SBATCH --account=eysenbach

module purge
module load anaconda3/2023.3
module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0

# Activate the correct environment
source /home/mb6458/miniconda3/bin/activate contrastive_rl

# Set the library paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mb6458/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


# *** NEW: Adjust JAX GPU memory preallocation ***
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# Debug prints
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi

seeds=(788 789 790 791 792 793 794 795)
seeds=(567667 567668 567669 567670 567671 567672)
# seeds=(20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}


#hidden_sizes=(256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256)
#hidden_sizes=(256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256)
hidden_sizes=(256 256 256 256 256 256)
hidden_flags=()
for h in "${hidden_sizes[@]}"; do
  hidden_flags+=(--hidden_layer_sizes "$h")
done


# python lp_contrastive.py \
#     --env point_Spiral11x11 \
#     --seed "$SEED" \
#     --goal_neg_actor_steps 5000000 \
#     --num_steps 5000000 \
#     "${hidden_flags[@]}"
  ##  --goal_pos_actor_steps 4000000 \
# --perturbed_negatives_num 5 \
    # --perturbed_negatives_goal_num 5\
    #    --use_residual_mlp \
# 

python -u lp_contrastive.py \
    --env point_Wall11x11 \
    --seed "$SEED" \
    --num_steps 8000000 \
    "${hidden_flags[@]}" \
    --time_delta_minutes 40 \



