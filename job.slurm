#!/bin/bash
#SBATCH --job-name=contrastive_gpu                              
#SBATCH --cpus-per-task=4         
#SBATCH --mem-per-cpu=8G               
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

cd /home/mb6458/new-sgcrl/sgcrl || exit 1


python -u lp_contrastive.py --seed 99887766 --env sawyer_box


exit 0  