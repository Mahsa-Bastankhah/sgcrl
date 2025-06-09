#!/bin/bash
#SBATCH --job-name=cpc_impossible # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                 # memory (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:30:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=gliu2@andrew.cmu.edu
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

conda init
source activate base
conda activate contrastive_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/home/gliu2/miniconda/envs/contrastive_rl/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export CPATH=/home/gliu2/miniconda/envs/contrastive_rl/include
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4

python /home/gliu2/sgcrl/lp_contrastive.py --alg contrastive_cpc --env point_Impossible --num_steps 500_000 --seed 0 --log_dir_path '/home/gliu2/rebuttal/checkpoints/' --time_delta_minutes 5