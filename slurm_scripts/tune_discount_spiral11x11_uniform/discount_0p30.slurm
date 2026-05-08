#!/bin/bash
#SBATCH --job-name=ppo_spiral11x11_uniform_0p30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-2
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

SEED=${SLURM_ARRAY_TASK_ID}
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_spiral11x11_discount_tune_uniform/discount_0p30/

CRL_STEPS=64
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.30
PPO_CLIP_COEF=0.2

python /home/gliu2/dist_matching/sgcrl/ppo_contrastive.py \
    --env=point_Spiral11x11 \
    --seed=${SEED} \
    --num_steps=2000000 \
    --log_dir_path=${LOG_DIR} \
    --ppo_crl_steps_per_iter=${CRL_STEPS} \
    --ppo_ent_coef=${ENT_COEF} \
    --ppo_actor_min_std=${MIN_STD} \
    --ppo_discount=${DISCOUNT_FACTOR} \
    --ppo_clip_coef=${PPO_CLIP_COEF} \
    --uniform_sampling
