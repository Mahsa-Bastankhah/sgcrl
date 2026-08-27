#!/bin/bash
#SBATCH --job-name=ppo_maniskill_pickcube
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# NOTE: --time/--mem are estimates, not measured for the 6x256 ResidualMLP
# architecture used here, or for PickCube specifically (heavier physics/
# success checks than PushCube). Revisit after the first real run's `sps`
# metric is observed in the learner log.

conda init
source activate base
conda activate /data/user_data/gliu2/conda_envs/contrastive_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/home/gliu2/miniconda/envs/contrastive_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CPATH=/home/gliu2/miniconda/envs/contrastive_rl/include
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
# Installing mani_skill/torch/gymnasium into this conda env pulled in a
# much newer protobuf than the pinned tensorflow==2.8/reverb stack
# expects, which otherwise breaks `import contrastive` with
# `TypeError: Descriptors cannot be created directly.`
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_ENTITY=grace_rl

SEED=${SLURM_ARRAY_TASK_ID}
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_maniskill_pickcube_1M/

CRL_STEPS=64
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2

python /home/gliu2/dist_matching/sgcrl/ppo_contrastive.py \
    --env=maniskill_pickcube \
    --seed=${SEED} \
    --num_steps=1000000 \
    --log_dir_path=${LOG_DIR} \
    --hidden_layer_sizes="256,256,256,256,256,256" \
    --ppo_crl_steps_per_iter=${CRL_STEPS} \
    --ppo_ent_coef=${ENT_COEF} \
    --ppo_actor_min_std=${MIN_STD} \
    --ppo_discount=${DISCOUNT_FACTOR} \
    --ppo_clip_coef=${PPO_CLIP_COEF} \
    --uniform_sampling \
    --wandb_project=sgcrl-maniskill \
    --render_video
