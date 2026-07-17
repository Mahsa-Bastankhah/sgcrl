#!/bin/bash
#SBATCH --job-name=reproduce_01efb4e7_s1_builderbench_creative_3_task2
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --output=%j.out

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench
if [ -d "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_01efb4e7" ]; then
    echo "Warning: Directory exists, deleting to ensure clean restart."
    rm -rf "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_01efb4e7"
fi
python -u ppo_contrastive.py                 --seed=1                 --log_dir_path=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_01efb4e7                 --exp_name=reproduce                 --num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=0 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes="256,256,256,256,256,256" --env=builderbench_creative_3_task1                 --env=builderbench_creative_3_task2 -obs_space=xy,quaternions,select
