#!/bin/bash
#SBATCH --job-name=ext_ent_anneal_05_precision_online_cat_select_c4c4ba51_s1_builderbench_creative_4_task1
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --output=%j.out

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench
export MUJOCO_GL=egl

if [ -d "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/ext_ent_anneal_05_precision_online_cat_select_c4c4ba51" ]; then
    echo "Warning: Directory exists, deleting to ensure clean restart."
    rm -rf "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/ext_ent_anneal_05_precision_online_cat_select_c4c4ba51"
fi

python -u ppo_contrastive.py             --seed=1             --log_dir_path=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/ext_ent_anneal_05_precision_online_cat_select_c4c4ba51             --exp_name=ext_ent_anneal_05_precision_online_cat_select             --num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=150 --ppo_video_interval=150 --ppo_video_fps=10 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes="256,256,256,256,256,256" --env=builderbench_creative_3_task1 --ppo_rollout_length=50 --ppo_crl_steps_per_iter=25 --use_wandb=true --wandb_project=dist-matching --wandb_entity=doina-precup --wandb_mode=online             --env=builderbench_creative_4_task1 --ppo_actor_min_std=0.01 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=600000000 --ppo_categorical_select
