#!/bin/bash
#SBATCH --job-name=pd_fm_creative3_task1_all_tricks_anneal_ent_coeff_ca59974d_s0_builderbench_creative_3_task1
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
if [ -d "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/pd_fm_creative3_task1_all_tricks_anneal_ent_coeff_ca59974d" ]; then
    echo "Warning: Directory exists, deleting to ensure clean restart."
    rm -rf "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/pd_fm_creative3_task1_all_tricks_anneal_ent_coeff_ca59974d"
fi
python -u ppo_contrastive.py                 --seed=0                 --log_dir_path=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/pd_fm_creative3_task1_all_tricks_anneal_ent_coeff_ca59974d                 --exp_name=pd_fm_creative3_task1_all_tricks_anneal_ent_coeff                 --num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=0 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes="256,256,256,256,256,256" --env=builderbench_creative_3_task1 --ppo_rollout_length=50 --ppo_crl_steps_per_iter=25                 --env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_actor_min_std=0.01
