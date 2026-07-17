#!/bin/bash
#SBATCH --job-name=eval_reproduce_warmup_7fa79d02_s0
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=%j.out
#SBATCH --dependency=afterany:10161023

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench

# Ensure eval output goes where the WandB sync script expects it
mkdir -p /network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02/ppo_builderbench_creative_4_task1_0/logs/eval/
echo "{\"command\": \"python -u ppo_contrastive.py                 --seed=0                 --log_dir_path=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02                 --exp_name=reproduce_warmup                 --num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=0 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes="256,256,256,256,256,256" --env=builderbench_creative_3_task1                 --env=builderbench_creative_4_task1 --ppo_warmup_percent=0.05\"}" > "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02/ppo_builderbench_creative_4_task1_0/metadata.json"

# Run the evaluation
python scripts/ppo_builderbench_checkpoint_eval.py     --run_dir=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02/ppo_builderbench_creative_4_task1_0     --csv_output=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02/ppo_builderbench_creative_4_task1_0/logs/eval/logs.csv


mkdir -p /network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02/ppo_builderbench_creative_4_task1_0/videos/
python scripts/ppo_builderbench_rollout_video.py     --checkpoint=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02/ppo_builderbench_creative_4_task1_0/checkpoints     --env=builderbench_creative_4_task1     --output=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02/ppo_builderbench_creative_4_task1_0/videos/     --fps=10

# Immediately sync the results of this experiment to WandB
python scripts/csv_runs_to_wandb.py     --project dist-matching     --entity doina-precup     --run-dir /network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_warmup_7fa79d02/ppo_builderbench_creative_4_task1_0
