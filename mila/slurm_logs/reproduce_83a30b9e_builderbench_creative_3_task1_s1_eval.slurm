#!/bin/bash
#SBATCH --job-name=eval_reproduce_83a30b9e_s1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=%j.out
#SBATCH --dependency=afterany:10105009

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench

# Ensure eval output goes where the WandB sync script expects it
mkdir -p /network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_83a30b9e/ppo_builderbench_creative_3_task1_1/logs/eval/
echo "{\"command\": \"python -u ppo_contrastive.py                 --env=builderbench_creative_3_task1                 --seed=1                 --log_dir_path=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_83a30b9e                 --exp_name=reproduce                 --num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=0 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes="256,256,256,256,256,256"                 \"}" > "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_83a30b9e/ppo_builderbench_creative_3_task1_1/metadata.json"

# Run the evaluation
python scripts/ppo_builderbench_checkpoint_eval.py     --run_dir=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_83a30b9e/ppo_builderbench_creative_3_task1_1     --csv_output=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_83a30b9e/ppo_builderbench_creative_3_task1_1/logs/eval/logs.csv     --incremental

# Immediately sync the results of this experiment to WandB
python scripts/csv_runs_to_wandb.py     --project dist-matching     --entity doina-precup     --run-dir /network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/reproduce_83a30b9e/ppo_builderbench_creative_3_task1_1
