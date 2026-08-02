#!/bin/bash
# ==============================================================================
# SGCRL PPO Contrastive Baseline + Evaluation
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ---------- USER CONFIGURATION ------------------------------------------------
# ENVS=( "builderbench_creative_3_task1" )
# ENVS=( "builderbench_creative_4_task1" "builderbench_creative_4_task6" "builderbench_creative_3_task2" "builderbench_creative_3_task5")
SEEDS=( 0  1 )

LOG_ROOT="/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"
# ------------------------------------------------------------------------------

BASE_FLAGS="--num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=0 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes=\"256,256,256,256,256,256\" --env=builderbench_creative_3_task1 --ppo_rollout_length=50 --ppo_crl_steps_per_iter=25"

EXPERIMENTS=( 

    # "debug|--num_steps=20_000_000"
    # "reproduce|--env=builderbench_creative_4_task1 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"
    # "reproduce|--env=builderbench_creative_4_task6 --obs_space="xy,quaternions,select""
    # "reproduce|--env=builderbench_creative_3_task2 -obs_space="xy,quaternions,select""
    # "reproduce2|"
    # "reproduce_longer_rollout|--env=builderbench_creative_4_task1 --num_steps=300000000 --ppo_rollout_length=128" # Roughly double which is automatically calculated to 60.
    # "reproduce_longer_rollout|--env=builderbench_creative_4_task1 --num_steps=300000000 --ppo_rollout_length=256" # Roughly quadruple which is automatically calculated to 60.
    # "reproduce|--env=builderbench_creative_2_task2"
    # "reproduce|--env=builderbench_creative_2_task3"
    # "reproduce_warmup|--env=builderbench_creative_4_task1 --num_steps=300000000 --ppo_warmup_percent=0.05"
    # "reproduce_warmup|--env=builderbench_creative_4_task1 --num_steps=300000000 --ppo_warmup_percent=0.10"
    # "reproduce_warmup|--env=builderbench_creative_4_task1 --num_steps=300000000 --ppo_warmup_percent=0.20"
    # "reproduce_cleanr_actor|--env=builderbench_creative_4_task1 --num_steps=300000000 --ppo_cleanrl_actor=True"
    # "reproduce_longer_env_epi|--env=builderbench_creative_4_task1 --num_steps=300000000 --builderbench_episode_length_multiplier=5"
    # "reproduce_longer_env_epi|--env=builderbench_creative_4_task1 --num_steps=300000000 --builderbench_episode_length_multiplier=10"
    # "reproduce_longer_env_epi|--env=builderbench_creative_4_task6 --builderbench_episode_length_multiplier=5"
    # "reproduce_rms_obs_norm|--env=builderbench_creative_4_task1 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000 --obs_norm_mode=tied_rsnorm"
    # "reproduce_z_scale|--env=builderbench_creative_4_task1 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000 --obs_norm_mode=z_scale"
    # "pd_nf|--env=builderbench_creative_4_task1 --num_steps=300000000 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --ppo_repr_mode=nf --nf_rep_size=256 --nf_num_blocks=12 --nf_coupling_width=512 --nf_grad_clip=1.0 --nf_noise_std=0.05 --nf_goal_std_min=0.02 --nf_goal_enc_size=0"
    # "pd_nf_tau05|--env=builderbench_creative_4_task1 --num_steps=300000000 --builderbench_permute_start_boxes=true --ppo_repr_mode=nf --ppo_nf_reward_tau=0.5 --nf_rep_size=256 --nf_num_blocks=12 --nf_coupling_width=512 --nf_grad_clip=1.0 --nf_noise_std=0.05 --nf_goal_std_min=0.02 --nf_goal_enc_size=0"
    # "pd_td3_logq_tau05|--env=builderbench_creative_4_task1 --num_steps=300000000 --builderbench_permute_start_boxes=true --ppo_repr_mode=td3 --noppo_td3_cross_batch_goals --ppo_td3_log_reward --ppo_td3_reward_tau=0.5"

    # --- Previous Active Experiments (Commented out) ---
    # 1. Permutation Isolation (hue: builderbench_permute_start_boxes)
    # "permute_start_boxes|--env=builderbench_creative_4_task1 --builderbench_permute_start_boxes=false --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"
    # "permute_start_boxes|--env=builderbench_creative_4_task1 --builderbench_permute_start_boxes=true --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"

    # 2. Exploration & Selection Bin Un-locking (hue: ppo_ent_coef)
    # "high_std_ent_anneal|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"
    # "high_std_ent_anneal|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.10 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"
    # "high_std_ent_anneal|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.20 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"

    # 4. Representation Smoothing (hue: ppo_crl_repr_tau)
    # "crl_tau|--env=builderbench_creative_4_task1 --ppo_crl_repr_tau=0.05 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"
    # "crl_tau|--env=builderbench_creative_4_task1 --ppo_crl_repr_tau=0.10 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"
    # "crl_tau|--env=builderbench_creative_4_task1 --ppo_crl_repr_tau=0.50 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"

    # 7. TD3 Density Mode Goal Tolerance (hue: ppo_td3_goal_tol)
    # "td3_tol|--env=builderbench_creative_4_task1 --ppo_repr_mode=td3 --ppo_td3_goal_tol=0.02 --noppo_td3_cross_batch_goals --ppo_td3_log_reward --ppo_td3_reward_tau=0.5 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"
    # "td3_tol|--env=builderbench_creative_4_task1 --ppo_repr_mode=td3 --ppo_td3_goal_tol=0.04 --noppo_td3_cross_batch_goals --ppo_td3_log_reward --ppo_td3_reward_tau=0.5 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=300000000"

    # --- Active Experiments: Extended Timesteps & Multi-Stage Entropy Annealing (600M Steps) ---
    "ext_ent_anneal_05_to_01|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=600000000"
    "ext_ent_anneal_05_to_005|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.005 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=600000000"
    "ext_ent_anneal_10_to_005|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.10 --ppo_ent_coef_final=0.005 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=600000000"
    "ext_ent_anneal_05_precision|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.01 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=600000000"

    # --- Active Experiments: Good Experience Replay & Bootstrapping (SIL & Mixed) ---
    "good_buf_sil|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_use_good_buffer=True --ppo_good_buffer_mode=sil --ppo_good_buffer_min_cubes=2 --ppo_good_buffer_coef=0.1 --num_steps=600000000"
    "good_buf_sil|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_use_good_buffer=True --ppo_good_buffer_mode=sil --ppo_good_buffer_min_cubes=4 --ppo_good_buffer_coef=0.1 --num_steps=600000000"
    "good_buf_mixed|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_use_good_buffer=True --ppo_good_buffer_mode=mixed --ppo_good_buffer_min_cubes=2 --ppo_good_buffer_coef=0.5 --num_steps=600000000"
    "good_buf_mixed|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.03 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_use_good_buffer=True --ppo_good_buffer_mode=mixed --ppo_good_buffer_min_cubes=4 --ppo_good_buffer_coef=0.5 --num_steps=600000000"
)

mkdir -p "$SCRIPT_DIR/slurm_logs"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    EXP_NAME="${EXPERIMENT%%|*}"
    EXP_FLAGS="${EXPERIMENT##*|}"

    if [[ "$EXP_FLAGS" =~ --env=([^ ]+) ]]; then
        ENV_ID="${BASH_REMATCH[1]}"
    else
        ENV_ID="builderbench_creative_3_task1" 
    fi
    echo "Running experiment: $EXP_NAME with env: $ENV_ID and flags: $EXP_FLAGS"


    # for ENV_ID in "${ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do


            # 1. Create a unique signature of the parameters and hash it
            SIG_STR="env=${ENV_ID}_seed=${seed}_base=${BASE_FLAGS}_exp=${EXP_FLAGS}"
            CMD_HASH=$(echo "$SIG_STR" | md5sum | cut -c1-8)
            SAFE_NAME="${EXP_NAME//+/_plus_}_${CMD_HASH}"

            # 2. Build the command using SAFE_NAME for the log_dir_path
            CMD="python -u ppo_contrastive.py \
                --seed=${seed} \
                --log_dir_path=${LOG_ROOT}/${SAFE_NAME} \
                --exp_name=${EXP_NAME} \
                ${BASE_FLAGS} \
                ${EXP_FLAGS}"


            echo "CMD: $CMD for SAFE_NAME: $SAFE_NAME"
            SLURM_SCRIPT="$SCRIPT_DIR/slurm_logs/${SAFE_NAME}_${ENV_ID}_s${seed}.slurm"

            # 3. Create Training Script
            cat <<EOT > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${SAFE_NAME}_s${seed}_${ENV_ID}
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
if [ -d "${LOG_ROOT}/${SAFE_NAME}" ]; then
    echo "Warning: Directory exists, deleting to ensure clean restart."
    rm -rf "${LOG_ROOT}/${SAFE_NAME}"
fi
${CMD}
EOT

            echo "Submitting ${EXP_NAME} | env=${ENV_ID} | seed=${seed}..."
            # TRAIN_OUTPUT=$(sbatch "$SLURM_SCRIPT")
            TRAIN_JOB_ID=$(echo "$TRAIN_OUTPUT" | awk '{print $4}')
            echo "  Training Job ID: $TRAIN_JOB_ID"

            echo "Run dir is ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}"

            # 4. Create Evaluation Script (points to the SAFE_NAME directory)
            EVAL_SCRIPT="$SCRIPT_DIR/slurm_logs/${SAFE_NAME}_${ENV_ID}_s${seed}_eval.slurm"
            cat <<EOT > "$EVAL_SCRIPT"
#!/bin/bash
#SBATCH --job-name=eval_${SAFE_NAME}_s${seed}
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=%j.out
##SBATCH --dependency=afterany:${TRAIN_JOB_ID}

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench

# Ensure eval output goes where the WandB sync script expects it
mkdir -p ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/logs/eval/
echo "{\"command\": \"${CMD}\"}" > "${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/metadata.json"

# Run the evaluation
python scripts/ppo_builderbench_checkpoint_eval.py \
    --run_dir=${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed} \
    --csv_output=${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/logs/eval/logs.csv


mkdir -p ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/videos/
python scripts/ppo_builderbench_rollout_video.py \
    --checkpoint=${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/checkpoints \
    --env=${ENV_ID} \
    --output=${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/videos/ \
    --fps=10

# Immediately sync the results of this experiment to WandB
python scripts/csv_runs_to_wandb.py \
    --project dist-matching \
    --entity doina-precup \
    --run-dir ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}
EOT

            sbatch "$EVAL_SCRIPT"
            echo "  Submitted evaluation job dependent on $TRAIN_JOB_ID"

        done  # seeds
    # done  # envs
done  # experiments

echo ""
echo "All jobs submitted."
