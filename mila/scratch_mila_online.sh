#!/bin/bash
# ==============================================================================
# SGCRL PPO Contrastive Baseline + Online WandB Evaluation & In-Train Video Rendering
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ---------- USER CONFIGURATION ------------------------------------------------
SEEDS=( 0  1 )

LOG_ROOT="/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"
# ------------------------------------------------------------------------------

BASE_FLAGS="--num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=150 --ppo_video_interval=150 --ppo_video_fps=10 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes=\"256,256,256,256,256,256\" --env=builderbench_creative_3_task1 --ppo_rollout_length=50 --ppo_crl_steps_per_iter=25 --use_wandb=true --wandb_project=dist-matching --wandb_entity=doina-precup --wandb_mode=online"

EXPERIMENTS=( 
    # --- Active Flow Matching Experiments ---
    # Baseline
    "pd_fm_creative3_task1_baseline|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select"

    # Goal Normalization
    "pd_fm_creative3_task1_goal_norm|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_norm_goals=True"

    # All Tricks + Goal Noise + Goal Norm
    "pd_fm_creative3_task1_all_tricks_noise_norm|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --fm_goal_noise_std=0.02 --fm_norm_goals=True"

    # --- External Reward Scale = 1 Variants ---
    # Baseline (scale=1)
    "pd_fm_creative3_task1_baseline_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1"

    # Goal Normalization (scale=1)
    "pd_fm_creative3_task1_goal_norm_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_norm_goals=True --ppo_external_reward_scale=1"

    # All Tricks + Goal Noise + Goal Norm (scale=1)
    "pd_fm_creative3_task1_all_tricks_noise_norm_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --fm_goal_noise_std=0.02 --fm_norm_goals=True --ppo_external_reward_scale=1"
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

        # 3. Create Training Script (with Online Eval, Video Rendering, and WandB Logging)
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
export MUJOCO_GL=egl

if [ -d "${LOG_ROOT}/${SAFE_NAME}" ]; then
    echo "Warning: Directory exists, deleting to ensure clean restart."
    rm -rf "${LOG_ROOT}/${SAFE_NAME}"
fi

${CMD}
EOT

        echo "Submitting ${EXP_NAME} | env=${ENV_ID} | seed=${seed}..."
        TRAIN_OUTPUT=$(sbatch "$SLURM_SCRIPT")
        TRAIN_JOB_ID=$(echo "$TRAIN_OUTPUT" | awk '{print $4}')
        echo "  Training Job ID: $TRAIN_JOB_ID"
        echo "  Run dir is ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}"

    done  # seeds
done  # experiments

echo ""
echo "All jobs submitted cleanly. Training, evaluation, video rendering, and WandB logging will execute live."
