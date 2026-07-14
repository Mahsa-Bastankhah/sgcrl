#!/bin/bash
# ==============================================================================
# SGCRL PPO Contrastive Baseline + Evaluation
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ---------- USER CONFIGURATION ------------------------------------------------
# ENVS=( "builderbench_creative_3_task1" )
# ENVS=( "builderbench_creative_4_task1" "builderbench_creative_4_task6" "builderbench_creative_3_task2" "builderbench_creative_3_task5")
SEEDS=( 0 1 )

LOG_ROOT="/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"
# ------------------------------------------------------------------------------

BASE_FLAGS="--num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=0 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes=\"256,256,256,256,256,256\" --env=builderbench_creative_3_task1"

EXPERIMENTS=( 

    # "debug|--num_steps=20_000_000"
    # "reproduce|--env=builderbench_creative_4_task1"
    "reproduce|--env=builderbench_creative_4_task6 --obs_space="xy,quaternions,select""
    "reproduce|--env=builderbench_creative_3_task2 -obs_space="xy,quaternions,select""
    # "reproduce2|"



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
#SBATCH --time=24:00:00
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
            TRAIN_OUTPUT=$(sbatch "$SLURM_SCRIPT")
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
#SBATCH --dependency=afterany:${TRAIN_JOB_ID}

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
