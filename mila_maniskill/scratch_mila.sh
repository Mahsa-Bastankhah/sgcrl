#!/bin/bash

# ==============================================================================
# SECTION 1: USER CONFIGURATION
# ==============================================================================

# 1. Define where the temp file lives
#    We use the script's own directory to keep things contained
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TASK_FILE="$SCRIPT_DIR/mila_tasks.tmp"
rm -f "$TASK_FILE" # Clear old runs

# 2. Get Project Root (So we can find train.py)
#    Assumes this script is in <root>/tamia/scratch_tamia.sh
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. YOUR EXPERIMENT LOOPS
#    Append every command you want to run to $TASK_FILE
echo "Generating tasks..."

for seed in 0 1; do

for arch in contrastive; do


exp_name='creative-5-task2_stable_crl_eval_sg'
echo "python stable_crl.py --env_id creative-5-task2 --use_pd --pd_duration 5 --architecture ${ach} --repetition_factor 12 --entropy_cost 0.01 --seed=${seed} --exp_name=${exp_name} --eval_single_goal True" >> "$TASK_FILE"

# exp_name='creative-5-task1_stable_crl_eval_sg'
# echo "python stable_crl.py --env_id creative-5-task1 --use_pd --pd_duration 5 --architecture ${ach} --repetition_factor 12 --entropy_cost 0.01 --seed=${seed} --exp_name=${exp_name} --eval_single_goal True" >> "$TASK_FILE"
#
#
# exp_name='creative-7-task2_stable_crl_eval_sg'
# echo "python stable_crl.py --env_id creative-7-task2 --use_pd --pd_duration 5 --architecture ${ach} --repetition_factor 12 --entropy_cost 0.01 --seed=${seed} --exp_name=${exp_name} --eval_single_goal True" >> "$TASK_FILE"
#
#
# exp_name='creative-8-task2_stable_crl_eval_sg'
# echo "python stable_crl.py --env_id creative-8-task2 --use_pd --pd_duration 5 --architecture ${ach} --repetition_factor 12 --entropy_cost 0.01 --seed=${seed} --exp_name=${exp_name} --eval_single_goal True" >> "$TASK_FILE"

exp_name='creative-5-task2_sgcrl'
echo "python stable_crl.py --env_id creative-5-task2 --use_pd --pd_duration 5 --architecture ${ach} --seed=${seed} --exp_name=${exp_name} --single_goal" >> "$TASK_FILE"

exp_name='creative-4-task1_sgcrl'
echo "python stable_crl.py --env_id creative-5-task2 --use_pd --pd_duration 5 --architecture ${ach} --seed=${seed} --exp_name=${exp_name} --single_goal" >> "$TASK_FILE"

# exp_name='creative-5-task1_sgcrl'
# echo "python stable_crl.py --env_id creative-5-task1 --use_pd --pd_duration 5 --architecture ${ach} --seed=${seed} --exp_name=${exp_name} --single_goal" >> "$TASK_FILE"
#
#
# exp_name='creative-7-task2_sgcrl'
# echo "python stable_crl.py --env_id creative-7-task2 --use_pd --pd_duration 5 --architecture ${ach} --seed=${seed} --exp_name=${exp_name} --single_goal" >> "$TASK_FILE"
#
#
# exp_name='creative-8-task2_sgcrl'
# echo "python stable_crl.py --env_id creative-8-task2 --use_pd --pd_duration 5 --architecture ${ach} --seed=${seed} --exp_name=${exp_name} --single_goal" >> "$TASK_FILE"

done
done


# for seed in 0 1 ; do
#   exp_name="creative-8-task2_cat_sel_sgcrl_fair_s${seed}"
#   echo "python stable_crl.py --seed=${seed} --single_goal --categorical_select --env_id creative-8-task2 --use_pd --pd_duration 5 --architecture default --no-permute_start_boxes --entropy_cost 0.05 --entropy_cost_final 0.0 --num_timesteps 300000000 --exp_name=${exp_name}" >> "$TASK_FILE"
# done
#
# Comparison runs matching NF-default baseline setup:
# 1. --no-permute_start_boxes (fixed start lane per cube ID)
# 2. --entropy_cost 0.05 --entropy_cost_final 0.0 (entropy annealing 0.05 -> 0.0)
# 3. --num_timesteps 300000000 (300M steps)

# for seed in 0 1 ; do
#   exp_name="creative-8-task2_cat_sel_sgcrl_fair_s${seed}"
#   echo "python stable_crl.py --seed=${seed} --single_goal --categorical_select --env_id creative-8-task2 --use_pd --pd_duration 5 --architecture default --no-permute_start_boxes --entropy_cost 0.05 --entropy_cost_final 0.0 --num_timesteps 300000000 --exp_name=${exp_name}" >> "$TASK_FILE"
# done
#
# for seed in 0 1 2; do
#   exp_name="creative-7-task2_cat_sel_sgcrl_fair_s${seed}"
#   echo "python stable_crl.py --seed=${seed} --single_goal --categorical_select --env_id creative-7-task2 --use_pd --pd_duration 5 --architecture default --no-permute_start_boxes --entropy_cost 0.05 --entropy_cost_final 0.0 --num_timesteps 300000000 --exp_name=${exp_name}" >> "$TASK_FILE"
# done

# for seed in 1; do
# exp_name='creative-8-task2_cat_sel_sgcrl'
# echo "python stable_crl.py --seed=${seed} --single_goal --categorical_select --env_id creative-8-task2 --use_pd --pd_duration 5 --architecture default --exp_name=${exp_name}" >> "$TASK_FILE"
# # exp_name='creative-7-task2_cat_sel_sgcrl'
# # echo "python stable_crl.py --seed=${seed} --single_goal --categorical_select --env_id creative-7-task2 --use_pd --pd_duration 5 --architecture default --exp_name=${exp_name}" >> "$TASK_FILE"
# done






#
# ==============================================================================
# SECTION 2: THE BACKEND (DO NOT TOUCH)
# ==============================================================================

# --- CONFIGURATION FOR BATCHING ---
GPUS_PER_NODE=1
TASKS_PER_GPU=1  # <--- Change this to run more/fewer tasks per GPU
CHUNK_SIZE=$(( GPUS_PER_NODE * TASKS_PER_GPU ))
SERIAL=true

# Read all tasks
if [ ! -f "$TASK_FILE" ]; then
    echo "Error: No tasks generated."
    exit 1
fi

mapfile -t TASKS < "$TASK_FILE"
NUM_TASKS=${#TASKS[@]}

echo "Found $NUM_TASKS tasks. Splitting into batches of $CHUNK_SIZE..."

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/slurm_logs"

for (( i=0; i<NUM_TASKS; i+=CHUNK_SIZE )); do
    JOB_ID=$((i / CHUNK_SIZE))
    SLURM_SCRIPT="$SCRIPT_DIR/submit_batch_${JOB_ID}.slurm"

    # -- 1. Start writing the SLURM script --
    cat <<EOT > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=rebrac_mila_${JOB_ID}
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:${GPUS_PER_NODE}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=128G

source /home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench/.venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=\$SLURM_TMPDIR/wandb
export WANDB_CACHE_DIR=\$SLURM_TMPDIR/.cache/wandb
export WANDB_CONFIG_DIR=\$SLURM_TMPDIR/.config/wandb
export WANDB_DATA_DIR=\$SLURM_TMPDIR/.data/wandb

export MUJOCO_GL=egl

echo "Starting Batch $JOB_ID..."

EOT

    # -- 2. Inject commands using a Loop --
    # Iterate over each GPU
    for (( gpu=0; gpu<GPUS_PER_NODE; gpu++ )); do
        # Iterate over the "slots" on that GPU
        for (( slot=0; slot<TASKS_PER_GPU; slot++ )); do
            
            # Calculate the global index of the task we want
            # offset = (gpu * TASKS_PER_GPU) + slot
            offset=$(( (gpu * TASKS_PER_GPU) + slot ))
            task_index=$(( i + offset ))

            # Retrieve the task command
            cmd="${TASKS[$task_index]}"

            # Only add if the command is not empty (handles partial last batches)
            # if [ -n "$cmd" ]; then
            #     echo "CUDA_VISIBLE_DEVICES=$gpu $cmd &" >> "$SLURM_SCRIPT"
            # fi
            # if [ -n "$cmd" ]; then
            #     echo "(cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=$gpu $cmd) &" >> "$SLURM_SCRIPT"
            # fi
            if [ -n "$cmd" ]; then
                if [ "$SERIAL" = true ]; then
                    echo "(cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=$gpu $cmd)" >> "$SLURM_SCRIPT"
                else
                    echo "(cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=$gpu $cmd) &" >> "$SLURM_SCRIPT"
                fi
            fi
        done
    done

    # -- 3. Finish script --
    # echo "wait" >> "$SLURM_SCRIPT"
    if [ "$SERIAL" = false ] && [ "$TASKS_PER_GPU" -gt 1 ]; then
        echo "wait" >> "$SLURM_SCRIPT"
    fi

    # -- 4. Submit --
    echo "Submitting Batch $JOB_ID (Tasks starting at $i)..."
    sbatch "$SLURM_SCRIPT"
    
    # Optional: Delete the script after submission
    # rm "$SLURM_SCRIPT" 
done

echo "All jobs submitted."
