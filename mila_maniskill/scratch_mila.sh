#!/bin/bash

# ==============================================================================
# SECTION 1: USER CONFIGURATION (EXPERIMENT DEFINITION)
# ==============================================================================

# 1. Define where the temp file lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TASK_FILE="$SCRIPT_DIR/mila_tasks.tmp"
rm -f "$TASK_FILE" # Clear old runs

# 2. Get Project Root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. EXPERIMENT LOOPS
echo "Generating tasks..."

WANDB_PROJECT="dist-matching"
WANDB_ENTITY="doina-precup"

for seed in 0 1; do

  # # --------------------------------------------------------------------------
  # # 1. ManiSkill In-Room Push Drawer (CloseCabinetDrawer) - PPO + CRL
  # # --------------------------------------------------------------------------
  # echo "python ppo_contrastive.py \
  #     --env=maniskill_close_cabinet_drawer \
  #     --seed=${seed} \
  #     --num_steps=600000000 \
  #     --log_dir_path=maniskill_close_cabinet_drawer/ \
  #     --hidden_layer_sizes=\"256,256,256,256,256,256\" \
  #     --ppo_crl_steps_per_iter=128 \
  #     --ppo_ent_coef=0.05 \
  #     --ppo_actor_min_std=0.01 \
  #     --ppo_discount=0.99 \
  #     --ppo_clip_coef=0.2 \
  #     --ppo_crl_repr_tau=0.5 \
  #     --ppo_num_envs=512 \
  #     --maniskill_native_vec \
  #     --uniform_sampling \
  #     --wandb_project=${WANDB_PROJECT} \
  #     --wandb_entity=${WANDB_ENTITY} \
  #     --wandb_group=ppo_close_cabinet_drawer_crl \
  #     --render_video" >> "$TASK_FILE"
  #
  # # --------------------------------------------------------------------------
  # # 2. ManiSkill In-Room Push Drawer (CloseCabinetDrawer) - PPO + RND
  # # --------------------------------------------------------------------------
  # echo "python ppo_rnd.py \
  #     --env=maniskill_close_cabinet_drawer \
  #     --seed=${seed} \
  #     --num_steps=600000000 \
  #     --log_dir_path=maniskill_close_cabinet_drawer_rnd/ \
  #     --hidden_layer_sizes=\"256,256,256,256,256,256\" \
  #     --ppo_ent_coef=0.05 \
  #     --ppo_actor_min_std=0.01 \
  #     --ppo_discount=0.99 \
  #     --ppo_clip_coef=0.2 \
  #     --rnd_int_coef=1.0 \
  #     --rnd_ext_coef=1.0 \
  #     --rnd_int_discount=0.99 \
  #     --ppo_num_envs=512 \
  #     --maniskill_native_vec \
  #     --wandb_project=${WANDB_PROJECT} \
  #     --wandb_entity=${WANDB_ENTITY} \
  #     --wandb_group=ppo_close_cabinet_drawer_rnd \
  #     --render_video" >> "$TASK_FILE"

  # --------------------------------------------------------------------------
  # 3. ManiSkill-HAB Adjacent-Room Spawn -> Close Drawer - PPO + CRL
  # --------------------------------------------------------------------------
  echo "python ppo_contrastive.py \
      --env=maniskill_close_subtask_train \
      --seed=${seed} \
      --num_steps=70000000 \
      --log_dir_path=maniskill_close_subtask_train/ \
      --hidden_layer_sizes=\"256,256,256,256,256,256\" \
      --ppo_crl_steps_per_iter=256 \
      --ppo_ent_coef=0.05 \
      --ppo_actor_min_std=0.01 \
      --ppo_discount=0.99 \
      --ppo_clip_coef=0.2 \
      --ppo_crl_repr_tau=0.5 \
      --ppo_num_envs=64 \
      --ppo_checkpoint_interval=20 \
      --maniskill_native_vec \
      --uniform_sampling \
      --ppo_crl_add_extrinsic_reward \
      --wandb_project=${WANDB_PROJECT} \
      --wandb_entity=${WANDB_ENTITY} \
      --wandb_group=ppo_close_subtask_crl \
      --render_video \
      --video_every_steps=20000000" >> "$TASK_FILE"

  # --------------------------------------------------------------------------
  # 4. ManiSkill-HAB Adjacent-Room Spawn -> Close Drawer - PPO + RND
  # --------------------------------------------------------------------------
  echo "python ppo_rnd.py \
      --env=maniskill_close_subtask_train \
      --seed=${seed} \
      --num_steps=70000000 \
      --log_dir_path=maniskill_close_subtask_train_rnd/ \
      --hidden_layer_sizes=\"256,256,256,256,256,256\" \
      --ppo_ent_coef=0.05 \
      --ppo_actor_min_std=0.01 \
      --ppo_discount=0.99 \
      --ppo_clip_coef=0.2 \
      --rnd_int_coef=1.0 \
      --rnd_ext_coef=1.0 \
      --rnd_int_discount=0.99 \
      --ppo_num_envs=64 \
      --ppo_checkpoint_interval=20 \
      --maniskill_native_vec \
      --wandb_project=${WANDB_PROJECT} \
      --wandb_entity=${WANDB_ENTITY} \
      --wandb_group=ppo_close_subtask_rnd \
      --render_video \
      --video_every_steps=20000000" >> "$TASK_FILE"

  # --------------------------------------------------------------------------
  # 5. ManiSkill-HAB Adjacent-Room Spawn -> Open Drawer - PPO + CRL
  # --------------------------------------------------------------------------
  echo "python ppo_contrastive.py \
      --env=maniskill_open_subtask_train \
      --seed=${seed} \
      --num_steps=70000000 \
      --log_dir_path=maniskill_open_subtask_train/ \
      --hidden_layer_sizes=\"256,256,256,256,256,256\" \
      --ppo_crl_steps_per_iter=256 \
      --ppo_ent_coef=0.05 \
      --ppo_actor_min_std=1e-5 \
      --ppo_discount=0.99 \
      --ppo_clip_coef=0.2 \
      --ppo_crl_repr_tau=0.5 \
      --ppo_num_envs=64 \
      --ppo_checkpoint_interval=20 \
      --maniskill_native_vec \
      --uniform_sampling \
      --ppo_crl_add_extrinsic_reward \
      --wandb_project=${WANDB_PROJECT} \
      --wandb_entity=${WANDB_ENTITY} \
      --wandb_group=ppo_open_subtask_crl \
      --render_video \
      --video_every_steps=20000000" >> "$TASK_FILE"

  # --------------------------------------------------------------------------
  # 6. ManiSkill-HAB Adjacent-Room Spawn -> Open Drawer - PPO + RND
  # --------------------------------------------------------------------------
  echo "python ppo_rnd.py \
      --env=maniskill_open_subtask_train \
      --seed=${seed} \
      --num_steps=70000000 \
      --log_dir_path=maniskill_open_subtask_train_rnd/ \
      --hidden_layer_sizes=\"256,256,256,256,256,256\" \
      --ppo_ent_coef=0.05 \
      --ppo_actor_min_std=1e-5 \
      --ppo_discount=0.99 \
      --ppo_clip_coef=0.2 \
      --rnd_int_coef=1.0 \
      --rnd_ext_coef=1.0 \
      --rnd_int_discount=0.99 \
      --ppo_num_envs=64 \
      --ppo_checkpoint_interval=20 \
      --maniskill_native_vec \
      --wandb_project=${WANDB_PROJECT} \
      --wandb_entity=${WANDB_ENTITY} \
      --wandb_group=ppo_open_subtask_rnd \
      --render_video \
      --video_every_steps=20000000" >> "$TASK_FILE"

done

# ==============================================================================
# SECTION 2: THE BACKEND (SLURM BATCH GENERATION & SUBMISSION)
# ==============================================================================

# --- CONFIGURATION FOR BATCHING ---
GPUS_PER_NODE=1
TASKS_PER_GPU=1
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
#SBATCH --job-name=maniskill_mila_${JOB_ID}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=128G

source "$PROJECT_ROOT/mainskill_env/bin/activate"

export PATH="$PROJECT_ROOT/mainskill_env/bin:$PATH"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_ENTITY=${WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT}
export MS_ASSET_DIR=/network/scratch/m/mohammad-sami-nur.islam/maniskill_data
export MSHAB_TASK=set_table
export MSHAB_SPLIT=train
export MSHAB_OBJ=kitchen_counter

echo "Starting Batch $JOB_ID..."

EOT

    # -- 2. Inject commands using a Loop --
    for (( gpu=0; gpu<GPUS_PER_NODE; gpu++ )); do
        for (( slot=0; slot<TASKS_PER_GPU; slot++ )); do
            offset=$(( (gpu * TASKS_PER_GPU) + slot ))
            task_index=$(( i + offset ))
            cmd="${TASKS[$task_index]}"

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
    if [ "$SERIAL" = false ] && [ "$TASKS_PER_GPU" -gt 1 ]; then
        echo "wait" >> "$SLURM_SCRIPT"
    fi

    # -- 4. Submit --
    echo "Submitting Batch $JOB_ID (Tasks starting at $i)..."
    sbatch "$SLURM_SCRIPT"
done

echo "All jobs submitted."
