#!/bin/bash
#SBATCH --job-name=ppo_rnd_maniskill_close_subtask_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# PPO+RND baseline for CloseSubtaskTrain-v0, sibling to this directory's
# close_subtask_train.slurm (PPO-on-phi.psi). Same env setup, same
# architecture/hyperparameters where reasonable; no CRL critic/replay, no
# --uniform_sampling, no q_params EMA (TAU) -- those are CRL-specific.
#
# Reward is sparse and DELIBERATELY narrower than mshab's own
# info['success'] (which also requires the arm back at rest, the robot
# static, and under the cumulative-force limit): it's 1 iff the drawer
# joint itself is closed, matching the evaluator's success/success_1000
# semantics (contrastive.utils.DrawerClosedSuccessObserver) rather than
# ManiSkill/mshab's broader native success signal. See
# env_utils.ManiskillVecEnv's `success_key='drawer_closed'` (passed
# automatically by ppo_rnd.py for this env_name -- no flag needed here),
# plus an RND intrinsic exploration bonus on top.
#
# --time/--mem/NUM_ENVS are the same estimates as close_subtask_train.slurm
# -- not re-measured for this architecture specifically. See that script's
# comments for the full history of GPU-memory-driven NUM_ENVS reductions
# (256 -> 64) for this task.

# See close_subtask_train.slurm's comment on why `conda init; source
# activate base; conda activate <env>` (open_cabinet_drawer.slurm's pattern)
# was tried and empirically failed on a real compute node -- sourcing
# conda.sh directly is used instead.
source /home/gliu2/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/gliu2/conda_envs/mshab_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/data/user_data/gliu2/conda_envs/mshab_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# Same torch-bundled-libcusolver-vs-system-CUDA-11 fix as
# open_cabinet_drawer.slurm/close_subtask_train.slurm -- see those scripts'
# comments for the full explanation.
TORCH_CUSOLVER_LIB=$(python -c "import os, nvidia.cusolver; print(os.path.join(os.path.dirname(nvidia.cusolver.__file__), 'lib'))" 2>/dev/null)
if [ -n "${TORCH_CUSOLVER_LIB}" ] && [ -d "${TORCH_CUSOLVER_LIB}" ]; then
  export LD_LIBRARY_PATH="${TORCH_CUSOLVER_LIB}:${LD_LIBRARY_PATH}"
fi
export CPATH=/data/user_data/gliu2/conda_envs/mshab_rl/include
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_ENTITY=grace_rl

# All ManiSkill/mshab scene assets (ycb, ReplicaCAD, ReplicaCADRearrange)
# and the mshab_rl conda env live under /data/user_data, NOT $HOME -- same
# as close_subtask_train.slurm.
export MS_ASSET_DIR=/data/user_data/gliu2/maniskill_data
export MSHAB_TASK=set_table
export MSHAB_SPLIT=train
export MSHAB_OBJ=kitchen_counter

REARRANGE_DIR="${MS_ASSET_DIR}/data/scene_datasets/replica_cad_dataset/rearrange"
TASK_PLAN_FP="${REARRANGE_DIR}/task_plans/${MSHAB_TASK}/close/${MSHAB_SPLIT}/${MSHAB_OBJ}.json"
SPAWN_DATA_FP="${REARRANGE_DIR}/spawn_data/${MSHAB_TASK}/close/${MSHAB_SPLIT}/spawn_data.pt"

# Idempotent asset download + task-plan/spawn-data existence check, flock-
# guarded so only one of this script's concurrent array tasks downloads.
# See close_subtask_train.slurm's comment for why `-y` is required here.
ASSET_LOCK=${MS_ASSET_DIR}/.mshab_replica_cad.lock
mkdir -p "$(dirname "${ASSET_LOCK}")"
(
  flock -x 200
  if [ ! -f "${TASK_PLAN_FP}" ] || [ ! -f "${SPAWN_DATA_FP}" ]; then
    python -m mani_skill.utils.download_asset ycb -y
    python -m mani_skill.utils.download_asset ReplicaCAD -y
    python -m mani_skill.utils.download_asset ReplicaCADRearrange -y
  fi
  if [ ! -f "${TASK_PLAN_FP}" ] || [ ! -f "${SPAWN_DATA_FP}" ]; then
    echo "ERROR: expected task plan / spawn data not found after download:"
    echo "  ${TASK_PLAN_FP}"
    echo "  ${SPAWN_DATA_FP}"
    echo "Available close-subtask task plan dirs under ${REARRANGE_DIR}/task_plans:"
    find "${REARRANGE_DIR}/task_plans" -maxdepth 2 -type d -name close 2>/dev/null
    exit 1
  fi
) 200>"${ASSET_LOCK}"

SEED=${SLURM_ARRAY_TASK_ID}
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_rnd_maniskill_close_subtask_train_adjacent_room/

ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2

# Same RND weighting/discount defaults as open_cabinet_drawer_rnd.slurm --
# see that script's comment.
RND_INT_COEF=1.0
RND_EXT_COEF=1.0
RND_INT_DISCOUNT=0.99

# Same GPU-memory-driven env count as close_subtask_train.slurm (dropped
# 512 -> 256 -> 64 across two real-run crashes; see that script's comments
# for the full history).
NUM_ENVS=64

python /home/gliu2/dist_matching/sgcrl/ppo_rnd.py \
    --env=maniskill_close_subtask_train \
    --seed=${SEED} \
    --num_steps=70000000 \
    --log_dir_path=${LOG_DIR} \
    --hidden_layer_sizes="256,256,256,256,256,256" \
    --ppo_ent_coef=${ENT_COEF} \
    --ppo_actor_min_std=${MIN_STD} \
    --ppo_discount=${DISCOUNT_FACTOR} \
    --ppo_clip_coef=${PPO_CLIP_COEF} \
    --rnd_int_coef=${RND_INT_COEF} \
    --rnd_ext_coef=${RND_EXT_COEF} \
    --rnd_int_discount=${RND_INT_DISCOUNT} \
    --ppo_num_envs=${NUM_ENVS} \
    --ppo_checkpoint_interval=20 \
    --maniskill_native_vec \
    --wandb_project=sgcrl-maniskill-close-subtask-train \
    --wandb_group=ppo_rnd_goal_fast \
    --render_video \
    --video_every_steps=20000000
