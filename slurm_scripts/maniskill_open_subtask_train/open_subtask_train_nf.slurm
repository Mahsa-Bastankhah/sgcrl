#!/bin/bash
#SBATCH --job-name=ppo_maniskill_open_subtask_train_nf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Sibling of ../maniskill_close_subtask_train/close_subtask_train.slurm:
# same scene, same "adjacent room" robot spawn, same Fetch robot,
# hyperparameters, and GPU/PhysX buffer constraints -- only the goal is
# reversed (drawer starts CLOSED, success means OPENING it -- see
# env_utils.ManiskillOpenSubtaskTrain / OpenSubtaskTrain-v0). See that
# script's comments for the full rationale behind --time/--mem/NUM_ENVS
# (not re-measured for this env specifically; reused as-is since the
# scene/episode-length/PhysX-buffer situation is identical).

# NOTE: --time/--mem are estimates, carried over unchanged from
# close_subtask_train.slurm -- revisit after the first real run's `sps`
# metric is observed in the learner log.

# mshab (https://github.com/arth-shukla/mshab) requires a specific `mshab`
# branch of ManiSkill3-beta, incompatible with the pip mani_skill==3.0.1
# release the contrastive_rl env uses for OpenCabinetDrawer-v1. It lives in
# its own conda env, cloned from contrastive_rl, under /data/user_data
# (compute-node-only mount with far more free space than $HOME) --
# see the one-time setup instructions in this folder's README/plan.
#
# See close_subtask_train.slurm's comment for why `conda init; source
# activate base; conda activate <env>` (open_cabinet_drawer.slurm's
# pattern) was tried and empirically failed on a real compute node --
# sourcing conda.sh directly is used instead.
source /home/gliu2/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/gliu2/conda_envs/mshab_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/data/user_data/gliu2/conda_envs/mshab_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# Same torch-bundled-libcusolver-vs-system-CUDA-11 fix as
# open_cabinet_drawer.slurm/close_subtask_train.slurm -- see those scripts'
# comments for the full explanation. Put torch's own bundled (correct)
# libcusolver first.
TORCH_CUSOLVER_LIB=$(python -c "import os, nvidia.cusolver; print(os.path.join(os.path.dirname(nvidia.cusolver.__file__), 'lib'))" 2>/dev/null)
if [ -n "${TORCH_CUSOLVER_LIB}" ] && [ -d "${TORCH_CUSOLVER_LIB}" ]; then
  export LD_LIBRARY_PATH="${TORCH_CUSOLVER_LIB}:${LD_LIBRARY_PATH}"
fi
export CPATH=/data/user_data/gliu2/conda_envs/mshab_rl/include
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
# Same tensorflow==2.8/reverb-vs-newer-protobuf conflict as
# open_cabinet_drawer.slurm.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_ENTITY=grace_rl

# All ManiSkill/mshab scene assets (ycb, ReplicaCAD, ReplicaCADRearrange)
# and the new conda env live under /data/user_data/gliu2, NOT $HOME --
# $HOME has only a few GB free, ReplicaCAD*/mshab assets are much larger
# than the PartNet-Mobility cache open_cabinet_drawer.slurm uses.
export MS_ASSET_DIR=/data/user_data/gliu2/maniskill_data
# "open"/"close" subtasks only occur in the "set_table" long-horizon HAB
# task in mshab's standard task data (opening/closing a drawer/fridge/
# cabinet before/after placing an object) -- same MSHAB_TASK as close.
export MSHAB_TASK=set_table
export MSHAB_SPLIT=train
# OpenSubtaskTrain-v0 with num_envs>1 requires every parallel env's subtask
# to share one articulation_type (mshab's own
# SequentialTaskEnv._merge_open_subtasks asserts this, mirroring close's
# _merge_close_subtasks) -- task_plans/.../all.json mixes "kitchen_counter"
# and "fridge" and would break that assertion under
# --maniskill_native_vec, so env_utils.py's _mshab_open_subtask_paths()
# uses the single-type <obj>.json instead, same as close.
export MSHAB_OBJ=kitchen_counter

REARRANGE_DIR="${MS_ASSET_DIR}/data/scene_datasets/replica_cad_dataset/rearrange"
TASK_PLAN_FP="${REARRANGE_DIR}/task_plans/${MSHAB_TASK}/open/${MSHAB_SPLIT}/${MSHAB_OBJ}.json"
SPAWN_DATA_FP="${REARRANGE_DIR}/spawn_data/${MSHAB_TASK}/open/${MSHAB_SPLIT}/spawn_data.pt"

# Idempotent asset download + task-plan/spawn-data existence check, flock-
# guarded so only one of this script's concurrent array tasks downloads.
# See close_subtask_train.slurm's comment for why `-y` is required here.
# NOTE: unlike close_subtask_train.slurm, this does NOT regenerate
# spawn_data.pt if missing -- the "adjacent room" open spawn_data.pt must
# be built ahead of time via build_open_spawn_data.slurm (which reuses
# close's already-validated spawn poses; see that script and
# scripts/build_open_spawn_data.py). If it's missing, fail loudly instead
# of silently falling back to mshab's own stock same-room spawn_data.pt.
ASSET_LOCK=${MS_ASSET_DIR}/.mshab_replica_cad.lock
mkdir -p "$(dirname "${ASSET_LOCK}")"
(
  flock -x 200
  if [ ! -f "${TASK_PLAN_FP}" ]; then
    python -m mani_skill.utils.download_asset ycb -y
    python -m mani_skill.utils.download_asset ReplicaCAD -y
    python -m mani_skill.utils.download_asset ReplicaCADRearrange -y
  fi
  if [ ! -f "${TASK_PLAN_FP}" ]; then
    echo "ERROR: expected task plan not found after download: ${TASK_PLAN_FP}"
    echo "Available open-subtask task plan dirs under ${REARRANGE_DIR}/task_plans:"
    find "${REARRANGE_DIR}/task_plans" -maxdepth 2 -type d -name open 2>/dev/null
    exit 1
  fi
  if [ ! -f "${SPAWN_DATA_FP}" ]; then
    echo "ERROR: expected (adjacent-room) open spawn data not found:"
    echo "  ${SPAWN_DATA_FP}"
    echo "Run build_open_spawn_data.slurm first (see this directory)."
    exit 1
  fi
) 200>"${ASSET_LOCK}"

SEED=${SLURM_ARRAY_TASK_ID}
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_maniskill_open_subtask_train_adjacent_room_nf/

# Density-estimator (NF) update steps per PPO iteration. Lower than this
# task's CRL steps (256) -- the RealNVP flow (nf_num_blocks coupling
# layers) is much more expensive per gradient step than CRL's single MLP
# encoder pass; 10 matches the collaborator's NF-default.slurm tuning and
# the open_cabinet_drawer_nf.slurm sibling script.
CRL_STEPS=10
ENT_COEF=0.05
MIN_STD=1e-5
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
# EMA decay for the "reward" copy of the NF params (mirrors this task's
# CRL TAU/ppo_crl_repr_tau, but for ppo_nf_reward_tau instead).
REWARD_TAU=0.5

# Same GPU-memory-driven env count as close_subtask_train.slurm (dropped
# 512 -> 256 -> 64 across two real-run crashes on the close side; see that
# script's comments for the full history -- the same ReplicaCAD scene /
# PhysX buffer constraints apply here unchanged).
NUM_ENVS=64

python /home/gliu2/dist_matching/sgcrl/ppo_contrastive.py \
    --env=maniskill_open_subtask_train \
    --seed=${SEED} \
    --num_steps=70000000 \
    --log_dir_path=${LOG_DIR} \
    --hidden_layer_sizes="256,256,256,256,256,256" \
    --ppo_repr_mode=nf \
    --nf_rep_size=64 \
    --nf_num_blocks=6 \
    --nf_coupling_width=192 \
    --nf_sa_hidden=192 \
    --nf_sa_num_layers=3 \
    --nf_grad_clip=1.0 \
    --nf_noise_std=0.05 \
    --nf_goal_std_min=0.02 \
    --nf_goal_enc_size=0 \
    --nf_mix_task_goal_stats \
    --ppo_crl_steps_per_iter=${CRL_STEPS} \
    --ppo_ent_coef=${ENT_COEF} \
    --ppo_actor_min_std=${MIN_STD} \
    --ppo_discount=${DISCOUNT_FACTOR} \
    --ppo_clip_coef=${PPO_CLIP_COEF} \
    --ppo_nf_reward_tau=${REWARD_TAU} \
    --ppo_num_envs=${NUM_ENVS} \
    --ppo_checkpoint_interval=20 \
    --maniskill_native_vec \
    --ppo_crl_add_extrinsic_reward \
    --wandb_project=sgcrl-maniskill-open-subtask-train \
    --wandb_group=ppo_nf_goal_fast_mixtask \
    --render_video \
    --video_every_steps=20000000
