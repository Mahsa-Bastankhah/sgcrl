#!/bin/bash
#SBATCH --job-name=ppo_maniskill_close_subtask_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=12-16
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err



# NOTE: --time/--mem are estimates, not measured. CloseSubtaskTrain-v0 has
# 2x OpenCabinetDrawer-v1's episode length (200 vs 100 steps) plus a
# ReplicaCAD-Rearrange *scene* (furnished room, not a single procedurally
# sampled cabinet) reconfigured every episode, so both knobs are bumped up
# further from the open_cabinet_drawer.slurm starting guess. Revisit after
# the first real run's `sps` metric is observed in the learner log.

# mshab (https://github.com/arth-shukla/mshab) requires a specific `mshab`
# branch of ManiSkill3-beta, incompatible with the pip mani_skill==3.0.1
# release the contrastive_rl env uses for OpenCabinetDrawer-v1. It lives in
# its own conda env, cloned from contrastive_rl, under /data/user_data
# (compute-node-only mount with far more free space than $HOME) --
# see the one-time setup instructions in this folder's README/plan.
#
# NOTE: open_cabinet_drawer.slurm's `conda init; source activate base;
# conda activate <env>` pattern was tried here first and empirically failed
# on a real compute node (`source activate base` -> "activate: No such file
# or directory", then `conda activate` itself errored with "Run 'conda init'
# before 'conda activate'") -- bash has no `set -e` so the script kept
# going with whatever env was already default-active (silently NOT
# mshab_rl), which only surfaced later as a confusing `ModuleNotFoundError:
# No module named 'mshab'` deep in env_utils.py. Sourcing conda.sh directly
# is the standard, reliably-idempotent activation hook and is used instead.
source /home/gliu2/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/gliu2/conda_envs/mshab_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/data/user_data/gliu2/conda_envs/mshab_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# Same torch-bundled-libcusolver-vs-system-CUDA-11 fix as
# open_cabinet_drawer.slurm -- see that script's comment for the full
# explanation. Put torch's own bundled (correct) libcusolver first.
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
# "close" subtasks only occur in the "set_table" long-horizon HAB task in
# mshab's standard task data (closing a drawer/fridge/cabinet after placing
# an object) -- verified against the downloaded asset listing (tidy_house/
# prepare_groceries have no close/ subdir at all under task_plans/).
export MSHAB_TASK=set_table
export MSHAB_SPLIT=train
# CloseSubtaskTrain-v0 with num_envs>1 requires every parallel env's subtask
# to share one articulation_type (mshab's own
# SequentialTaskEnv._merge_close_subtasks asserts this) -- task_plans/.../
# all.json mixes "kitchen_counter" and "fridge" and breaks that assertion
# under --maniskill_native_vec, so env_utils.py's
# _mshab_close_subtask_paths() uses the single-type <obj>.json instead.
# kitchen_counter chosen as the closer analog to OpenCabinetDrawer-v1's
# cabinet drawers; verified both files exist under train/ and val/.
export MSHAB_OBJ=kitchen_counter

REARRANGE_DIR="${MS_ASSET_DIR}/data/scene_datasets/replica_cad_dataset/rearrange"
TASK_PLAN_FP="${REARRANGE_DIR}/task_plans/${MSHAB_TASK}/close/${MSHAB_SPLIT}/${MSHAB_OBJ}.json"
SPAWN_DATA_FP="${REARRANGE_DIR}/spawn_data/${MSHAB_TASK}/close/${MSHAB_SPLIT}/spawn_data.pt"

# Idempotent asset download + task-plan/spawn-data existence check, flock-
# guarded so only one of this script's 5 concurrent array tasks downloads.
# `-y` is required here (without it, or MS_SKIP_ASSET_DOWNLOAD_PROMPT=1,
# download_asset falls back to Python's interactive `input()` for "output
# dir does not exist, Create?", which has no stdin to read under sbatch and
# would hang the job forever) -- safe because this block only runs when the
# target files are missing in the first place (nothing downloaded yet), so
# `-y`'s "already exists, Remove+redownload?"=yes behavior never triggers
# on an already-complete download (unlike PartNet-Mobility's downloader,
# where OpenCabinetDrawer-v1's script avoids `-y` for exactly that reason --
# see that script's comment).
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
# NOTE: was .../ppo_maniskill_close_subtask_train_1M/ -- that dir's
# checkpoints predate the adjacent-room spawn + 400-step episode changes
# (mshab/envs/close.py, gen_spawn_positions.py) and were already past
# --num_steps=16000000 (18.8M/18.8M/15.2M for seeds 0/1/2), so runs against
# that dir auto-resumed and immediately exited with zero new training steps.
# New log dir so this starts a fresh run under the new env dynamics.
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_maniskill_close_subtask_train_adjacent_room/

CRL_STEPS=256
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
# Slow "reward" copy of q_params (target-network-style EMA), same as
# open_cabinet_drawer.slurm -- see ppo_learner.py's ppo_crl_repr_tau.
TAU=0.5

# NOTE: originally 512 (matching open_cabinet_drawer.slurm), but a real run
# (job 9136398, all 3 array tasks) crashed with CUDA_ERROR_ILLEGAL_ADDRESS
# inside PhysX GPU contact kernels at iteration ~30-35 -- mshab's
# SequentialTaskEnv._default_sim_config hardcodes GPU contact/patch buffer
# sizes with no num_envs scaling, and mshab's own quickstart only validates
# up to num_envs=252 (63 RCAD scenes x 4/scene) for this much heavier
# per-env scene (~96 actors, vs. OpenCabinetDrawer-v1's ~2). Dropped to 256
# to stay close to mshab's own tested scale, on top of env_utils.py's
# doubled GPU buffer sizes (see _maniskill_extra_env_kwargs) as a second
# safety margin. num_envs need not divide the number of distinct
# ReplicaCAD scenes in the task plan --
# require_build_configs_repeated_equally_across_envs=False is passed by
# env_utils.py's ManiskillCloseSubtaskTrain/ManiskillVecEnv construction.
#
# Dropped further, 256 -> 32, after a second real run (job 9143247) crashed
# at iteration 84 with a DIFFERENT, non-configurable overflow: PhysX's fixed
# GPU collision_stack_size (4194304 bytes -- confirmed unexposed by both
# mani_skill==3.0.0b18's GPUMemoryConfig dataclass and this env's pinned
# sapien==3.0.0b1's compiled physx.set_gpu_memory_config binding, so unlike
# the other four buffers above it truly cannot be raised via kwarg; see
# env_utils.py's matching comment). The crash reported needing >= 67263240
# bytes at 256 envs -- if collision-stack demand scales roughly with total
# actor count (256 envs x ~96 actors/env), 4194304/67263240 * 256 ~= 16
# envs would just barely fit *that one observed peak*, with no headroom for
# worse contact spikes later in training. 32 (2x that estimate) trades some
# of that margin back for throughput -- unverified, next crash (if any)
# should narrow this further.
NUM_ENVS=64

python /home/gliu2/dist_matching/sgcrl/ppo_contrastive.py \
    --env=maniskill_close_subtask_train \
    --seed=${SEED} \
    --num_steps=70000000 \
    --log_dir_path=${LOG_DIR} \
    --hidden_layer_sizes="256,256,256,256,256,256" \
    --ppo_crl_steps_per_iter=${CRL_STEPS} \
    --ppo_ent_coef=${ENT_COEF} \
    --ppo_actor_min_std=${MIN_STD} \
    --ppo_discount=${DISCOUNT_FACTOR} \
    --ppo_clip_coef=${PPO_CLIP_COEF} \
    --ppo_crl_repr_tau=${TAU} \
    --ppo_num_envs=${NUM_ENVS} \
    --ppo_checkpoint_interval=20 \
    --maniskill_native_vec \
    --uniform_sampling \
    --wandb_project=sgcrl-maniskill-close-subtask-train \
    --wandb_group=ppo_goal_fast \
    --render_video \
    --video_every_steps=20000000
