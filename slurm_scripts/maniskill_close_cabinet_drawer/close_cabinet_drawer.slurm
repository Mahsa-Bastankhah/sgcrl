#!/bin/bash
#SBATCH --job-name=ppo_maniskill_close_cabinet_drawer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Sibling of maniskill_open_cabinet_drawer/open_cabinet_drawer.slurm for
# CloseCabinetDrawer-v1 (env_utils.py's own close-direction registration --
# ManiSkill3 ships OpenCabinetDrawer-v1/OpenCabinetDoor-v1 but no native
# "close" task, so this repo registers one as a subclass of
# OpenCabinetDrawerEnv: same Fetch-vs-PartNet-Mobility-cabinet asset/robot
# setup, drawers just start fully open instead of fully closed, and success
# means pushing back down to (near) closed instead of pulling open).
#
# NOTE: --time/--mem/architecture/hyperparameters copied as-is from the open
# task's script -- same underlying sim (same asset loading, same Fetch
# mobile-base robot, same 100-step episodes), so the same estimates/starting
# guess apply. Revisit after the first real run's `sps` metric is observed.

conda init
source activate base
conda activate /data/user_data/gliu2/conda_envs/contrastive_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/home/gliu2/miniconda/envs/contrastive_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# torch==2.8.0+cu128 (used internally by ManiSkill's IK solver,
# torch.linalg.solve in pd_ee_pose.py's compute_ik) needs its own bundled
# CUDA 12.x libcusolver, but jaxlib==0.4.7+cuda11.cudnn82 needs the
# *system* CUDA 11 install above -- see
# maniskill_open_cabinet_drawer/open_cabinet_drawer.slurm's comment for the
# full explanation. Put torch's own bundled (correct) libcusolver first.
TORCH_CUSOLVER_LIB=$(python -c "import os, nvidia.cusolver; print(os.path.join(os.path.dirname(nvidia.cusolver.__file__), 'lib'))" 2>/dev/null)
if [ -n "${TORCH_CUSOLVER_LIB}" ] && [ -d "${TORCH_CUSOLVER_LIB}" ]; then
  export LD_LIBRARY_PATH="${TORCH_CUSOLVER_LIB}:${LD_LIBRARY_PATH}"
fi
export CPATH=/home/gliu2/miniconda/envs/contrastive_rl/include
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
# Installing mani_skill/torch/gymnasium into this conda env pulled in a
# much newer protobuf than the pinned tensorflow==2.8/reverb stack
# expects, which otherwise breaks `import contrastive` with
# `TypeError: Descriptors cannot be created directly.`
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_ENTITY=grace_rl

# CloseCabinetDrawer-v1 uses the exact same PartNet-Mobility cabinet assets
# as OpenCabinetDrawer-v1 (it's registered as a subclass with the same
# asset_download_ids=['partnet_mobility_cabinet']), cached under
# ~/.maniskill/data and shared across seeds/runs/tasks once fetched. Reuse
# the SAME lock file as the open task's script so the two task's array jobs
# also serialize against each other, not just against themselves. Do NOT
# shell out to `mani_skill.utils.download_asset ... -y` here -- see that
# script's comment for why.
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
ASSET_LOCK=/home/gliu2/.maniskill/data/.partnet_mobility_cabinet.lock
mkdir -p "$(dirname "${ASSET_LOCK}")"
(
  flock -x 200
  python -c "
import sys
sys.path.insert(0, '/home/gliu2/dist_matching/sgcrl')
import gymnasium as gym
import env_utils  # noqa: F401  registers 'CloseCabinetDrawer-v1'
gym.make('CloseCabinetDrawer-v1', obs_mode='state_dict', control_mode='pd_ee_delta_pos', num_envs=1).close()
"
) 200>"${ASSET_LOCK}"

SEED=${SLURM_ARRAY_TASK_ID}
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_maniskill_close_cabinet_drawer_1M/

CRL_STEPS=128
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
# Slow "reward" copy of q_params (target-network-style EMA): the CRL loss
# still trains the live/online q_params every step, but the PPO reward is
# read from this tau-averaged copy instead. See ppo_learner.py's
# ppo_crl_repr_tau / q_params_reward.
TAU=0.5

# 512 parallel envs: only tractable via the native GPU-batched ManiSkill
# sim (one SAPIEN scene simulating all copies via the PhysX GPU backend).
# The default vec-env path spins up `ppo_num_envs` independent single-env
# SAPIEN scenes stepped one at a time in a Python loop -- fine at the
# default of 8, but would be catastrophically slow/memory-heavy at 512.
# If this OOMs on the allocated GPU, CloseCabinetDrawerEnv's own (inherited
# from OpenCabinetDrawerEnv) `_default_sim_config`
# (max_rigid_contact_count/max_rigid_patch_count) may need to be raised for
# this many parallel articulated-cabinet+Fetch scenes -- not currently
# plumbed through from this repo's env_utils.py.
NUM_ENVS=512

python /home/gliu2/dist_matching/sgcrl/ppo_contrastive.py \
    --env=maniskill_close_cabinet_drawer \
    --seed=${SEED} \
    --num_steps=600000000 \
    --log_dir_path=${LOG_DIR} \
    --hidden_layer_sizes="256,256,256,256,256,256" \
    --ppo_crl_steps_per_iter=${CRL_STEPS} \
    --ppo_ent_coef=${ENT_COEF} \
    --ppo_actor_min_std=${MIN_STD} \
    --ppo_discount=${DISCOUNT_FACTOR} \
    --ppo_clip_coef=${PPO_CLIP_COEF} \
    --ppo_crl_repr_tau=${TAU} \
    --ppo_num_envs=${NUM_ENVS} \
    --maniskill_native_vec \
    --uniform_sampling \
    --wandb_project=sgcrl-maniskill-close-cabinet-drawer \
    --wandb_group=ppo_goal_fast \
    --render_video
