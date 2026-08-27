#!/bin/bash
#SBATCH --job-name=ppo_maniskill_open_cabinet_drawer
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

# NOTE: --time/--mem are estimates, not measured for the 6x256 ResidualMLP
# architecture used here, or for OpenCabinetDrawer specifically. This task
# has 2x the episode length of PushCube/PickCube (100 vs 50 steps) and
# heavier physics (mobile-base Fetch robot + an articulated, procedurally
# sampled PartNet-Mobility cabinet reconfigured every episode), so both
# knobs are bumped up from the tabletop scripts as a starting guess.
# Revisit after the first real run's `sps` metric is observed in the
# learner log.

conda init
source activate base
conda activate /data/user_data/gliu2/conda_envs/contrastive_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/home/gliu2/miniconda/envs/contrastive_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# torch==2.8.0+cu128 (used internally by ManiSkill's IK solver,
# torch.linalg.solve in pd_ee_pose.py's compute_ik) needs its own bundled
# CUDA 12.x libcusolver, but jaxlib==0.4.7+cuda11.cudnn82 needs the
# *system* CUDA 11 install above (that's what /usr/local/cuda-11/lib64 is
# for) -- and both happen to ship a same-named libcusolver.so.11, so
# whichever comes first on LD_LIBRARY_PATH wins for BOTH frameworks. This
# was never exercised before --maniskill_native_vec (CPU-backed sim keeps
# ManiSkill's IK tensors on CPU, so torch's CUDA linalg lib is never
# dlopen'd at all), so the mismatch sat latent: the old CUDA-11-native
# libcusolver.so.11 quietly "won" the search and nothing needed the newer
# symbol it's missing (cusolverDnXsyevBatched_bufferSize) until GPU
# tensors from the native-vec sim started hitting torch's IK solve.
# Put torch's own bundled (correct) libcusolver first; leave
# cuDNN/cuBLAS/etc. from the system CUDA 11 install alone so jaxlib's own
# dependency stays satisfied.
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

# OpenCabinetDrawer-v1 needs its PartNet-Mobility cabinet assets, cached
# under ~/.maniskill/data and shared across seeds/runs once fetched.  Do
# NOT shell out to `mani_skill.utils.download_asset ... -y` here -- with
# -y it unconditionally deletes+re-downloads every asset in the group on
# every invocation, which would race across this script's 5 concurrent
# array tasks. Instead set MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 so ManiSkill's
# own gym.make() auto-downloads (idempotently, one asset at a time, only
# what's actually missing) the first time the env is built, and flock a
# one-time warm-up `gym.make(...).close()` so only one of the 5 array
# tasks does that first-time download while the others wait on the lock.
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
ASSET_LOCK=/home/gliu2/.maniskill/data/.partnet_mobility_cabinet.lock
mkdir -p "$(dirname "${ASSET_LOCK}")"
(
  flock -x 200
  python -c "
import gymnasium as gym
import mani_skill.envs
gym.make('OpenCabinetDrawer-v1', obs_mode='state_dict', control_mode='pd_ee_delta_pos', num_envs=1).close()
"
) 200>"${ASSET_LOCK}"

SEED=${SLURM_ARRAY_TASK_ID}
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_maniskill_open_cabinet_drawer_1M/

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
# If this OOMs on the allocated GPU, OpenCabinetDrawerEnv's own
# `_default_sim_config` (max_rigid_contact_count/max_rigid_patch_count)
# may need to be raised for this many parallel articulated-cabinet+Fetch
# scenes -- not currently plumbed through from this repo's env_utils.py.
NUM_ENVS=512

python /home/gliu2/dist_matching/sgcrl/ppo_contrastive.py \
    --env=maniskill_open_cabinet_drawer \
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
    --wandb_project=sgcrl-maniskill-open-cabinet-drawer \
    --wandb_group=ppo_goal_fast \
    --render_video
