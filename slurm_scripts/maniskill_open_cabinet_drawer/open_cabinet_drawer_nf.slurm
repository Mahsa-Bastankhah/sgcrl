#!/bin/bash
#SBATCH --job-name=ppo_maniskill_open_cabinet_drawer_nf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# NOTE: --time/--mem are carried over from the CRL version of this job
# (open_cabinet_drawer.slurm) -- not separately measured for the NF
# (RealNVP) density estimator. A 6-block coupling flow forward+backward
# pass is more expensive per step than CRL's single MLP encoder pass, but
# ppo_crl_steps_per_iter is correspondingly much lower here (10 vs 128),
# so overall wall-clock is not expected to be worse. Revisit after the
# first real run's `sps` metric is observed.

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
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_maniskill_open_cabinet_drawer_nf_mixtask_1M/

# Density-estimator (NF) update steps per PPO iteration. Lower than the
# CRL script's 128 -- the RealNVP flow (nf_num_blocks coupling layers) is
# much more expensive per gradient step than CRL's single MLP encoder
# pass; 10 matches the collaborator's NF-default.slurm tuning.
CRL_STEPS=10
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
# EMA decay for the "reward" copy of the NF params (mirrors the CRL
# script's TAU/ppo_crl_repr_tau, but for ppo_nf_reward_tau instead).
REWARD_TAU=0.5

# 512 parallel envs: only tractable via the native GPU-batched ManiSkill
# sim (one SAPIEN scene simulating all copies via the PhysX GPU backend).
NUM_ENVS=512

python /home/gliu2/dist_matching/sgcrl/ppo_contrastive.py \
    --env=maniskill_open_cabinet_drawer \
    --seed=${SEED} \
    --num_steps=600000000 \
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
    --maniskill_native_vec \
    --wandb_project=sgcrl-maniskill-open-cabinet-drawer \
    --wandb_group=ppo_nf_goal_fast_mixtask \
    --render_video
