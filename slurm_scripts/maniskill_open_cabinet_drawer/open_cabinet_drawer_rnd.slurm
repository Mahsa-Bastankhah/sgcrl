#!/bin/bash
#SBATCH --job-name=ppo_rnd_maniskill_open_cabinet_drawer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=3-4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# PPO+RND baseline for OpenCabinetDrawer-v1, sibling to this directory's
# open_cabinet_drawer.slurm (PPO-on-phi.psi). Same env setup, same
# architecture/hyperparameters where reasonable; no CRL critic/replay, no
# --uniform_sampling, no q_params EMA (TAU) -- those are CRL-specific.
# Reward is sparse: 1 if ManiSkill's own success signal fires (which for
# this task coincides with the evaluator's success/success_1000 semantics --
# see contrastive.utils.SuccessObserver and env_utils.ManiskillVecEnv's
# `success_key` kwarg) plus an RND intrinsic exploration bonus. Same
# --time/--mem caveat as open_cabinet_drawer.slurm: estimates, not measured
# for this architecture/task specifically.

conda init
source activate base
conda activate /data/user_data/gliu2/conda_envs/contrastive_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/home/gliu2/miniconda/envs/contrastive_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# torch==2.8.0+cu128 (used internally by ManiSkill's IK solver,
# torch.linalg.solve in pd_ee_pose.py's compute_ik) needs its own bundled
# CUDA 12.x libcusolver, but jaxlib==0.4.7+cuda11.cudnn82 needs the
# *system* CUDA 11 install above -- see open_cabinet_drawer.slurm's comment
# for the full explanation. Put torch's own bundled (correct) libcusolver
# first.
TORCH_CUSOLVER_LIB=$(python -c "import os, nvidia.cusolver; print(os.path.join(os.path.dirname(nvidia.cusolver.__file__), 'lib'))" 2>/dev/null)
if [ -n "${TORCH_CUSOLVER_LIB}" ] && [ -d "${TORCH_CUSOLVER_LIB}" ]; then
  export LD_LIBRARY_PATH="${TORCH_CUSOLVER_LIB}:${LD_LIBRARY_PATH}"
fi
export CPATH=/home/gliu2/miniconda/envs/contrastive_rl/include
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
# Same tensorflow==2.8/reverb-vs-newer-protobuf conflict as
# open_cabinet_drawer.slurm.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_ENTITY=grace_rl

# OpenCabinetDrawer-v1 needs its PartNet-Mobility cabinet assets, cached
# under ~/.maniskill/data and shared across seeds/runs once fetched. Same
# flock-guarded idempotent warm-up as open_cabinet_drawer.slurm -- see that
# script's comment for why `-y` is deliberately NOT used here.
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
LOG_DIR=/home/gliu2/dist_matching/sgcrl/logs/ppo_rnd_maniskill_open_cabinet_drawer_1M/

ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2

# Intrinsic (RND) vs extrinsic (sparse success) advantage weighting, and the
# discount used for the intrinsic-reward GAE/return stream. 1.0/1.0 matches
# the collaborator's gym-based PPO+RND script's defaults; revisit if the
# intrinsic bonus dominates/vanishes relative to the sparse extrinsic signal
# once real `rnd/*` wandb metrics are observed.
RND_INT_COEF=1.0
RND_EXT_COEF=1.0
RND_INT_DISCOUNT=0.99

# 512 parallel envs: same native GPU-batched ManiSkill sim as
# open_cabinet_drawer.slurm -- see that script's comment.
NUM_ENVS=512

python /home/gliu2/dist_matching/sgcrl/ppo_rnd.py \
    --env=maniskill_open_cabinet_drawer \
    --seed=${SEED} \
    --num_steps=600000000 \
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
    --maniskill_native_vec \
    --wandb_project=sgcrl-maniskill-open-cabinet-drawer \
    --wandb_group=ppo_rnd_goal_fast \
    --render_video
