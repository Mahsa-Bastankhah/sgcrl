#!/bin/bash
#SBATCH --job-name=akt_nf
#SBATCH --output=slurm/ppo_allegro_kuka_throw_e1024_nf_%A_%a.log
#SBATCH --error=slurm/ppo_allegro_kuka_throw_e1024_nf_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-0

# PPO + Normalizing-Flow density on NVIDIA Isaac Gym AllegroKukaThrow.
#   - Native-batched Isaac Gym vec env (E=1024, GPU PhysX) driven by
#     contrastive/ppo_learner_isaacgym.py.
#   - PhysX sim is created BEFORE JAX CUDA init (Preview 4 GPU kernels fail
#     if JAX grabs the device first).
#   - Goal for the NF = object absolute xyz. Bucket/target is a FIXED constant
#     of the env (never in the obs). Compact obs = [state(49) | goal(3)].
#   - eval/video are disabled for v1 (single-env dm_env eval can't open a second
#     Isaac Gym sim in the same process; batched eval is a follow-up).
#
# Smoke test:   sbatch jobs/job_ppo_allegro_kuka_throw_nf.slurm     (small NUM_STEPS)
# Full run:     NUM_STEPS=300000000 sbatch jobs/job_ppo_allegro_kuka_throw_nf.slurm

unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

# Isaac Gym needs the conda env's libpython3.8.so.1.0 on the loader path.
# Driver libcuda.so comes from the node; CUDA runtime is pip-bundled by
# torch (cu121) and jax[cuda12_pip], so we do NOT load a system cudatoolkit.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2

# Share one GPU between PhysX (Isaac Gym), torch, and JAX: do NOT let JAX
# preallocate the whole card, and cap its fraction so PhysX has room for the
# E=1024 dexterous-hand sim.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

ENV=${ENV:-allegro_kuka_throw}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-2000000}          # smoke default; override for full run
PPO_NUM_ENVS=${PPO_NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-10000000}
EP_LEN=${EP_LEN:-300}
CRL_STEPS=${CRL_STEPS:-10}
ENT_COEF=${ENT_COEF:-0.005}
MIN_STD=${MIN_STD:-1e-5}
DISCOUNT_FACTOR=${DISCOUNT_FACTOR:-0.99}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.2}
REWARD_TAU=${REWARD_TAU:-0.5}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}

LOG_DIR="${LOG_DIR:-logs/ppo_allegro_kuka_throw_e1024_nf/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=nf reward_tau=${REWARD_TAU}"
echo "num_envs=${PPO_NUM_ENVS} episode_length=${EP_LEN} rollout_length=${EP_LEN} crl_steps=${CRL_STEPS}"
echo "isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "steps_per_iter=$((PPO_NUM_ENVS * EP_LEN))"
echo "num_iterations=$((NUM_STEPS / (PPO_NUM_ENVS * EP_LEN)))"

nvidia-smi || true
# Sanity: Isaac Gym import order (isaacgym before torch) + JAX GPU backend.
python -u -c "
import isaacgym  # must precede torch
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import sgcrl_jax_acme_compat, jax
print('JAX backend:', jax.default_backend(), 'devices:', jax.devices())
"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --isaacgym_episode_length="${EP_LEN}" \
  --isaacgym_pipeline="${ISAACGYM_PIPELINE}" \
  --ppo_rollout_length="${EP_LEN}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_eval_interval=0 \
  --ppo_video_interval=0 \
  --ppo_checkpoint_interval=100 \
  --ppo_checkpoint_replay_max="${MAX_REPLAY_SIZE}" \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau="${REWARD_TAU}" \
  --nf_rep_size=64 \
  --nf_num_blocks=6 \
  --nf_coupling_width=256 \
  --nf_sa_hidden=256 \
  --nf_sa_num_layers=3 \
  --nf_grad_clip=1.0 \
  --nf_noise_std=0.05 \
  --nf_goal_std_min=0.02 \
  --nf_goal_enc_size=0 \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
