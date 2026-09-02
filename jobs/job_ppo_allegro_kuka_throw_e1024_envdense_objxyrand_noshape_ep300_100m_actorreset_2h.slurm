#!/bin/bash
#SBATCH --job-name=akt_envdense_objxy_ar
#SBATCH --output=slurm/ppo_allegro_kuka_throw_e1024_envdense_objxyrand_noshape_ep300_100m_actorreset_2h_%A_%a.log
#SBATCH --error=slurm/ppo_allegro_kuka_throw_e1024_envdense_objxyrand_noshape_ep300_100m_actorreset_2h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# PPO on NVIDIA AllegroKuka dense rew_buf (lift + distance). Sanity: can the
# 23-DoF throw succeed at all under the env's own shaping? 100M / 2h.
# Object xyz reset noise only; shape frozen. No palm, no NF.
#   sbatch jobs/job_ppo_allegro_kuka_throw_e1024_envdense_objxyrand_noshape_ep300_100m_actorreset_2h.slurm

unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

ENV=${ENV:-allegro_kuka_throw}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-100000000}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-10000000}
EP_LEN=${EP_LEN:-300}
ENT_COEF=${ENT_COEF:-0.05}
ENT_COEF_FINAL=${ENT_COEF_FINAL:-0.01}
MIN_STD=${MIN_STD:-1e-5}
DISCOUNT_FACTOR=${DISCOUNT_FACTOR:-0.99}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.2}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}

LOG_DIR="${LOG_DIR:-logs/ppo_allegro_kuka_throw_e1024_envdense_objxyrand_noshape_ep300_100m_actorreset_2h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} ppo_reward_mode=env_dense"
echo "num_envs=${PPO_NUM_ENVS} episode_length=${EP_LEN} rollout_length=${EP_LEN}"
echo "isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "PPO on NVIDIA rew_buf; crl_steps forced to 0"
echo "ppo_actor_min_std=${MIN_STD}"
echo "ppo_anneal_ent_coef=true ${ENT_COEF}->${ENT_COEF_FINAL}"
echo "ppo_eval_interval=0"
echo "init: object xyz noise ON; shape/joints/quat/force OFF"
echo "palm_goal=false  table_push=false  (throw packing, bucket goal)"
echo "actor_reset=true (ep_length_mean < 200; last-layer / BB rule)"
echo "steps_per_iter=$((PPO_NUM_ENVS * EP_LEN))"
echo "num_iterations=$((NUM_STEPS / (PPO_NUM_ENVS * EP_LEN)))"

nvidia-smi || true
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
  --noisaacgym_randomize_init \
  --isaacgym_randomize_object_xyz \
  --ppo_actor_reset_ep_len=200 \
  --ppo_rollout_length="${EP_LEN}" \
  --ppo_reward_mode=env_dense \
  --ppo_crl_steps_per_iter=0 \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_anneal_ent_coef=true \
  --ppo_ent_coef_final="${ENT_COEF_FINAL}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_eval_interval=0 \
  --ppo_video_interval=0 \
  --ppo_checkpoint_interval=50 \
  --ppo_checkpoint_replay_max="${MAX_REPLAY_SIZE}" \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
