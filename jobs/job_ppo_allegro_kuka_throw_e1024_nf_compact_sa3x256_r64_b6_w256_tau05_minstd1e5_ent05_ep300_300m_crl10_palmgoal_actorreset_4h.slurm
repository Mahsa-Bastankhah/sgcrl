#!/bin/bash
#SBATCH --job-name=akt_nfc_palm_ar_e05
#SBATCH --output=slurm/ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_actorreset_4h_%A_%a.log
#SBATCH --error=slurm/ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_actorreset_4h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# PPO + NF compact on AllegroKukaThrow, 4h (300M).
# Palm+AR (rand), entropy coef fixed at 0.05 (no anneal). Reward = log p(g|s,a).
#   sbatch jobs/job_ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_actorreset_4h.slurm

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
NUM_STEPS=${NUM_STEPS:-300000000}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-10000000}
EP_LEN=${EP_LEN:-300}
CRL_STEPS=${CRL_STEPS:-10}
ENT_COEF=${ENT_COEF:-0.05}
MIN_STD=${MIN_STD:-1e-5}
DISCOUNT_FACTOR=${DISCOUNT_FACTOR:-0.99}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.2}
REWARD_TAU=${REWARD_TAU:-0.5}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}

NF_REP_SIZE=${NF_REP_SIZE:-64}
NF_NUM_BLOCKS=${NF_NUM_BLOCKS:-6}
NF_COUPLING_WIDTH=${NF_COUPLING_WIDTH:-256}
NF_SA_HIDDEN=${NF_SA_HIDDEN:-256}
NF_SA_NUM_LAYERS=${NF_SA_NUM_LAYERS:-3}

LOG_DIR="${LOG_DIR:-logs/ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_actorreset_4h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=nf reward_tau=${REWARD_TAU}"
echo "num_envs=${PPO_NUM_ENVS} episode_length=${EP_LEN} rollout_length=${EP_LEN} crl_steps=${CRL_STEPS}"
echo "isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "ppo_use_external_reward=false (NF reward only)"
echo "ppo_actor_min_std=${MIN_STD}"
echo "ppo_anneal_ent_coef=false  ppo_ent_coef=${ENT_COEF} (fixed)"
echo "ppo_actor_reset_ep_len=200 (not 0.8T=240)"
echo "ppo_eval_interval=0"
echo "isaacgym_palm_goal=true  state=q,qd,palm,obj (52)  goal=[slide palm, obj in bucket] (6)"
echo "isaacgym_fixed_target_xyz=0.37,0.08,0.312  isaacgym_palm_goal_xyz=0.17,0.08,0.57"
echo "nf_state_only=false  (forward log p(g|s,a) / r(s,a); default)"
echo "nf_train_backward=false  (no sidecar backward NF)"
echo "init randomization=ON; bucket/palm goal FIXED"
echo "actor_reset=true (ep_length_mean < 200; last-layer / BB rule)"
echo "steps_per_iter=$((PPO_NUM_ENVS * EP_LEN))"
echo "num_iterations=$((NUM_STEPS / (PPO_NUM_ENVS * EP_LEN)))"
echo "nf_compact: sa${NF_SA_NUM_LAYERS}x${NF_SA_HIDDEN} r${NF_REP_SIZE} b${NF_NUM_BLOCKS} w${NF_COUPLING_WIDTH}  (not compact-small)"

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
  --isaacgym_palm_goal=true \
  --isaacgym_fixed_target_xyz=0.37,0.08,0.312 \
  --isaacgym_palm_goal_xyz=0.17,0.08,0.57 \
  --ppo_rollout_length="${EP_LEN}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_anneal_ent_coef=false \
  --ppo_actor_reset_ep_len=200 \
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
  --nf_train_backward=false \
  --nf_rep_size="${NF_REP_SIZE}" \
  --nf_num_blocks="${NF_NUM_BLOCKS}" \
  --nf_coupling_width="${NF_COUPLING_WIDTH}" \
  --nf_sa_hidden="${NF_SA_HIDDEN}" \
  --nf_sa_num_layers="${NF_SA_NUM_LAYERS}" \
  --nf_grad_clip=1.0 \
  --nf_noise_std=0.05 \
  --nf_goal_std_min=0.02 \
  --nf_goal_enc_size=0 \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
