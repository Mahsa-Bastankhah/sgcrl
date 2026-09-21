#!/bin/bash
#SBATCH --job-name=akt_nf_center_ltable15_hover09x05_curl10
#SBATCH --output=slurm/ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_largetable15_tablespawn_hover09x05_curl10_rand005_spawnx017y004_noshape_mixtaskg_mix50_center_extrew1_zmax07_4h_%A_%a.log
#SBATCH --error=slurm/ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_largetable15_tablespawn_hover09x05_curl10_rand005_spawnx017y004_noshape_mixtaskg_mix50_center_extrew1_zmax07_4h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# Same learning configuration as center run 3809312. Environment changes:
# 1.5 x 1.5 m table shifted away from the robot; cube at (0.17,0.04,0.555)
# with correlated ±5 mm xy randomization (hand follows each cube). Close
# non-contact hover: palm standoff +0.09 m y / +0.05 m z, fingers at 0.10 curl
# so the extended paddle sits ~3 cm off the cube (contact very likely once the
# policy moves) without striking it. Goal is object xyz only.

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
ENT_COEF_FINAL=${ENT_COEF_FINAL:-0.01}
MIN_STD=${MIN_STD:-1e-5}
DISCOUNT_FACTOR=${DISCOUNT_FACTOR:-0.99}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.2}
REWARD_TAU=${REWARD_TAU:-0.5}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}
EXT_REW_SCALE=${EXT_REW_SCALE:-1}
RESET_Z_ABOVE=${RESET_Z_ABOVE:-0.7}
SPAWN_DY=${SPAWN_DY:-0.09}
SPAWN_ABOVE=${SPAWN_ABOVE:-0.05}
FINGER_CURL=${FINGER_CURL:-0.10}
CORRELATED_XY=${CORRELATED_XY:-0.005}

NF_REP_SIZE=${NF_REP_SIZE:-64}
NF_NUM_BLOCKS=${NF_NUM_BLOCKS:-6}
NF_COUPLING_WIDTH=${NF_COUPLING_WIDTH:-192}
NF_SA_HIDDEN=${NF_SA_HIDDEN:-192}
NF_SA_NUM_LAYERS=${NF_SA_NUM_LAYERS:-3}

LOG_DIR="${LOG_DIR:-logs/ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_largetable15_tablespawn_hover09x05_curl10_rand005_spawnx017y004_noshape_mixtaskg_mix50_center_extrew1_zmax07_4h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=nf reward_tau=${REWARD_TAU}"
echo "num_envs=${PPO_NUM_ENVS} episode_length=${EP_LEN} rollout_length=${EP_LEN} crl_steps=${CRL_STEPS}"
echo "isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "ppo_use_external_reward=true ppo_external_reward_scale=${EXT_REW_SCALE} (AFTER return-norm; env.success())"
echo "ppo_actor_min_std=${MIN_STD}"
echo "ppo_anneal_ent_coef=true ${ENT_COEF}->${ENT_COEF_FINAL}"
echo "ppo_eval_interval=10"
echo "table=1.5x1.5m bounds x=[-0.75,0.75] y=[-1.30,0.20], robot-facing edge unchanged"
echo "init: cube=(0.17,0.04,0.555) + correlated ±${CORRELATED_XY} xy (hand follows cube)"
echo "init: palm standoff dy=+${SPAWN_DY} dz=+${SPAWN_ABOVE}, finger_curl=${FINGER_CURL}, ~3cm non-contact hover; shape OFF"
echo "table_push=true object_goal=(0.0,0.0,0.555) (3-D object-only center goal)"
echo "palm_goal=OFF joint_goal=OFF obs_dim=49 goal_dim=3"
echo "isaacgym_reset_z_above=${RESET_Z_ABOVE} (also NVIDIA fall reset z<0.1)"
echo "nf_mix_task_goal_stats=true nf_mix_task_goal_frac=0.50"
echo "nf_state_only=false (forward log p(g|s,a) / r(s,a); default)"
echo "actor_reset=off (ppo_actor_reset_ep_len=1)"
echo "steps_per_iter=$((PPO_NUM_ENVS * EP_LEN))"
echo "num_iterations=$((NUM_STEPS / (PPO_NUM_ENVS * EP_LEN)))"
echo "nf_compact_small: sa${NF_SA_NUM_LAYERS}x${NF_SA_HIDDEN} r${NF_REP_SIZE} b${NF_NUM_BLOCKS} w${NF_COUPLING_WIDTH}"

nvidia-smi || true
python -u -c "
import isaacgym
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
  --isaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_table_push \
  --isaacgym_table_push_xyz=0.0,0.0,0.555 \
  --isaacgym_table_spawn \
  --isaacgym_table_spawn_object_xy=0.17,0.04 \
  --isaacgym_table_spawn_behind \
  --isaacgym_table_spawn_behind_dy="${SPAWN_DY}" \
  --isaacgym_table_spawn_behind_above="${SPAWN_ABOVE}" \
  --isaacgym_table_spawn_finger_curl_scale="${FINGER_CURL}" \
  --isaacgym_table_spawn_correlated_xy="${CORRELATED_XY}" \
  --isaacgym_large_table \
  --isaacgym_reset_z_above="${RESET_Z_ABOVE}" \
  --ppo_actor_reset_ep_len=1 \
  --ppo_rollout_length="${EP_LEN}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_anneal_ent_coef=true \
  --ppo_ent_coef_final="${ENT_COEF_FINAL}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_eval_interval=10 \
  --ppo_video_interval=0 \
  --ppo_checkpoint_interval=100 \
  --ppo_checkpoint_replay_max="${MAX_REPLAY_SIZE}" \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau="${REWARD_TAU}" \
  --ppo_use_external_reward \
  --ppo_external_reward_scale="${EXT_REW_SCALE}" \
  --nf_mix_task_goal_stats=true \
  --nf_mix_task_goal_frac=0.50 \
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
