#!/bin/bash
#SBATCH --job-name=allegro_four2mh_sanity_nf
#SBATCH --output=slurm/ppo_allegro_four2mh_control_sanity_e1024_nf_200m_ep150_4h_%A_%a.log
#SBATCH --error=slurm/ppo_allegro_four2mh_control_sanity_e1024_nf_200m_ep150_4h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-1024}
EP_LEN=${EP_LEN:-150}
LOG_DIR="${LOG_DIR:-logs/ppo_allegro_four2mh_control_sanity_e1024_nf_compact_small_200m_ep150_4h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} seed=${SEED} num_steps=${NUM_STEPS}"
echo "task=four2mh_control_sanity object_free=true state_dim=46 goal_dim=8"
echo "2 joints/finger MED-HARD: idx/mid/ring/thumb=(.42,1.00)(.37,.92)(.32,.87)(.50,.82)"
echo "success max_abs_error: hard<=0.10 easy<=0.20 very_easy<=0.30 rad"
echo "episode_length=${EP_LEN} num_envs=${PPO_NUM_ENVS} ppo_eval_interval=10"
echo "nf_state_only=false reward=log_p(goal|state,action)"
nvidia-smi || true

python -u ppo_contrastive.py \
  --env=allegro_kuka_throw \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --isaacgym_episode_length="${EP_LEN}" \
  --isaacgym_pipeline=gpu \
  --noisaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_control_sanity_mode=four2mh \
  --isaacgym_large_table \
  --ppo_rollout_length="${EP_LEN}" \
  --ppo_crl_steps_per_iter=10 \
  --ppo_ent_coef=0.05 \
  --ppo_anneal_ent_coef=true \
  --ppo_ent_coef_final=0.01 \
  --ppo_actor_min_std=1e-5 \
  --ppo_discount=0.99 \
  --ppo_clip_coef=0.2 \
  --ppo_eval_interval=10 \
  --ppo_video_interval=100 \
  --ppo_checkpoint_interval=100 \
  --max_replay_size=10000000 \
  --ppo_checkpoint_replay_max=10000000 \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau=0.5 \
  --ppo_use_external_reward \
  --ppo_external_reward_scale=1 \
  --nf_mix_task_goal_stats=true \
  --nf_mix_task_goal_frac=0.50 \
  --nf_rep_size=64 \
  --nf_num_blocks=6 \
  --nf_coupling_width=192 \
  --nf_sa_hidden=192 \
  --nf_sa_num_layers=3 \
  --nf_grad_clip=1.0 \
  --nf_noise_std=0.05 \
  --nf_goal_std_min=0.02 \
  --nf_goal_enc_size=0 \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
