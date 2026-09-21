#!/bin/bash
#SBATCH --job-name=akt_nf_hand16fig_hard_full_s0
#SBATCH --output=slurm/ppo_allegro_hand16fig_hardzigzag_fullrange_e1024_nf_compactsmall_ep50_200m_seed0_%j.log
#SBATCH --error=slurm/ppo_allegro_hand16fig_hardzigzag_fullrange_e1024_nf_compactsmall_ep50_200m_seed0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas

# Object-free 16-DOF hard zigzag + full-range Uniform hand reset.
# State = all 23 q + 23 qd (no trim_sa). Goal = 16 hand q.
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

EP_LEN=50
LOG_DIR="logs/ppo_allegro_hand16fig_hardzigzag_fullrange_e1024_nf_compactsmall_fully_scaled_ep50_200m_seed0_4h/"
mkdir -p slurm "${LOG_DIR}"
echo "seed=0 E=1024 ep=${EP_LEN} steps=200000000"
echo "task=hand16fig object_free=true state=46 (all 23q+23qd) goal=16"
echo "figure: hard zigzag anti-synergy (index/mid/ring/thumb all different)"
echo "reset: Uniform[lo,hi] on all 16 hand joints  arm frozen  no trim_sa"
echo "eval nf_binary_accuracy every 10 iters"
echo "coordinate_mode=fully_scaled  large_table"
echo "nf compact-small r(s,a) mix50 extrew1 ent=0.01 fixed minstd=1e-4"
echo "ppo_eval_interval=10"
echo "LOG_DIR=${LOG_DIR}"
nvidia-smi || true

python -u ppo_contrastive.py \
  --env=allegro_kuka_throw \
  --seed=0 \
  --num_steps=200000000 \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs=1024 \
  --isaacgym_episode_length="${EP_LEN}" \
  --ppo_rollout_length="${EP_LEN}" \
  --isaacgym_pipeline=gpu \
  --isaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_control_sanity_mode=hand16fig \
  --isaacgym_control_sanity_trim_init_mode=full_range \
  --isaacgym_coordinate_mode=fully_scaled \
  --isaacgym_large_table \
  --ppo_crl_steps_per_iter=10 \
  --ppo_ent_coef=0.01 \
  --ppo_anneal_ent_coef=false \
  --ppo_actor_min_std=1e-4 \
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
