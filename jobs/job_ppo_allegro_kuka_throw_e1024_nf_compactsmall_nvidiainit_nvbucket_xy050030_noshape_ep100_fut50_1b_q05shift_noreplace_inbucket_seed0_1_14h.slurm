#!/bin/bash
#SBATCH --job-name=akt_nf_nvidiainit_ep100_fut50_1b
#SBATCH --output=slurm/ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit_nvbucket_xy050030_noshape_ep100_fut50_1b_q05shift_noreplace_inbucket_seed0_1_%A_%a.log
#SBATCH --error=slurm/ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit_nvbucket_xy050030_noshape_ep100_fut50_1b_q05shift_noreplace_inbucket_seed0_1_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-1

# NVIDIA-init pick+throw, T=100, NF future-goal sample cap 50, 1B, seeds 0/1.
# Same env as the T=50 / T=150 nvidiainit runs. 14:59 covers ~1B at T=100.
#   sbatch jobs/job_ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit_nvbucket_xy050030_noshape_ep100_fut50_1b_q05shift_noreplace_inbucket_seed0_1_14h.slurm
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

SEED=${SLURM_ARRAY_TASK_ID:-0}
EP_LEN=100
FUT_H=50
NUM_STEPS=1000000000
LOG_DIR="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit_nvbucket_xy050030_noshape_ep100_fut50_1b_q05shift_noreplace_inbucket_seed${SEED}_14h/"
mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  host=$(hostname)"
echo "seed=${SEED} E=1024 ep=${EP_LEN} crl_future_horizon=${FUT_H} steps=${NUM_STEPS}"
echo "table_push=OFF  bucket=(0.50,-0.30,0.40)  goal still z+0.05 for NF"
echo "success=in-bucket cylinder (r=0.12 h=0.198)  NVIDIA success-respawn OFF"
echo "init: NVIDIA default (hand-in-air, cube on table, randomize pose/joints)"
echo "  NO object-shape randomization"
echo "table=NVIDIA table_narrow  (no hide_table, no large_table, no table_spawn)"
echo "nf compact-small r(s,a) mix50 extrew1  q05shift ema=0.1  no fall_penalty"
echo "ppo_success_replace_fails=0"
echo "ppo_eval_interval=10"
echo "ppo_video_interval=100  (1-ep stoch camera, T=100)"
echo "LOG_DIR=${LOG_DIR}"
nvidia-smi || true

python -u ppo_contrastive.py \
  --env=allegro_kuka_throw \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs=1024 \
  --isaacgym_episode_length="${EP_LEN}" \
  --ppo_rollout_length="${EP_LEN}" \
  --crl_future_horizon="${FUT_H}" \
  --isaacgym_pipeline=gpu \
  --isaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_fixed_target_xyz=0.50,-0.30,0.40 \
  --ppo_actor_reset_ep_len=1 \
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
  --ppo_checkpoint_replay_max=10000000 \
  --max_replay_size=10000000 \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau=0.5 \
  --ppo_nf_reward_shift_q=0.05 \
  --ppo_nf_reward_shift_ema=0.1 \
  --ppo_use_external_reward \
  --ppo_external_reward_scale=1 \
  --ppo_success_replace_fails=0 \
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
