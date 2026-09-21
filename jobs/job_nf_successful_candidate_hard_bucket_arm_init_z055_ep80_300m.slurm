#!/bin/bash
#SBATCH --job-name=akt_nf_hard_bucket_arm_init_z055
#SBATCH --output=slurm/successful_candidate_hard_bucket_arm_init_z055_ep80_300m_%j.log
#SBATCH --error=slurm/successful_candidate_hard_bucket_arm_init_z055_ep80_300m_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas

# Twin of NF 3884249 (keep-arm pad-hold, hard bucket) with near-rim goal z=0.55, T=80, 300M.
# Bucket actor stays at (0.45,-0.60,0.40). Success still in-bucket cylinder.
#   sbatch jobs/job_nf_successful_candidate_hard_bucket_arm_init_z055_ep80_300m.slurm
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

SEED=0
EP_LEN=80
NUM_STEPS=300000000
LOG_DIR="logs/successful_candidate/hard_bucket_arm_init_z055_ep80_q40_300m_seed${SEED}_4h/"
mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  host=$(hostname)"
echo "seed=${SEED} E=1024 ep=${EP_LEN} steps=${NUM_STEPS}"
echo "table_push=OFF  bucket=(0.45,-0.60,0.40)  goal_z=0.55 (near rim)"
echo "success=in-bucket cylinder (r=0.12 h=0.198)  NVIDIA success-respawn OFF"
echo "init: keep_arm palm-up wrist=-pi  curl=1  cube on palm pad  NO init noise"
echo "table=NVIDIA table_narrow  (no hide_table, no large_table)"
echo "nf compact-small r(s,a) mix50 extrew1  q40shift ema=0.1  no fall_penalty"
echo "ppo_success_replace_fails=0"
echo "ppo_eval_interval=10"
echo "ppo_video_interval=100"
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
  --isaacgym_pipeline=gpu \
  --noisaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_fixed_target_xyz=0.45,-0.60,0.40 \
  --isaacgym_goal_z=0.55 \
  --isaacgym_table_spawn \
  --isaacgym_table_spawn_object_xy=0.0,0.0 \
  --isaacgym_table_spawn_in_hand \
  --isaacgym_table_spawn_in_hand_keep_arm \
  --isaacgym_table_spawn_in_hand_offset=0.042 \
  --isaacgym_table_spawn_in_hand_obj_noise=0 \
  --isaacgym_table_spawn_in_hand_wrist_offset=-3.141592653589793 \
  --isaacgym_table_spawn_in_hand_wrist_noise=0 \
  --isaacgym_table_spawn_finger_curl_scale=1.0 \
  --isaacgym_table_spawn_finger_noise=0 \
  --isaacgym_table_spawn_arm_noise=0 \
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
  --ppo_nf_reward_shift_q=0.40 \
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
