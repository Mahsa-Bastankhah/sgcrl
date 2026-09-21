#!/bin/bash
#SBATCH --job-name=akt_nf_keeparm_padhold_fallm4_s0
#SBATCH --output=slurm/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_largetable15_objgoal_norand_ep50_200m_fallm4_nosettle_seed0_%j.log
#SBATCH --error=slurm/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_largetable15_objgoal_norand_ep50_200m_fallm4_nosettle_seed0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00
#SBATCH --account=seas

# Same as keeparm pad-hold large-table objgoal ep50, plus:
#   - mid-episode reset teleports to stored pad pose (no 8-step settle)
#   - after return-norm, -4 when the cube falls (object z < 0.1).
# Timeouts are not falls. 7h was requested for this 200M recipe.
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
LOG_DIR="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_largetable15_objgoal_norand_ep50_200m_fallm4_nosettle_seed0_7h/"
mkdir -p slurm "${LOG_DIR}"
echo "seed=0 E=1024 ep=${EP_LEN} steps=200000000"
echo "table_push object_goal=(0.00,-0.22,0.555)  palm_goal=OFF  state49/goal3"
echo "init: keep_arm palm-up wrist=-pi  curl=1  cube on palm pad  NO init noise"
echo "  pad = palm + 0.034 world-up + 0.012 along-fingers"
echo "  settle=construction+VecEnv.reset only (skip mid-episode reset)"
echo "  noisaacgym_randomize_init  wrist_noise=0  obj_noise=0  finger_noise=0  arm_noise=0"
echo "table=1.5x1.5  isaacgym_large_table"
echo "nf compact-small r(s,a) mix50 extrew1  fall_penalty=-4 after-norm"
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
  --noisaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_table_push \
  --isaacgym_table_push_xyz=0.00,-0.22,0.555 \
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
  --isaacgym_large_table \
  --ppo_actor_reset_ep_len=1 \
  --ppo_crl_steps_per_iter=10 \
  --ppo_ent_coef=0.05 \
  --ppo_anneal_ent_coef=true \
  --ppo_ent_coef_final=0.01 \
  --ppo_actor_min_std=1e-5 \
  --ppo_discount=0.99 \
  --ppo_clip_coef=0.2 \
  --ppo_eval_interval=10 \
  --ppo_video_interval=0 \
  --ppo_checkpoint_interval=100 \
  --ppo_checkpoint_replay_max=10000000 \
  --max_replay_size=10000000 \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau=0.5 \
  --ppo_use_external_reward \
  --ppo_external_reward_scale=1 \
  --ppo_isaacgym_fall_penalty=-4 \
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
