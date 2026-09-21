#!/bin/bash
#SBATCH --job-name=akt_mpo_keeparm_000m040_ball75_50m
#SBATCH --output=slurm/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy000m040_norand_ep50_50m_goalball75_seed0_%j.log
#SBATCH --error=slurm/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy000m040_norand_ep50_50m_goalball75_seed0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=seas

# MPO twin of NF keep-arm hidetable (0.00,-0.40) 7.5cm goal-ball (3884297), 50M.
#   sbatch jobs/job_mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy000m040_norand_ep50_50m_goalball75_seed0_1h.slurm
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
EP_LEN=50
NUM_ENVS=1024
NUM_STEPS=50000000
POLICY_UPDATES=800
POLICY_BATCH=2048
CRL_UPDATES=10
LOG_DIR="logs/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy000m040_norand_ep50_50m_goalball75_seed${SEED}_1h/"
mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  host=$(hostname)"
echo "seed=${SEED} E=${NUM_ENVS} ep=${EP_LEN} steps=${NUM_STEPS}"
echo "table_push=OFF  bucket=(0.00,-0.40,0.40)  palm_goal=OFF"
echo "success=goal_ball (z+0.05, r=0.075=7.5cm, one frame)"
echo "init: keep_arm palm-up wrist=-pi  curl=1  cube on palm pad  NO init noise"
echo "table=HIDDEN  isaacgym_hide_table"
echo "MPO SPI=$((POLICY_UPDATES * POLICY_BATCH / (NUM_ENVS * EP_LEN)))"
echo "policy_updates=${POLICY_UPDATES} batch=${POLICY_BATCH} crl_updates=${CRL_UPDATES}"
echo "mpo_eval_interval=10"
echo "LOG_DIR=${LOG_DIR}"
nvidia-smi || true

python -u baseline-agents/mpo-crl.py \
  --env=allegro_kuka_throw \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --mpo_log_dir_path="${LOG_DIR}" \
  --mpo_num_envs="${NUM_ENVS}" \
  --ppo_num_envs="${NUM_ENVS}" \
  --mpo_rollout_length="${EP_LEN}" \
  --isaacgym_episode_length="${EP_LEN}" \
  --isaacgym_pipeline=gpu \
  --noisaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_fixed_target_xyz=0.00,-0.40,0.40 \
  --isaacgym_throw_success=goal_ball \
  --isaacgym_hide_table \
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
  --mpo_policy_updates_per_iter="${POLICY_UPDATES}" \
  --mpo_policy_batch_size="${POLICY_BATCH}" \
  --mpo_crl_updates_per_iter="${CRL_UPDATES}" \
  --mpo_min_replay_size=10000 \
  --mpo_num_action_samples=20 \
  --discount=0.99 \
  --mpo_repr_target_update_rate=0.005 \
  --mpo_target_update_period=100 \
  --mpo_policy_learning_rate=3e-4 \
  --mpo_dual_learning_rate=1e-2 \
  --mpo_policy_hidden_sizes="256,256,256,256,256,256" \
  --hidden_layer_sizes="256,256,256,256,256,256" \
  --max_replay_size=10000000 \
  --mpo_eval_interval=10 \
  --mpo_checkpoint_interval=100 \
  --mpo_checkpoint_keep_last=5
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
