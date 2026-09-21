#!/bin/bash
#SBATCH --job-name=akt_mpo_arminit_z055_ep150
#SBATCH --output=slurm/newbl_mpo_hard_arminit_z055_ep150_%j.log
#SBATCH --error=slurm/newbl_mpo_hard_arminit_z055_ep150_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --account=seas

# MPO-CRL twin of working keep-arm hard NF. T=150 fut80 z=0.55 300M.
#   sbatch jobs/job_mpo_new_allegrobaselines_hard_arminit_z055_ep150_300m_seed0.slurm
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
EP_LEN=150
FUT_H=80
NUM_ENVS=1024
NUM_STEPS=300000000
POLICY_UPDATES=800
POLICY_BATCH=2048
CRL_UPDATES=10
LOG_DIR="logs/new_allegrobaselines/mpo_hard_arminit_z055_ep150_fut80_300m_seed0_14h/"
mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  host=$(hostname)"
echo "MPO seed=${SEED} E=${NUM_ENVS} ep=${EP_LEN} crl_future_horizon=${FUT_H} steps=${NUM_STEPS} goal_z=0.55"
echo "init: keep-arm palm-up pad-hold  bucket=(0.45,-0.60)  fut80"
echo "success=in-bucket  mpo_eval_interval=10"
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
  --crl_future_horizon="${FUT_H}" \
  --isaacgym_pipeline=gpu \
  --noisaacgym_randomize_init \
  --noisaacgym_randomize_object_xyz \
  --noisaacgym_randomize_object_shape \
  --isaacgym_fixed_target_xyz=0.45,-0.60,0.40 \
  --isaacgym_goal_z=0.55 \
  --isaacgym_throw_success=in_bucket \
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
