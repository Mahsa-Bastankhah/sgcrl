#!/bin/bash
#SBATCH --job-name=akt_mpo_hand16peace_s0
#SBATCH --output=slurm/mpo_crl_allegro_hand16peace_e1024_ep50_200m_seed0_%j.log
#SBATCH --error=slurm/mpo_crl_allegro_hand16peace_e1024_ep50_200m_seed0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas

# MPO-CRL baseline on the V-sign / peace pose (same env as peace NF).
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
NUM_ENVS=1024
POLICY_UPDATES=800
POLICY_BATCH=2048
CRL_UPDATES=10
LOG_DIR="logs/mpo_crl_allegro_hand16peace_e1024_ep50_200m_seed0_4h/"
mkdir -p slurm "${LOG_DIR}"
echo "seed=0 E=${NUM_ENVS} ep=${EP_LEN} steps=200000000"
echo "task=hand16peace object_free=true state=46 (all 23q+23qd) goal=16"
echo "figure: V-sign index+middle open, ring curled, thumb tucked"
echo "coordinate_mode=fully_scaled  no trim_sa  frozen init  large_table"
echo "MPO SPI=$((POLICY_UPDATES * POLICY_BATCH / (NUM_ENVS * EP_LEN)))"
echo "policy_updates=${POLICY_UPDATES} batch=${POLICY_BATCH} crl_updates=${CRL_UPDATES}"
echo "mpo_eval_interval=10"
echo "LOG_DIR=${LOG_DIR}"
nvidia-smi || true

python -u baseline-agents/mpo-crl.py \
  --env=allegro_kuka_throw \
  --seed=0 \
  --num_steps=200000000 \
  --mpo_log_dir_path="${LOG_DIR}" \
  --mpo_num_envs="${NUM_ENVS}" \
  --ppo_num_envs="${NUM_ENVS}" \
  --mpo_rollout_length="${EP_LEN}" \
  --isaacgym_episode_length="${EP_LEN}" \
  --isaacgym_pipeline=gpu \
  --noisaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_control_sanity_mode=hand16peace \
  --isaacgym_coordinate_mode=fully_scaled \
  --isaacgym_large_table \
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
