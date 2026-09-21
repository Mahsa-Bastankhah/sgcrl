#!/bin/bash
#SBATCH --job-name=mpo_bin_z3to6
#SBATCH --output=slurm/final_metaworld_runs/mpo_crl_sawyer_bin_safeinit_z3to6_40m_s%a_%A.log
#SBATCH --error=slurm/final_metaworld_runs/mpo_crl_sawyer_bin_safeinit_z3to6_40m_s%a_%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-1

# MPO-CRL Sawyer bin, same recipe as job_mpo_crl_sawyer_bin_tau0p85_40m.slurm
# except orange NF init: object XY random, TCP XY on object, z Uniform(3–6cm),
# gripper 0.4, bounded/rejection servo. No MPO XY±5cm gripper jitter.
module purge

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

CUDA_LOADED=0
try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    if [ -z "${CUDA_HOME:-}" ]; then
      for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
        if [ -d "${_cand}/lib64" ]; then
          export CUDA_HOME="${_cand}"
          break
        fi
      done
    fi
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || \
  try_cuda cudatoolkit/12.5 || \
  try_cuda cudatoolkit/12.4 || true

if [ "$CUDA_LOADED" != 1 ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1
      echo "Fallback CUDA_HOME=${CUDA_HOME} (module load failed)"
      break
    fi
  done
fi

if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit and no /usr/local/cuda* found."
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEED=${SLURM_ARRAY_TASK_ID:-0}
ENV=sawyer_bin
NUM_STEPS=${NUM_STEPS:-40000000}
LOG_DIR="${LOG_DIR:-logs/final_metaworld_runs/mpo_crl_sawyer_bin_safeinit_z3to6_40m/}"
NUM_ENVS=4
ROLLOUT_T=256
CRL_UPDATES=128
POLICY_UPDATES=128
POLICY_BATCH=256
MIN_REPLAY=10000
DISCOUNT=0.99
REPR_TARGET_RATE=0.15

mkdir -p slurm/final_metaworld_runs "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} log_dir=${LOG_DIR}"
echo "E=${NUM_ENVS} T=${ROLLOUT_T} policy_updates=${POLICY_UPDATES} crl_updates=${CRL_UPDATES}"
echo "uniform_sampling=false repr_target_rate=${REPR_TARGET_RATE} discount=${DISCOUNT}"
echo "mpo_eval_interval=10"
echo "init: SAFE; TCP XY=obj; z Uniform 3-6cm; object XY random; no gripper XY jitter"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u baseline-agents/mpo-crl.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --mpo_log_dir_path="${LOG_DIR}" \
  --mpo_num_envs="${NUM_ENVS}" \
  --mpo_rollout_length="${ROLLOUT_T}" \
  --mpo_policy_updates_per_iter="${POLICY_UPDATES}" \
  --mpo_crl_updates_per_iter="${CRL_UPDATES}" \
  --mpo_policy_batch_size="${POLICY_BATCH}" \
  --mpo_crl_batch_size=256 \
  --mpo_min_replay_size="${MIN_REPLAY}" \
  --mpo_num_action_samples=20 \
  --discount="${DISCOUNT}" \
  --uniform_sampling=false \
  --ppo_crl_loss_direction=forward \
  --mpo_repr_target_update_rate="${REPR_TARGET_RATE}" \
  --mpo_target_update_period=100 \
  --mpo_policy_learning_rate=3e-4 \
  --mpo_dual_learning_rate=1e-2 \
  --mpo_policy_hidden_sizes="256,256,256,256,256,256" \
  --hidden_layer_sizes="256,256,256,256,256,256" \
  --sawyer_randomize_init=true \
  --sawyer_bin_safe_grasp_reset=true \
  --bin_randomize_tcp_z=true \
  --mpo_eval_interval=10 \
  --mpo_eval_episodes=5 \
  --mpo_checkpoint_interval=200 \
  --mpo_checkpoint_keep_last=5
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
