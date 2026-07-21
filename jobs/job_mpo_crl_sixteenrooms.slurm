#!/bin/bash
#SBATCH --job-name=mpo_crl_16r
#SBATCH --output=slurm/mpo_crl_sixteenrooms_%A_%a.log
#SBATCH --error=slurm/mpo_crl_sixteenrooms_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=seas
#SBATCH --array=0-2

# Online MPO-CRL on plain SixteenRooms (2-D state/action/goal).
# Same Acme-MPO-style loop as FourRooms verify: E=8, T=128,
# policy_updates=128, batch=256 → SPI=32; CRL updates=64.
# Policy Q = target φ·ψ; CRL uses geometric future goals.
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
  module avail cudatoolkit 2>&1 || true
  ls -d /usr/local/cuda* 2>&1 || true
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source .venv/bin/activate

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export TF_CPP_MIN_LOG_LEVEL=2

ENV="point_SixteenRooms"
NUM_STEPS="${NUM_STEPS:-10000000}"
SEED_OFFSET="${SEED_OFFSET:-0}"
SEED=$(( SEED_OFFSET + ${SLURM_ARRAY_TASK_ID:-0} ))
LOG_DIR="${LOG_DIR:-logs/mpo_crl_sixteenrooms/}"

NUM_ENVS=8
ROLLOUT_T=128
CRL_UPDATES=64
# 1024 inserts/iteration; 128 * batch256 / 1024 = 32 samples per insert.
POLICY_UPDATES=128
MIN_REPLAY=10000
DISCOUNT=0.99
REPR_TARGET_RATE=0.005

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} log_dir=${LOG_DIR}"
echo "E=${NUM_ENVS} T=${ROLLOUT_T} policy_updates=${POLICY_UPDATES} crl_updates=${CRL_UPDATES}"
echo "repr_target_update_rate=${REPR_TARGET_RATE} discount=${DISCOUNT}"
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
  --mpo_policy_batch_size=256 \
  --mpo_min_replay_size="${MIN_REPLAY}" \
  --mpo_num_action_samples=20 \
  --discount="${DISCOUNT}" \
  --mpo_repr_target_update_rate="${REPR_TARGET_RATE}" \
  --mpo_target_update_period=100 \
  --mpo_policy_learning_rate=3e-4 \
  --mpo_dual_learning_rate=1e-2 \
  --mpo_policy_hidden_sizes="256,256,256,256,256,256" \
  --hidden_layer_sizes="256,256,256,256,256,256" \
  --mpo_eval_interval=10 \
  --mpo_eval_episodes=5 \
  --mpo_checkpoint_interval=200 \
  --mpo_checkpoint_keep_last=5
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
