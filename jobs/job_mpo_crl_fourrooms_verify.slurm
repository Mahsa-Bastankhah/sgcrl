#!/bin/bash
#SBATCH --job-name=mpo_crl_4r
#SBATCH --output=slurm/mpo_crl_fourrooms_verify_%j.log
#SBATCH --error=slurm/mpo_crl_fourrooms_verify_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=seas

# FourRooms verification before scaling MPO-CRL back to SixteenRooms.
# E=8 and T=128 insert 1024 transitions per iteration. 128 MPO updates with
# batch 256 give 32 replay samples per insert, matching Acme MPO's default SPI.
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
      break
    fi
  done
fi

if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load CUDA."
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

ENV=point_FourRooms
SEED=${SEED:-0}
NUM_STEPS=${NUM_STEPS:-1000000}
LOG_DIR=${LOG_DIR:-logs/mpo_crl_fourrooms_verify/}

mkdir -p slurm "${LOG_DIR}"

echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS}"
echo "E=8 T=128 policy_updates=128 batch=256 SPI=32"
echo "CRL updates=64; repr target update rate=0.005"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u baseline-agents/mpo-crl.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --mpo_log_dir_path="${LOG_DIR}" \
  --mpo_num_envs=8 \
  --mpo_rollout_length=128 \
  --mpo_policy_updates_per_iter=128 \
  --mpo_crl_updates_per_iter=64 \
  --mpo_policy_batch_size=256 \
  --mpo_min_replay_size=10000 \
  --mpo_num_action_samples=20 \
  --discount=0.99 \
  --mpo_repr_target_update_rate=0.005 \
  --mpo_target_update_period=100 \
  --mpo_policy_learning_rate=3e-4 \
  --mpo_dual_learning_rate=1e-2 \
  --mpo_policy_hidden_sizes="256,256,256,256,256,256" \
  --hidden_layer_sizes="256,256,256,256,256,256" \
  --mpo_eval_interval=10 \
  --mpo_eval_episodes=5 \
  --mpo_checkpoint_interval=100 \
  --mpo_checkpoint_keep_last=3
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
