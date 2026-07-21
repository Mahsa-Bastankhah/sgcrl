#!/bin/bash
#SBATCH --job-name=ppo_rnd_peg_40m
#SBATCH --output=slurm/ppo_rnd_peg_40m_%A_%a.log
#SBATCH --error=slurm/ppo_rnd_peg_40m_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=seas
#SBATCH --array=0-1

# PPO+RND on Sawyer peg for 40M env steps (2 seeds). Walltime sized from
# prior 10M peg RND (~2.5h @ ~1k sps) → 40M ≈ 10h; 24h leaves headroom.
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
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

ENV=sawyer_peg
NUM_STEPS=${NUM_STEPS:-40000000}
SEED=${SLURM_ARRAY_TASK_ID:-0}
LOG_DIR=${LOG_DIR:-logs/ppo_rnd_peg_40m}

mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} env=${ENV} seed=${SEED} num_steps=${NUM_STEPS}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u baseline-agents/ppo-rnd-gym.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir="${LOG_DIR}" \
  --num_envs=8 \
  --rollout_length=256 \
  --num_epochs=4 \
  --num_minibatches=8 \
  --learning_rate=1e-4 \
  --discount=.99 \
  --int_discount=.99 \
  --gae_lambda=.95 \
  --clip_coef=.2 \
  --ent_coef=.01 \
  --int_coef=.25 \
  --ext_coef=1 \
  --hidden_sizes=256,256,256 \
  --eval_interval=50 \
  --eval_episodes=10 \
  --checkpoint_interval=500
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
