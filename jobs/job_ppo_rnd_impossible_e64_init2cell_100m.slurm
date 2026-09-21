#!/bin/bash
#SBATCH --job-name=ppo_rnd_imp_r
#SBATCH --output=slurm/ppo_rnd_impossible_e64_init2cell_100m_%j.log
#SBATCH --error=slurm/ppo_rnd_impossible_e64_init2cell_100m_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --account=seas

# Impossible maze, PPO-RND, 100M env steps. Same env / collection / init
# as the succeeding NF tiny 100M job:
#   E=64 T=128, start jitter Chebyshev ≤2 of (0,0) + [0,1)^2,
#   goal (6,8), eval + PNGs from canonical start (0,0).
# PPO-side knobs match NF (6x256, 10 epochs, ent=0.05, lr=3e-4).
# RND-specific: int_coef=1, ext_coef=1, obs RMS (kept).
set -euo pipefail

ROOT=/n/fs/mislresearch/sgcrl
PYTHON="${ROOT}/.venv/bin/python"
ENV=point_Impossible
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-100000000}"
LOG_DIR="${LOG_DIR:-logs/ppo_rnd_impossible_e64_init2cell_100m/}"
RUN_DIR="${LOG_DIR}/ppo_rnd_${ENV}_${SEED}"

module purge
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

CUDA_LOADED=0
for CUDA_MODULE in cudatoolkit/12.6 cudatoolkit/12.5 cudatoolkit/12.4; do
  if module load "${CUDA_MODULE}" 2>/dev/null; then
    CUDA_LOADED=1
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    echo "Loaded CUDA module: ${CUDA_MODULE}"
    break
  fi
done
if [[ "${CUDA_LOADED}" != "1" ]]; then
  for CANDIDATE in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [[ -d "${CANDIDATE}/lib64" ]]; then
      CUDA_LOADED=1
      export CUDA_HOME="${CANDIDATE}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      echo "Using CUDA fallback: ${CUDA_HOME}"
      break
    fi
  done
fi
if [[ "${CUDA_LOADED}" != "1" || -z "${CUDA_HOME:-}" ]]; then
  echo "ERROR: no CUDA toolkit found"
  exit 1
fi

cd "${ROOT}"
source .venv/bin/activate
unset -f python python3 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export TF_CPP_MIN_LOG_LEVEL=2

mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} log_dir=${LOG_DIR}"
echo "E=64 T=128 epochs=10 minibatches=4 ent_coef=0.05 lr=3e-4"
echo "eval_interval=10 video_interval=200"
echo "maze_randomize_start=true jitter_cells=2 (Chebyshev ≤2 of (0,0) + [0,1)^2)"
echo "RND: int_coef=1 ext_coef=1 hidden=6x256"
echo "run_dir=${RUN_DIR}"
echo "ppo_eval_interval=10"
nvidia-smi || true
"${PYTHON}" -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

"${PYTHON}" -u baseline-agents/ppo-rnd-gym.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir="${LOG_DIR}" \
  --num_envs=64 \
  --rollout_length=128 \
  --num_epochs=10 \
  --num_minibatches=4 \
  --learning_rate=3e-4 \
  --discount=.99 \
  --int_discount=.99 \
  --gae_lambda=.95 \
  --clip_coef=.2 \
  --ent_coef=.05 \
  --int_coef=1 \
  --ext_coef=1 \
  --hidden_sizes=256,256,256,256,256,256 \
  --eval_interval=10 \
  --eval_episodes=5 \
  --checkpoint_interval=400 \
  --video_interval=200 \
  --maze_randomize_start \
  --maze_start_jitter_cells=2
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
