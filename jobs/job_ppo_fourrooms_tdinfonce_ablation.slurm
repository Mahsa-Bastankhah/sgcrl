#!/bin/bash
#SBATCH --job-name=ppo_4r_tdi_abl
#SBATCH --output=slurm/ppo_fourrooms_tdinfonce_ablation_%j.log
#SBATCH --error=slurm/ppo_fourrooms_tdinfonce_ablation_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# FourRooms TD-InfoNCE ablation (mix γ=0.99).
# Required/typical env:
#   ABLATION=logsum   → logsumexp coef=0.01, no repr_norm
#   ABLATION=reprnorm → repr_norm, logsumexp coef=0
module purge
unset LD_PRELOAD
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

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

_NVIDIA_LIB_ROOT="${VIRTUAL_ENV}/lib/python3.9/site-packages/nvidia"
if [ -d "${_NVIDIA_LIB_ROOT}" ]; then
  _NVLIBS="$(find "${_NVIDIA_LIB_ROOT}" -name lib -type d 2>/dev/null | paste -sd: -)"
  if [ -n "${_NVLIBS}" ]; then
    export LD_LIBRARY_PATH="${_NVLIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export TF_CPP_MIN_LOG_LEVEL=2

ENV="point_FourRooms"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-300000}"
CRL_REPR_TAU="${CRL_REPR_TAU:-0.5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
ABLATION="${ABLATION:?ABLATION required: logsum|reprnorm}"

CRL_STEPS=64
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2

EXTRA_FLAGS=()
case "${ABLATION}" in
  logsum)
    LOG_DIR="${LOG_DIR:-logs/ppo_fourrooms_tdinfonce_ablation_logsum/}"
    EXTRA_FLAGS+=(--ppo_td_infonce_logsumexp_coef=0.01)
    # no --repr_norm
    ABL_DESC="logsumexp=0.01, repr_norm=OFF"
    ;;
  reprnorm)
    LOG_DIR="${LOG_DIR:-logs/ppo_fourrooms_tdinfonce_ablation_reprnorm/}"
    EXTRA_FLAGS+=(--repr_norm --ppo_td_infonce_logsumexp_coef=0.0)
    ABL_DESC="logsumexp=OFF, repr_norm=ON"
    ;;
  *)
    echo "ERROR: ABLATION must be logsum or reprnorm (got ${ABLATION})"
    exit 2
    ;;
esac

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS}"
echo "ablation=${ABLATION}: ${ABL_DESC}"
echo "mix_gamma=0.99 (default)  HER/discount=${DISCOUNT_FACTOR}"
echo "eval_interval=${EVAL_INTERVAL}"
echo "log_dir=${LOG_DIR}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_repr_mode=tdinfonce \
  --ppo_crl_repr_tau="${CRL_REPR_TAU}" \
  --discount="${DISCOUNT_FACTOR}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_eval_interval="${EVAL_INTERVAL}" \
  --hidden_layer_sizes="256,256,256,256,256,256" \
  "${EXTRA_FLAGS[@]}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
