#!/bin/bash
#SBATCH --job-name=ppo_ablate
#SBATCH --output=slurm/ppo_ablate_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas
#SBATCH --array=0-9

module purge

# Drop any broken inherited CUDA paths from the submit host.
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
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || true
if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit. Run 'module avail cudatoolkit' and edit this script."
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

ENV=${ENV:-point_FourRooms}
NUM_STEPS=${NUM_STEPS:-1000000}
SEED_OFFSET=${SEED_OFFSET:-30}

# 5 variants, no-repr-normalization only.
NUM_VARIANTS=5
SEED_INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} / NUM_VARIANTS ))
VARIANT_INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} % NUM_VARIANTS ))
SEED=$(( SEED_OFFSET + SEED_INDEX ))

# Sentinel values: <=0 means "do not override" in ppo_contrastive.py.
CRL_STEPS=-1
PPO_CLIP=-1
ACTOR_MIN_STD=-1
TAG=""

case "$VARIANT_INDEX" in
  0)
    TAG="crl_steps_128"
    CRL_STEPS=128
    ;;
  1)
    TAG="crl_steps_32"
    CRL_STEPS=32
    ;;
  2)
    TAG="clip_0p1"
    PPO_CLIP=0.1
    ;;
  3)
    TAG="minstd_0p05"
    ACTOR_MIN_STD=0.05
    ;;
  4)
    TAG="minstd_0p2"
    ACTOR_MIN_STD=0.2
    ;;
  *)
    echo "Invalid VARIANT_INDEX=$VARIANT_INDEX"
    exit 2
    ;;
esac

REPR_TAG="repr_raw"
REPR_FLAG="--norepr_norm"
LOG_DIR="logs/ppo_ablations/${TAG}_${REPR_TAG}/"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} variant=${TAG} repr_mode=${REPR_TAG}"
echo "overrides: crl_steps=${CRL_STEPS} clip=${PPO_CLIP} min_std=${ACTOR_MIN_STD}"
echo "log_dir=${LOG_DIR}"
nvidia-smi
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_clip_coef="${PPO_CLIP}" \
  --ppo_actor_min_std="${ACTOR_MIN_STD}" \
  "${REPR_FLAG}"
