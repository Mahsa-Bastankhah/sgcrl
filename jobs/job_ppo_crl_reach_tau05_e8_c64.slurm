#!/bin/bash
#SBATCH --job-name=ppo_crl_reach_t05
#SBATCH --output=slurm/ppo_crl_reach_tau05_e8_c64_%A_%a.log
#SBATCH --error=slurm/ppo_crl_reach_tau05_e8_c64_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=seas
#SBATCH --array=0-1

# sawyer_reach: CRL φ·ψ reward with ppo_crl_repr_tau=0.5
# Shared Metaworld sweep knobs: num_envs=8, rollout=256, crl_steps=64,
# reward EMA tau=0.5, seeds 0-1, hidden=256x6, 10M steps.
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

ENV=sawyer_reach
NUM_STEPS=${NUM_STEPS:-10000000}
SEED_OFFSET=${SEED_OFFSET:-0}
SEED=$(( SEED_OFFSET + ${SLURM_ARRAY_TASK_ID:-0} ))

LOG_DIR="${LOG_DIR:-logs/ppo_crl_reach_tau05_e8_c64/}"
PPO_NUM_ENVS=${PPO_NUM_ENVS:-8}
ROLLOUT_LENGTH=${ROLLOUT_LENGTH:-256}
CRL_STEPS=${CRL_STEPS:-64}
ENT_COEF=0.01
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
REWARD_TAU=${REWARD_TAU:-0.5}


mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=crl crl_repr_tau=${REWARD_TAU}"
echo "log_dir=${LOG_DIR} num_envs=${PPO_NUM_ENVS} rollout=${ROLLOUT_LENGTH} crl_steps=${CRL_STEPS} reward_tau=${REWARD_TAU}"

nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --ppo_rollout_length="${ROLLOUT_LENGTH}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_skip_first_eval=true \
  --hidden_layer_sizes="256,256,256,256,256,256" \
  --ppo_repr_mode=crl \
  --ppo_crl_loss_direction=forward \
  --ppo_crl_repr_tau="${REWARD_TAU}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
