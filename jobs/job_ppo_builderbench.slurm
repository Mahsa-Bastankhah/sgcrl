#!/bin/bash
#SBATCH --job-name=ppo_bb
#SBATCH --output=slurm/ppo_builderbench_%j.log
#SBATCH --error=slurm/ppo_builderbench_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=seas
#
# Generic BuilderBench PPO-CRL job. Override env / scale via sbatch --export:
#
#   sbatch --export=ALL,ENV=builderbench_creative_1_task2 jobs/job_ppo_builderbench.slurm
#   sbatch --export=ALL,ENV=builderbench_creative_3_task1,PPO_NUM_ENVS=64 jobs/job_ppo_builderbench.slurm
#
# Valid creative env ids: builderbench_creative_{N}_task{K}
#   creative-1: task1, task2
#   creative-2 .. creative-9: task1 only
#
# List all installed tasks:
#   python -c "from envs.builderbench_utils import list_creative_env_names as L; print(*L(), sep='\n')"

unset LD_PRELOAD

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
  echo "ERROR: Could not load cudatoolkit."
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench

source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV=${ENV:-builderbench_creative_1_task1}
SEED=${SEED:-0}
NUM_STEPS=${NUM_STEPS:-200000000}
LOG_DIR="${LOG_DIR:-logs/ppo_${ENV}/}"
PPO_NUM_ENVS=${PPO_NUM_ENVS:-8}
CRL_STEPS=${CRL_STEPS:-128}
# Replay buffer defaults in ppo_contrastive (when --max_replay_size omitted):
#   E>=1024 → 10M transitions, E>=256 → 1M transitions.
ENT_COEF=0.05
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
PPO_CHECKPOINT_INTERVAL=${PPO_CHECKPOINT_INTERVAL:-300}

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS}"
echo "log_dir=${LOG_DIR} num_envs=${PPO_NUM_ENVS} crl_steps=${CRL_STEPS}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_checkpoint_interval="${PPO_CHECKPOINT_INTERVAL}" \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=50 \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
