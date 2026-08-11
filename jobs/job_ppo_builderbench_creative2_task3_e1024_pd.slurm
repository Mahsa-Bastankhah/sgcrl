#!/bin/bash
#SBATCH --job-name=ppo_bb_c2t3_pd
#SBATCH --output=slurm/ppo_builderbench_creative2_task3_e1024_pd_%j.log
#SBATCH --error=slurm/ppo_builderbench_creative2_task3_e1024_pd_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --account=seas

# 2-cube tilting (creative-2-task3), PD control, filtered high-level obs (xyz+select).
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

ENV=${ENV:-builderbench_creative_2_task3}
SEED=${SEED:-0}
NUM_STEPS=${NUM_STEPS:-200000000}
LOG_DIR="${LOG_DIR:-logs/ppo_builderbench_creative2_task3_e1024_pd/}"
PPO_NUM_ENVS=${PPO_NUM_ENVS:-1024}

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS}"
echo "log_dir=${LOG_DIR} num_envs=${PPO_NUM_ENVS}"
echo "builderbench_use_pd=true pd_duration=5 pd_policy_obs=pos+select"
echo "steps_per_iter=$((PPO_NUM_ENVS * 40))"
echo "num_iterations=$((NUM_STEPS / (PPO_NUM_ENVS * 40)))"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --ppo_ent_coef=0.05 \
  --ppo_actor_min_std=0.01 \
  --ppo_discount=0.99 \
  --ppo_clip_coef=0.2 \
  --ppo_checkpoint_interval=300 \
  --builderbench_use_pd=true \
  --builderbench_pd_duration=5 \
  --ppo_rollout_length=40 \
  --ppo_crl_steps_per_iter=20 \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=0 \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
