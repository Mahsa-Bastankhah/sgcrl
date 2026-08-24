#!/bin/bash
#SBATCH --job-name=mpo_bb_c7t2
#SBATCH --output=slurm/mpo_crl_builderbench_creative7_task2_e1024_pd_%A_%a.log
#SBATCH --error=slurm/mpo_crl_builderbench_creative7_task2_e1024_pd_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-1

# Online MPO-CRL on BuilderBench creative-7-task2. Same hyperparameters as
# jobs/job_mpo_crl_builderbench_creative3_task2_e1024_pd.slurm (plain Gaussian
# MPO actor, no catselect). T scaled to PD macro ep_len for 7 cubes:
#   mj_len=100+7*50=450, pd_duration=5 → T=90.
# SPI: 1440 × 2048 / (1024×90) = 32. CRL updates = T/2 = 45.
# 200M env steps, 2 seeds. 128G (c1t1 OOM'd at 64G).
unset LD_PRELOAD
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

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

ENV=${ENV:-builderbench_creative_7_task2}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
NUM_ENVS=${NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-10000000}
ROLLOUT_T=${ROLLOUT_T:-90}
CRL_UPDATES=${CRL_UPDATES:-45}
POLICY_UPDATES=${POLICY_UPDATES:-1440}
POLICY_BATCH=${POLICY_BATCH:-2048}
REPR_TARGET_RATE=${REPR_TARGET_RATE:-0.005}
EVAL_INTERVAL=${EVAL_INTERVAL:-30}
EVAL_EPISODES=${EVAL_EPISODES:-8}
DISCOUNT=0.99

LOG_DIR="${LOG_DIR:-logs/mpo_crl_builderbench_creative7_task2_e1024_pd_fast_succ/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS}"
echo "log_dir=${LOG_DIR} num_envs=${NUM_ENVS}"
echo "max_replay_size=${MAX_REPLAY_SIZE}"
echo "builderbench_use_pd=true pd_duration=5"
echo "T=${ROLLOUT_T} policy_updates=${POLICY_UPDATES} batch=${POLICY_BATCH} crl_updates=${CRL_UPDATES}"
echo "SPI=$((POLICY_UPDATES * POLICY_BATCH / (NUM_ENVS * ROLLOUT_T)))"
echo "repr_target_update_rate=${REPR_TARGET_RATE}"
echo "eval_interval=${EVAL_INTERVAL} eval_episodes=${EVAL_EPISODES}"
echo "steps_per_iter=$((NUM_ENVS * ROLLOUT_T))"
echo "num_iterations=$((NUM_STEPS / (NUM_ENVS * ROLLOUT_T)))"
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS:-<unset>} CUDA_HOME=${CUDA_HOME:-<unset>}"
python -u -c "
import sgcrl_jax_acme_compat
import jax
backend = jax.default_backend()
print('JAX backend:', backend)
print('JAX devices:', jax.devices())
if backend != 'gpu':
  raise SystemExit(
      f'ERROR: expected JAX backend gpu, got {backend!r}. '
      'Refusing to train on CPU (would be ~40x slower).'
  )
"

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
  --mpo_min_replay_size=10000 \
  --mpo_num_action_samples=20 \
  --discount="${DISCOUNT}" \
  --mpo_repr_target_update_rate="${REPR_TARGET_RATE}" \
  --mpo_target_update_period=100 \
  --mpo_policy_learning_rate=3e-4 \
  --mpo_dual_learning_rate=1e-2 \
  --mpo_policy_hidden_sizes="256,256,256,256,256,256" \
  --hidden_layer_sizes="256,256,256,256,256,256" \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --builderbench_use_pd=true \
  --builderbench_pd_duration=5 \
  --mpo_eval_interval="${EVAL_INTERVAL}" \
  --mpo_eval_episodes="${EVAL_EPISODES}" \
  --mpo_checkpoint_interval=150 \
  --mpo_checkpoint_keep_last=5
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
