#!/bin/bash
#SBATCH --job-name=mpo_c3t1_tdg
#SBATCH --output=slurm/mpo_crl_builderbench_creative3_task1_e1024_pd_td_critic_%A_%a.log
#SBATCH --error=slurm/mpo_crl_builderbench_creative3_task1_e1024_pd_td_critic_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=seas
#SBATCH --array=0

# Online MPO-CRL + fixed-goal scalar TD critic G(s,a) on BuilderBench
# creative-3-task1. Same compute as the direct-CRL MPO c3t1 job, with:
#   --mpo_use_td_critic=true
#   reward = target phi(s,a)·psi(g*)
#   reward standardized with EMA moments (rate 0.001)
#   expected-SARSA bootstrap with 20 target-policy actions
#   target G Polyak-updated every step (online weight 0.005)
#   G updates = policy updates (800/iter)
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

ENV=${ENV:-builderbench_creative_3_task1}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
NUM_ENVS=${NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-10000000}
ROLLOUT_T=${ROLLOUT_T:-50}
CRL_UPDATES=${CRL_UPDATES:-25}
POLICY_UPDATES=${POLICY_UPDATES:-800}
POLICY_BATCH=${POLICY_BATCH:-2048}
REPR_TARGET_RATE=${REPR_TARGET_RATE:-0.005}
# Online eval is disabled; evaluate later from checkpoints.
EVAL_INTERVAL=${EVAL_INTERVAL:-0}
EVAL_EPISODES=${EVAL_EPISODES:-8}
DISCOUNT=0.99
CRITIC_LR=${CRITIC_LR:-1e-4}
CRITIC_TARGET_PERIOD=${CRITIC_TARGET_PERIOD:-0}
CRITIC_TARGET_RATE=${CRITIC_TARGET_RATE:-0.005}
REWARD_NORM_RATE=${REWARD_NORM_RATE:-0.001}
BOOTSTRAP_ACTIONS=${BOOTSTRAP_ACTIONS:-20}

LOG_DIR="${LOG_DIR:-logs/mpo_crl_builderbench_creative3_task1_e1024_pd_td_critic/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS}"
echo "log_dir=${LOG_DIR} num_envs=${NUM_ENVS}"
echo "max_replay_size=${MAX_REPLAY_SIZE}"
echo "builderbench_use_pd=true pd_duration=5"
echo "T=${ROLLOUT_T} policy_updates=${POLICY_UPDATES} batch=${POLICY_BATCH} crl_updates=${CRL_UPDATES}"
echo "SPI=$((POLICY_UPDATES * POLICY_BATCH / (NUM_ENVS * ROLLOUT_T)))"
echo "repr_target_update_rate=${REPR_TARGET_RATE}"
echo "td_critic=true critic_lr=${CRITIC_LR} critic_target_period=${CRITIC_TARGET_PERIOD} critic_target_rate=${CRITIC_TARGET_RATE}"
echo "reward_norm=true reward_norm_rate=${REWARD_NORM_RATE} bootstrap_actions=${BOOTSTRAP_ACTIONS}"
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

GPU_DMON_LOG="slurm/gpu_dmon_mpo_c3t1_td_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
nvidia-smi dmon -s pucm -d 1 -o DT > "${GPU_DMON_LOG}" 2>&1 &
GPU_DMON_PID=$!

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
  --mpo_use_td_critic=true \
  --mpo_critic_learning_rate="${CRITIC_LR}" \
  --mpo_critic_grad_norm_clip=40.0 \
  --mpo_critic_hidden_sizes="512,512,256" \
  --mpo_critic_target_update_period="${CRITIC_TARGET_PERIOD}" \
  --mpo_critic_target_update_rate="${CRITIC_TARGET_RATE}" \
  --mpo_bootstrap_action_samples="${BOOTSTRAP_ACTIONS}" \
  --mpo_normalize_critic_reward=true \
  --mpo_critic_reward_norm_rate="${REWARD_NORM_RATE}" \
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
kill "${GPU_DMON_PID}" 2>/dev/null || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
