#!/bin/bash
#SBATCH --job-name=ppo_smoke_perf
#SBATCH --output=slurm/ppo_smoke_perf_fixes_%j.log
#SBATCH --error=slurm/ppo_smoke_perf_fixes_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=seas

# Smoke test for episode-flush + on-device reward/GAE/PPO changes.
# ~5 PPO iters at E=128 (JIT-heavy first iters; just checks no crash).
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

ENV=${ENV:-builderbench_creative_5_task2}
SEED=${SEED:-0}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-128}
ROLLOUT=${ROLLOUT:-50}
N_ITERS=${N_ITERS:-5}
NUM_STEPS=$((PPO_NUM_ENVS * ROLLOUT * N_ITERS))
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-200000}
LOG_DIR="${LOG_DIR:-logs/ppo_smoke_perf_fixes/}"

mkdir -p slurm "${LOG_DIR}"

echo "SMOKE: env=${ENV} seed=${SEED} E=${PPO_NUM_ENVS} T=${ROLLOUT} iters=${N_ITERS} steps=${NUM_STEPS}"
echo "log_dir=${LOG_DIR}"
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
  --ppo_checkpoint_interval=0 \
  --builderbench_use_pd=true \
  --builderbench_pd_duration=5 \
  --builderbench_permute_start_boxes=false \
  --builderbench_fixed_start_x=0.1 \
  --ppo_rollout_length="${ROLLOUT}" \
  --ppo_crl_steps_per_iter=5 \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=100000 \
  --ppo_eval_episodes=1 \
  --ppo_video_interval=0 \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --ppo_repr_mode=nf \
  --nf_state_only \
  --ppo_nf_reward_tau=0.5 \
  --nf_rep_size=64 \
  --nf_num_blocks=6 \
  --nf_coupling_width=256 \
  --nf_sa_hidden=256 \
  --nf_sa_num_layers=3 \
  --nf_grad_clip=1.0 \
  --nf_noise_std=0.05 \
  --nf_goal_std_min=0.02 \
  --nf_goal_enc_size=0 \
  --ppo_use_external_reward \
  --ppo_external_reward_scale=1 \
  --ppo_categorical_select \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
echo "SMOKE_EXIT=${PY_EXIT}"
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
