#!/bin/bash
#SBATCH --job-name=disc_bin_z3to6
#SBATCH --output=slurm/final_metaworld_runs/discover_sawyer_bin_e4_40m_safeinit_z3to6_ucbstd0_s%a_%A.log
#SBATCH --error=slurm/final_metaworld_runs/discover_sawyer_bin_e4_40m_safeinit_z3to6_ucbstd0_s%a_%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-1

# DISCOVER Sawyer bin, same recipe as job_discover_sawyer_bin_e4_40m_rand_ucbstd0_15h.slurm
# except orange NF init: object XY random, TCP XY on object, z Uniform(3–6cm),
# gripper 0.4, bounded/rejection servo.
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

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
ENV=sawyer_bin
NUM_STEPS=${NUM_STEPS:-40000000}
NUM_ENVS=${NUM_ENVS:-4}
BATCH=${BATCH:-150}
MULT=${MULT:-38}
EVAL_INTERVAL=${EVAL_INTERVAL:-17}
LOG_DIR="${LOG_DIR:-logs/final_metaworld_runs/discover_sawyer_bin_e4_40m_safeinit_z3to6_ucbstd0/}"

mkdir -p slurm/final_metaworld_runs "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} log_dir=${LOG_DIR}"
echo "num_envs=${NUM_ENVS} batch_size=${BATCH} train_step_multiplier=${MULT} UTD=38/150≈0.25"
echo "randomize_init=true safe_grasp=true tcp_z=U(3-6cm) eval_interval=${EVAL_INTERVAL}"
echo "discover: TD3+HER+UCB weights=[mean_ini=0,mean_tar=1,std_ini=0,std_tar=0]"
echo "early_stop_eval_success=0.9"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u baseline-agents/discover/train_sawyer.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num-timesteps="${NUM_STEPS}" \
  --num-envs="${NUM_ENVS}" \
  --num-eval-envs="${NUM_ENVS}" \
  --batch-size="${BATCH}" \
  --train-step-multiplier="${MULT}" \
  --log-dir="${LOG_DIR}" \
  --eval-interval="${EVAL_INTERVAL}" \
  --checkpoint-interval=200 \
  --ucb-std-ini=0 \
  --ucb-std-tar=0 \
  --randomize-init \
  --bin-safe-grasp-reset \
  --bin-randomize-tcp-z \
  --early-stop-eval-success=0.9
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
