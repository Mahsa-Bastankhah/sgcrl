#!/bin/bash
#SBATCH --job-name=xyz_density_extra
#SBATCH --output=slurm/xyz_density_extra_%A_%a.log
#SBATCH --error=slurm/xyz_density_extra_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=seas
#SBATCH --array=0-1

# xyz density probe extras: array 0=fm, 1=tdinfonce.
# Same protocol as job_xyz_density_probe.slurm (100M steps, a_z=0 train).
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
      echo "Fallback CUDA_HOME=${CUDA_HOME}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODES=(fm tdinfonce)
MODE=${MODE:-${MODES[${SLURM_ARRAY_TASK_ID:-0}]}}
SEED=${SEED:-0}
NUM_ENVS=${NUM_ENVS:-1024}
NUM_STEPS=${NUM_STEPS:-100000000}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-1000000}
LOG_ROOT="${LOG_ROOT:-logs/xyz_density_az0_g05_100m}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${MODE}/}"
FIG_DIR="${FIG_DIR:-figs/xyz_density_az0_g05_100m}"

mkdir -p slurm "${LOG_DIR}" "${FIG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID}"
echo "mode=${MODE} seed=${SEED} num_envs=${NUM_ENVS} num_steps=${NUM_STEPS}"
echo "zero_az=true  discount=0.5  ood_high_xy=15  val_every=5  probe_every=10"
echo "max_replay_size=${MAX_REPLAY_SIZE}"
echo "log_dir=${LOG_DIR}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX', jax.default_backend(), jax.devices())"

python -u xyz_density/train.py \
  --repr_mode="${MODE}" \
  --seed="${SEED}" \
  --num_envs="${NUM_ENVS}" \
  --episode_length=50 \
  --num_steps="${NUM_STEPS}" \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --min_replay_size=10000 \
  --batch_size=256 \
  --density_steps_per_iter=25 \
  --discount=0.5 \
  --noise_std=0.01 \
  --zero_az \
  --ood_high_xy=15 \
  --val_every=5 \
  --probe_every=10 \
  --val_num_envs=128 \
  --val_batch_size=256 \
  --log_dir="${LOG_DIR}" \
  --checkpoint_interval=100

# Refresh comparison plots so FM / TDInfoNCE join CRL/NF/TD3 curves.
python -u xyz_density/plot_results.py \
  --log_root="${LOG_ROOT}" \
  --out_dir="${FIG_DIR}" \
  --batch_size=256 || true

PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
