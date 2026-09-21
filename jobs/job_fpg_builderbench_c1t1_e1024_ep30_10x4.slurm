#!/bin/bash
#SBATCH --job-name=fpg_bb_c1t1_e1024_ep30
#SBATCH --output=slurm/fpg_builderbench_c1t1_e1024_ep30_10x4_%j.log
#SBATCH --error=slurm/fpg_builderbench_c1t1_e1024_ep30_10x4_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --account=seas

# f-PG (obj=fkl) c1t1, E=1024: 1 episode, 1 KDE, 10x4 PPO epochs.
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
  module avail cudatoolkit 2>&1 || true
  ls -d /usr/local/cuda* 2>/dev/null || true
  exit 1
fi

cd /n/fs/mislresearch/f-pg
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench

source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=warp
export SGCRL_ROOT=/n/fs/mislresearch/sgcrl
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
export PYTHONPATH="/n/fs/mislresearch/f-pg:${SGCRL_ROOT}:${PYTHONPATH:-}"

SEED=${SEED:-0}
CONFIG=${CONFIG:-/n/fs/mislresearch/f-pg/configs/pg_builderbench_c1t1_e1024_ep30.yaml}
LOGDIR=/n/fs/mislresearch/f-pg/logs/builderbench_c1t1_e1024_pd_nopermute_fixedx01_fkl_ep30_10x4_200m

mkdir -p /n/fs/mislresearch/sgcrl/slurm
mkdir -p /n/fs/mislresearch/f-pg/logs
rm -rf "${LOGDIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "f-PG c1t1 E=1024  obj=fkl  30-step episode + 1 KDE + 10x4 PPO"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "config=${CONFIG} seed=${SEED}"
echo "fresh log dir ${LOGDIR}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"

nvidia-smi || true
python -u -c "import os, sgcrl_jax_acme_compat, warp, jax; print('warp', warp.__version__); print('BUILDERBENCH_MJX_IMPL', os.environ.get('BUILDERBENCH_MJX_IMPL')); print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u fpg/run_builderbench.py --config="${CONFIG}" --seed="${SEED}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
