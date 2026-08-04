#!/bin/bash
#SBATCH --job-name=xyz_reeval_axyood
#SBATCH --output=slurm/xyz_reeval_axyood_%j.log
#SBATCH --error=slurm/xyz_reeval_axyood_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=seas

# Re-score axy-OOD cat-acc + density probes with a_x,a_y ~ U[2,4]
# (train used U[-1,1]), then refresh comparison plots.
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

LOG_ROOT="${LOG_ROOT:-logs/xyz_density_az0_g05_100m}"
FIG_DIR="${FIG_DIR:-figs/xyz_density_az0_g05_100m}"
AXY_LOW="${AXY_LOW:-2}"
AXY_HIGH="${AXY_HIGH:-4}"
MODES="${MODES:-crl,nf,nf_compact,td3,fm,tdinfonce,nf_tiny}"

mkdir -p slurm "${FIG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "log_root=${LOG_ROOT} out_dir=${FIG_DIR}"
echo "axy_ood eval: a_x,a_y ~ U[${AXY_LOW},${AXY_HIGH}], a_z=0"
echo "modes=${MODES}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX', jax.default_backend(), jax.devices())"

set -e

echo "=== reeval axy-OOD cat-acc + density probes ==="
python -u xyz_density/reeval_state_ood.py \
  --log_root="${LOG_ROOT}" \
  --target=axy_ood \
  --axy_ood_low="${AXY_LOW}" \
  --axy_ood_high="${AXY_HIGH}" \
  --discount=0.5 \
  --val_batch_size=256 \
  --val_num_envs=128 \
  --noise_std=0.01 \
  --modes="${MODES}"

echo "=== plot cat-acc + OOD density probes ==="
python -u xyz_density/plot_results.py \
  --log_root="${LOG_ROOT}" \
  --out_dir="${FIG_DIR}"

echo "=== done ==="
pkill -P $$ 2>/dev/null || true
