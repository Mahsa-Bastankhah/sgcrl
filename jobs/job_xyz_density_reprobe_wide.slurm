#!/bin/bash
#SBATCH --job-name=xyz_reprobe_wide
#SBATCH --output=slurm/xyz_reprobe_wide_%j.log
#SBATCH --error=slurm/xyz_reprobe_wide_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# Re-score state_ood probes at ±10 and overwrite over-time / latest figs.
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
# Comma-separated half-widths: 2 → unsuffixed figs; 10 → *_hw10.png
HALF_WIDTHS="${HALF_WIDTHS:-2,10}"
OOD_HIGH_XY="${OOD_HIGH_XY:-15}"

mkdir -p slurm "${FIG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "log_root=${LOG_ROOT} out_dir=${FIG_DIR} half_widths=${HALF_WIDTHS} ood_high_xy=${OOD_HIGH_XY}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX', jax.default_backend(), jax.devices())"

PY_EXIT=0
IFS=',' read -r -a _HW_ARR <<< "${HALF_WIDTHS}"
for HALF_WIDTH in "${_HW_ARR[@]}"; do
  HALF_WIDTH="$(echo "${HALF_WIDTH}" | tr -d '[:space:]')"
  [ -z "${HALF_WIDTH}" ] && continue
  if awk "BEGIN {exit !(${HALF_WIDTH} == 2)}"; then
    N_GRID=81
  else
    N_GRID=201
  fi
  echo "=== reprobe half_width=${HALF_WIDTH} n_grid=${N_GRID} ood_high_xy=${OOD_HIGH_XY} ==="
  python -u xyz_density/reprobe_wide.py \
    --log_root="${LOG_ROOT}" \
    --out_dir="${FIG_DIR}" \
    --half_width="${HALF_WIDTH}" \
    --n_grid="${N_GRID}" \
    --max_panels=6 \
    --noise_std=0.01 \
    --ood_high_xy="${OOD_HIGH_XY}" || PY_EXIT=$?
done

pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
