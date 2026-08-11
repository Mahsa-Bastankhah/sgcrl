#!/bin/bash
#SBATCH --job-name=xyz_reeval_nftd
#SBATCH --output=slurm/xyz_reeval_nf_td_%j.log
#SBATCH --error=slurm/xyz_reeval_nf_td_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=seas

# After TD-NF training: run all three OOD reevals for nf_td only, then
# refresh comparison plots + wide state_ood reprobes into
# figs/xyz_density_az0_g05_100m.
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
OOD_HIGH_XY="${OOD_HIGH_XY:-15}"
AXY_LOW="${AXY_LOW:-2}"
AXY_HIGH="${AXY_HIGH:-4}"
HALF_WIDTHS="${HALF_WIDTHS:-2,10}"
MODES="${MODES:-nf_td}"

mkdir -p slurm "${FIG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "log_root=${LOG_ROOT} out_dir=${FIG_DIR} modes=${MODES}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX', jax.default_backend(), jax.devices())"

set -e

echo "=== reeval state-OOD cat-acc + probes at ood_high_xy=${OOD_HIGH_XY} ==="
python -u xyz_density/reeval_state_ood.py \
  --log_root="${LOG_ROOT}" \
  --target=state_ood \
  --ood_high_xy="${OOD_HIGH_XY}" \
  --discount=0.5 \
  --val_batch_size=256 \
  --val_num_envs=128 \
  --noise_std=0.01 \
  --modes="${MODES}"

echo "=== reeval az-OOD cat-acc (a ~ U[-1,1]^3) ==="
python -u xyz_density/reeval_state_ood.py \
  --log_root="${LOG_ROOT}" \
  --target=az_ood \
  --discount=0.5 \
  --val_batch_size=256 \
  --val_num_envs=128 \
  --noise_std=0.01 \
  --no_probes \
  --modes="${MODES}"

echo "=== reeval axy-OOD cat-acc + probes (a_x,a_y ~ U[${AXY_LOW},${AXY_HIGH}]) ==="
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

echo "=== plot_results (all modes incl. nf_td) ==="
python -u xyz_density/plot_results.py \
  --log_root="${LOG_ROOT}" \
  --out_dir="${FIG_DIR}" \
  --batch_size=256

echo "=== wide reprobe state_ood ==="
IFS=',' read -r -a _HW_ARR <<< "${HALF_WIDTHS}"
for HALF_WIDTH in "${_HW_ARR[@]}"; do
  HALF_WIDTH="$(echo "${HALF_WIDTH}" | tr -d '[:space:]')"
  [ -z "${HALF_WIDTH}" ] && continue
  if awk "BEGIN {exit !(${HALF_WIDTH} == 2)}"; then
    N_GRID=81
  else
    N_GRID=201
  fi
  echo "=== reprobe half_width=${HALF_WIDTH} n_grid=${N_GRID} ==="
  python -u xyz_density/reprobe_wide.py \
    --log_root="${LOG_ROOT}" \
    --out_dir="${FIG_DIR}" \
    --half_width="${HALF_WIDTH}" \
    --n_grid="${N_GRID}" \
    --max_panels=6 \
    --noise_std=0.01 \
    --ood_high_xy="${OOD_HIGH_XY}"
done

echo "=== done ==="
pkill -P $$ 2>/dev/null || true
