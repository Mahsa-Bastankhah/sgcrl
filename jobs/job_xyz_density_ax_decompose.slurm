#!/bin/bash
#SBATCH --job-name=xyz_axdec
#SBATCH --output=slurm/xyz_density_ax_decompose_%A_%a.log
#SBATCH --error=slurm/xyz_density_ax_decompose_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas
#SBATCH --array=0-7

# Ax-decompose OOD eval on existing az0 (joint XY) checkpoints.
# Sweep a_x with a_y=0; score log p(x+0.5,y+0.5|s,a); cat-acc if argmax a_x=0.5.
# Array: 0=crl,1=nf,2=nf_compact,3=nf_tiny,4=nf_td,5=td3,6=fm,7=tdinfonce
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

MODES=(crl nf nf_compact nf_tiny nf_td td3 fm tdinfonce)
MODE=${MODE:-${MODES[${SLURM_ARRAY_TASK_ID:-0}]}}
LOG_ROOT="${LOG_ROOT:-logs/xyz_density_az0_g05_100m}"
OUT_DIR="${OUT_DIR:-figs/xyz_density_az0_g05_100m}"
DELTA=${DELTA:-0.5}
N_STATES=${N_STATES:-256}
N_AX=${N_AX:-21}

mkdir -p slurm "${OUT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID}"
echo "mode=${MODE} log_root=${LOG_ROOT} out_dir=${OUT_DIR}"
echo "delta=${DELTA} n_states=${N_STATES} n_ax=${N_AX} (ay=0, GT ax=${DELTA})"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX', jax.default_backend(), jax.devices())"

python -u xyz_density/eval_ax_decompose.py \
  --log_root="${LOG_ROOT}" \
  --out_dir="${OUT_DIR}" \
  --modes="${MODE}" \
  --delta="${DELTA}" \
  --n_states="${N_STATES}" \
  --n_ax="${N_AX}" \
  --noise_std=0.01 \
  --seed=0

# Refresh combined plot from whatever modes have finished.
python -u xyz_density/eval_ax_decompose.py \
  --log_root="${LOG_ROOT}" \
  --out_dir="${OUT_DIR}" \
  --n_ax="${N_AX}" \
  --delta="${DELTA}" \
  --plot_only || true

PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
