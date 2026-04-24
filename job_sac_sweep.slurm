#!/bin/bash
#SBATCH --job-name=q_sac_sweep
#SBATCH --output=slurm/q_sac_sweep_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --account=seas
#SBATCH --array=0-2          # 4 runs: 2×2 grid of HER_VALUES × SPI_VALUES

# ---------------------------------------------------------------------------
# Sweep definition
#
#   her_aux_coef ∈ {10, 15}
#   samples_per_insert ∈ {64, 128}
#
# Task id mapping (row-major over the 2x2 grid):
#   0 -> her=10  spi=64
#   1 -> her=10  spi=128
#   2 -> her=15  spi=64
#   3 -> her=15  spi=128
# ---------------------------------------------------------------------------
HER_VALUES=(20 25)
SPI_VALUES=(128 256)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
HER_IDX=$(( TASK_ID / ${#SPI_VALUES[@]} ))
SPI_IDX=$(( TASK_ID % ${#SPI_VALUES[@]} ))
HER=${HER_VALUES[$HER_IDX]}
SPI=${SPI_VALUES[$SPI_IDX]}

# Fail-fast: catch a mismatch between --array and HER_VALUES/SPI_VALUES
# *before* wasting the JAX/mujoco startup cost only to die in flags
# parsing with a confusing `could not convert string to float: ''`.
GRID_SIZE=$(( ${#HER_VALUES[@]} * ${#SPI_VALUES[@]} ))
if [ "${TASK_ID}" -ge "${GRID_SIZE}" ] || [ -z "${HER}" ] || [ -z "${SPI}" ]; then
  echo "ERROR: array task ${TASK_ID} has no corresponding sweep cell."
  echo "  HER_VALUES=(${HER_VALUES[*]})  (${#HER_VALUES[@]} entries)"
  echo "  SPI_VALUES=(${SPI_VALUES[*]})  (${#SPI_VALUES[@]} entries)"
  echo "  grid_size=${GRID_SIZE}, but --array range goes up to ${TASK_ID}."
  echo "  Resolved HER='${HER}'  SPI='${SPI}'  (either empty = bad mapping)."
  echo "  Fix: adjust --array=0-\$((GRID_SIZE-1)) or edit the *_VALUES arrays."
  exit 2
fi

module purge

# Drop any broken inherited CUDA paths from the submit host.
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
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || true

if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit. Run 'module avail cudatoolkit' and edit this script."
  exit 1
fi

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
cd /n/fs/mislresearch/sgcrl
source .venv/bin/activate

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mb6458/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Cap per-job GPU mem at 0.25 × 46GB ≈ 11GB.  The actual steady-state
# usage for q_sac on sawyer_bin is ~0.7GB (measured on a live run), so
# this is ~15× headroom while leaving enough room for up to 3 jobs to
# share one GPU without colliding on startup allocations.  Past
# failures: array tasks 1 and 3 of job 2883846 both landed on neu306
# with MEM_FRACTION=0.8 each and died with CUDA_ERROR_ILLEGAL_ADDRESS.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENV=sawyer_bin
ALG=q_sac
NUM_STEPS=8000000

# Each sweep cell gets its own log subdir so the {alg}_{env}_{seed} folder
# created inside acme does not collide across cells.
SWEEP_TAG="her${HER}_spi${SPI}"
LOG_DIR="logs/q_sac_sweep/${SWEEP_TAG}/"

SEED_OFFSET=${SEED_OFFSET:-8}
SEED=$(( TASK_ID * 0 + SEED_OFFSET ))  # keep seed fixed across cells

RUN_DIR="${LOG_DIR}${ALG}_${ENV}_${SEED}"
mkdir -p "${RUN_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "sweep cell: HER=${HER}  SPI=${SPI}  tag=${SWEEP_TAG}"
echo "env=${ENV}  alg=${ALG}  seed=${SEED}"
echo "run_dir=${RUN_DIR}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_HOME: ${CUDA_HOME}"
nvidia-smi

# Sanity-check JAX sees the GPU.
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

# ---------------------------------------------------------------------------
# Dump the full sweep config for later inspection.
# Writes:
#   ${RUN_DIR}/sweep_config.txt   — flat key=value summary of this cell
#   ${RUN_DIR}/sweep_config.json  — machine-readable version
# ---------------------------------------------------------------------------
cat > "${RUN_DIR}/sweep_config.txt" <<EOF
# Generated $(date -Iseconds) by job_sac_sweep.slurm
slurm_job_id=${SLURM_JOB_ID}
slurm_array_task_id=${TASK_ID}
sweep_tag=${SWEEP_TAG}
env=${ENV}
alg=${ALG}
seed=${SEED}
num_steps=${NUM_STEPS}
log_dir=${LOG_DIR}
run_dir=${RUN_DIR}
q_actor_her_aux_coef=${HER}
samples_per_insert=${SPI}
hostname=$(hostname)
EOF

export SWEEP_SEED=${SEED} SWEEP_ENV=${ENV} SWEEP_ALG=${ALG} \
       SWEEP_NUM_STEPS=${NUM_STEPS} SWEEP_LOG_DIR=${LOG_DIR} \
       SWEEP_SPI=${SPI} SWEEP_HER=${HER} SWEEP_RUN_DIR=${RUN_DIR} \
       SWEEP_TAG=${SWEEP_TAG}

python -u -c "
import json, os, sys
sys.path.insert(0, '.')
import sgcrl_jax_acme_compat  # noqa: F401
from contrastive.config import ContrastiveConfig
params = dict(
    seed=int(os.environ['SWEEP_SEED']),
    env_name=os.environ['SWEEP_ENV'],
    alg_name=os.environ['SWEEP_ALG'],
    max_number_of_steps=int(os.environ['SWEEP_NUM_STEPS']),
    log_dir=os.environ['SWEEP_LOG_DIR'],
    samples_per_insert=float(os.environ['SWEEP_SPI']),
    q_actor_her_aux_coef=float(os.environ['SWEEP_HER']),
    use_cpc=True, use_q_repr=True, twin_q_repr=True,
    reward_shaping_mode='q',
)
cfg = ContrastiveConfig(**params)
as_dict = {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else str(v))
           for k, v in cfg.__dict__.items()}
as_dict['sweep_tag'] = os.environ['SWEEP_TAG']
as_dict['slurm_job_id'] = os.environ.get('SLURM_JOB_ID', '')
as_dict['slurm_array_task_id'] = os.environ.get('SLURM_ARRAY_TASK_ID', '')
with open(os.path.join(os.environ['SWEEP_RUN_DIR'], 'sweep_config.json'), 'w') as f:
    json.dump(as_dict, f, indent=2, default=str)
print('Wrote', os.path.join(os.environ['SWEEP_RUN_DIR'], 'sweep_config.json'))
"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python -u lp_contrastive.py \
    --lp_launch_type=local_mt \
    --env="${ENV}" \
    --alg="${ALG}" \
    --seed="${SEED}" \
    --num_steps="${NUM_STEPS}" \
    --log_dir_path="${LOG_DIR}" \
    --samples_per_insert="${SPI}" \
    --q_actor_her_aux_coef="${HER}"
