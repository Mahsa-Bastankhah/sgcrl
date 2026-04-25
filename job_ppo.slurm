#!/bin/bash
#SBATCH --job-name=ppo_crl
#SBATCH --output=slurm/ppo_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas
#SBATCH --array=0-0          # seeds; change range to run more

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

# Load CUDA — same logic as job_kappa_sac.slurm.
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

# LD_LIBRARY_PATH — identical order to job_kappa_sac.slurm.
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210

# ---------------------------------------------------------------------------
# Config — edit to taste.  ppo_contrastive.py is NOT Launchpad-based; it runs
# as a single process (single GPU).  PPO-specific knobs live in
# contrastive/config.py::ContrastiveConfig (ppo_num_envs, ppo_rollout_length,
# ppo_num_epochs, ppo_num_minibatches, ppo_clip_coef, ...) and can be edited
# there or overridden per-run by subclassing in a wrapper script.
# Use --sample_goals for random goals (else fixed_goal_dict in
# ppo_contrastive.py, same as lp_contrastive).
# ---------------------------------------------------------------------------
ENV=point_FourRooms
NUM_STEPS=1000000
LOG_DIR=logs/ppo_fourrooms/

# Offset the array-id so submitting `--array=0-2` produces seeds 3, 4, 5.
SEED_OFFSET=${SEED_OFFSET:-26}
SEED=$(( ${SLURM_ARRAY_TASK_ID:-0} + SEED_OFFSET ))

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  alg=ppo  seed=${SEED}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_HOME: ${CUDA_HOME}"
nvidia-smi

# Verify JAX sees the GPU.
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

# ---------------------------------------------------------------------------
# Run — standalone single-process PPO on φ·ψ rewards + CRL off-policy updates.
# ---------------------------------------------------------------------------
python -u ppo_contrastive.py \
    --env="${ENV}" \
    --seed="${SEED}" \
    --num_steps="${NUM_STEPS}" \
    --log_dir_path="${LOG_DIR}" \
    --norepr_norm
