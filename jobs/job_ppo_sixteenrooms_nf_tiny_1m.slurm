#!/bin/bash
#SBATCH --job-name=ppo_16r_nf_tiny
#SBATCH --output=slurm/ppo_sixteenrooms_nf_tiny_1m_%j.log
#SBATCH --error=slurm/ppo_sixteenrooms_nf_tiny_1m_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=seas

set -euo pipefail

ROOT=/n/fs/mislresearch/sgcrl
PYTHON="${ROOT}/.venv/bin/python"
ENV=point_SixteenRooms
SEED=0
NUM_STEPS="${NUM_STEPS:-1000000}"
LOG_ROOT="${LOG_ROOT:-logs/ppo_sixteenrooms_density_diag_1m/nf_tiny}"
RUN_DIR="${LOG_ROOT}/ppo_${ENV}_${SEED}"
SNAPSHOT_DIR="${RUN_DIR}/density_snapshots"
FIXED_PLOT_DIR="${RUN_DIR}/density_evolution"
AUTOSCALE_PLOT_DIR="${RUN_DIR}/density_evolution_autoscale"

TRAIN_CMD=(
  "${PYTHON}" -u ppo_contrastive.py
  "--env=${ENV}"
  "--seed=${SEED}"
  "--num_steps=${NUM_STEPS}"
  "--log_dir_path=${LOG_ROOT}"
  "--ppo_repr_mode=nf"
  "--ppo_num_envs=4"
  "--ppo_rollout_length=512"
  "--ppo_crl_steps_per_iter=25"
  "--ppo_density_snapshot_interval=5"
  "--ppo_checkpoint_interval=0"
  "--ppo_eval_interval=10"
  "--noppo_use_external_reward"
  "--noppo_norm_obs"
  "--hidden_layer_sizes=256,256,256,256,256,256"
  "--nf_sa_num_layers=2"
  "--nf_sa_hidden=128"
  "--nf_rep_size=32"
  "--nf_num_blocks=4"
  "--nf_coupling_width=128"
)
FIXED_PLOT_CMD=(
  "${PYTHON}" -u scripts/plot_sixteenrooms_reward_evolution.py
  "--snapshot-dir=${SNAPSHOT_DIR}"
  "--output-dir=${FIXED_PLOT_DIR}"
  "--scale-mode=fixed"
  "--no-gif"
)
AUTOSCALE_PLOT_CMD=(
  "${PYTHON}" -u scripts/plot_sixteenrooms_reward_evolution.py
  "--snapshot-dir=${SNAPSHOT_DIR}"
  "--output-dir=${AUTOSCALE_PLOT_DIR}"
  "--scale-mode=frame_robust"
  "--no-gif"
)

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
  echo "method=nf_tiny repr_mode=nf seed=${SEED}"
  printf 'train_command='; printf '%q ' "${TRAIN_CMD[@]}"; printf '\n'
  printf 'fixed_plot_command='
  printf '%q ' "${FIXED_PLOT_CMD[@]}"; printf '\n'
  printf 'autoscale_plot_command='
  printf '%q ' "${AUTOSCALE_PLOT_CMD[@]}"; printf '\n'
  exit 0
fi

module purge
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

CUDA_LOADED=0
for CUDA_MODULE in cudatoolkit/12.6 cudatoolkit/12.5 cudatoolkit/12.4; do
  if module load "${CUDA_MODULE}" 2>/dev/null; then
    CUDA_LOADED=1
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    echo "Loaded CUDA module: ${CUDA_MODULE}"
    break
  fi
done
if [[ "${CUDA_LOADED}" != "1" ]]; then
  for CANDIDATE in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [[ -d "${CANDIDATE}/lib64" ]]; then
      CUDA_LOADED=1
      export CUDA_HOME="${CANDIDATE}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      echo "Using CUDA fallback: ${CUDA_HOME}"
      break
    fi
  done
fi
if [[ "${CUDA_LOADED}" != "1" || -z "${CUDA_HOME:-}" ]]; then
  echo "ERROR: no CUDA toolkit found"
  exit 1
fi

cd "${ROOT}"
source .venv/bin/activate
unset -f python python3 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export TF_CPP_MIN_LOG_LEVEL=2

mkdir -p slurm "${LOG_ROOT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "method=nf_tiny repr_mode=nf env=${ENV} seed=${SEED}"
echo "E=4 T=512 requested_steps=${NUM_STEPS} effective_steps=999424"
echo "density_updates_per_iter=25 snapshot_interval=5 checkpoint_interval=0"
echo "external_reward=false obs_norm=false nf_state_only=false"
echo "nf_tiny: sa2x128 r32 b4 w128"
echo "run_dir=${RUN_DIR}"
nvidia-smi || true
"${PYTHON}" -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

"${TRAIN_CMD[@]}"

# Plotting does not need the GPU. Both invocations write PNG sequences only.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
"${FIXED_PLOT_CMD[@]}"
"${AUTOSCALE_PLOT_CMD[@]}"
echo "Completed training and PNG plots:"
echo "  fixed=${FIXED_PLOT_DIR}"
echo "  autoscale=${AUTOSCALE_PLOT_DIR}"
