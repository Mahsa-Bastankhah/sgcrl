#!/bin/bash
#SBATCH --job-name=ppo_16r_cell_probe
#SBATCH --output=slurm/ppo_sixteenrooms_cell_probe_1m_%A_%a.log
#SBATCH --error=slurm/ppo_sixteenrooms_cell_probe_1m_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=seas
#SBATCH --array=0-4

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
  0) METHOD="nf_compact_small"; REPR_MODE="nf" ;;
  1) METHOD="crl";              REPR_MODE="crl" ;;
  2) METHOD="td3_logq";         REPR_MODE="td3" ;;
  3) METHOD="tdinfonce";        REPR_MODE="tdinfonce" ;;
  4) METHOD="nf_tiny";          REPR_MODE="nf" ;;
  *) echo "ERROR: SLURM_ARRAY_TASK_ID must be 0..4, got ${TASK_ID}"; exit 2 ;;
esac

ROOT=/n/fs/mislresearch/sgcrl
PYTHON="${ROOT}/.venv/bin/python"
ENV=point_SixteenRooms
SEED=0
NUM_STEPS="${NUM_STEPS:-1000000}"
LOG_ROOT="${LOG_ROOT:-logs/ppo_sixteenrooms_cell_probe_near2x0_far0x20_1m/${METHOD}}"
RUN_DIR="${LOG_ROOT}/ppo_${ENV}_${SEED}"
PROBE_CSV="${RUN_DIR}/cell_probe.csv"
PROBE_PLOT_DIR="${RUN_DIR}/cell_probe"

COMMON_ARGS=(
  "--env=${ENV}"
  "--seed=${SEED}"
  "--num_steps=${NUM_STEPS}"
  "--log_dir_path=${LOG_ROOT}"
  "--ppo_repr_mode=${REPR_MODE}"
  "--ppo_num_envs=4"
  "--ppo_rollout_length=512"
  "--ppo_crl_steps_per_iter=25"
  "--ppo_density_snapshot_interval=5"
  "--ppo_checkpoint_interval=0"
  "--ppo_eval_interval=10"
  "--noppo_use_external_reward"
  "--noppo_norm_obs"
  "--ppo_maze_cell_probe"
  "--ppo_maze_cell_probe_near_row=2"
  "--ppo_maze_cell_probe_near_col=0"
  "--ppo_maze_cell_probe_far_row=0"
  "--ppo_maze_cell_probe_far_col=20"
  "--hidden_layer_sizes=256,256,256,256,256,256"
)

METHOD_ARGS=()
case "${METHOD}" in
  nf_compact_small)
    METHOD_ARGS+=(
      "--nf_sa_num_layers=3"
      "--nf_sa_hidden=192"
      "--nf_rep_size=64"
      "--nf_num_blocks=6"
      "--nf_coupling_width=192"
    )
    ;;
  nf_tiny)
    METHOD_ARGS+=(
      "--nf_sa_num_layers=2"
      "--nf_sa_hidden=128"
      "--nf_rep_size=32"
      "--nf_num_blocks=4"
      "--nf_coupling_width=128"
    )
    ;;
  td3_logq)
    METHOD_ARGS+=("--ppo_td3_log_reward")
    ;;
esac

TRAIN_CMD=("${PYTHON}" -u ppo_contrastive.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}")
PLOT_CMD=(
  "${PYTHON}" -u scripts/plot_sixteenrooms_cell_probe.py
  "--probe-csv=${PROBE_CSV}"
  "--output-dir=${PROBE_PLOT_DIR}"
  "--method=${METHOD}"
)

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
  printf 'array_index=%s method=%s repr_mode=%s seed=%s\n' \
    "${TASK_ID}" "${METHOD}" "${REPR_MODE}" "${SEED}"
  printf 'train_command='; printf '%q ' "${TRAIN_CMD[@]}"; printf '\n'
  printf 'plot_command='; printf '%q ' "${PLOT_CMD[@]}"; printf '\n'
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
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} task=${TASK_ID}"
echo "method=${METHOD} repr_mode=${REPR_MODE} env=${ENV} seed=${SEED}"
echo "E=4 T=512 num_steps=${NUM_STEPS} density_updates_per_iter=25"
echo "cell_probe=true near=(2,0) far=(0,20) reward_interval=1"
echo "visit_count_interval=every_transition snapshot_interval=5"
echo "checkpoint_interval=0 ppo_eval_interval=10"
echo "external_reward=false obs_norm=false nf_state_only=false"
echo "run_dir=${RUN_DIR}"
nvidia-smi || true
"${PYTHON}" -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

"${TRAIN_CMD[@]}"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
"${PLOT_CMD[@]}"
echo "Completed training and cell-probe plot: ${PROBE_PLOT_DIR}"
