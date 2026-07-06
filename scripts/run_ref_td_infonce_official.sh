#!/usr/bin/env bash
# Run upstream TD-InfoNCE exactly as README.md (fetch_reach online GCRL).
#
# Usage:
#   bash scripts/run_ref_td_infonce_official.sh
#   ENV_NAME=fetch_push MAX_STEPS=1000000 bash scripts/run_ref_td_infonce_official.sh
#   DEBUG=1 bash scripts/run_ref_td_infonce_official.sh   # short smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD_ROOT="${ROOT}/external/td_infonce"

ENV_NAME="${ENV_NAME:-fetch_reach}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-500000}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/ref_td_infonce_${ENV_NAME}}"
LP_LAUNCH="${LP_LAUNCH:-local_mp}"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT}:${TD_ROOT}:${PYTHONPATH:-}"
export PYTHONSTARTUP="${ROOT}/scripts/ref_td_infonce_python_startup.py"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

# Official README tasks use OpenAI Fetch (gym_robotics + mujoco-py).
python - <<'PY' || pip install 'gym-robotics==1.0.1' -q
import gym_robotics  # noqa: F401
PY

cd "${TD_ROOT}"

ARGS=(
  --env_name="${ENV_NAME}"
  --max_number_of_steps="${MAX_STEPS}"
  --seed="${SEED}"
  --lp_launch_type="${LP_LAUNCH}"
  --exp_log_dir="${LOG_DIR}"
  --exp_log_dir_add_uid=false
  --wlog_vectors=true
)

if [ "${DEBUG:-0}" = "1" ]; then
  ARGS+=(--debug)
fi

echo "[ref_official] env=${ENV_NAME} seed=${SEED} steps=${MAX_STEPS}"
echo "[ref_official] log_dir=${LOG_DIR}"
echo "[ref_official] lp_launch_type=${LP_LAUNCH}"

exec python -u lp_td_infonce.py "${ARGS[@]}"
