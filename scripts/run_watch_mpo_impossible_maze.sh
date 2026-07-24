#!/usr/bin/env bash
# Login-node daemon: MPO point_Impossible trajectory plots for new ckpts.
# Renders on CPU via scripts/mpo_rollout_maze.py (no GPU jobs).
#
# Usage:
#   scripts/run_watch_mpo_impossible_maze.sh          # daemon, 5 min
#   scripts/run_watch_mpo_impossible_maze.sh 300      # daemon, 5 min
#   scripts/run_watch_mpo_impossible_maze.sh once     # single scan
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm plots/mpo_impossible_uniform_neg

ARG="${1:-300}"
LOG="${ROOT}/slurm/watch_mpo_impossible_maze.log"
PID_FILE="${ROOT}/slurm/watch_mpo_impossible_maze.pid"

# Prefer training venv (MPO + maze deps).
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/watch_mpo_impossible_maze.py --once
fi

if [[ -f "$PID_FILE" ]]; then
  old="$(cat "$PID_FILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "stopping previous watcher pid=${old}"
    kill "$old" 2>/dev/null || true
    sleep 1
  fi
fi

INTERVAL="$ARG"
nohup python -u scripts/watch_mpo_impossible_maze.py \
  --watch_interval "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE")  interval=${INTERVAL}s  log=${LOG}"
