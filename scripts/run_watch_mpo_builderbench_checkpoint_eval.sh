#!/usr/bin/env bash
# Login-node daemon: MPO BuilderBench checkpoint eval + success-vs-step plots.
# Submits short 1-GPU jobs for new checkpoints (≤55 min).
#
# Usage:
#   scripts/run_watch_mpo_builderbench_checkpoint_eval.sh          # daemon, 5 min
#   scripts/run_watch_mpo_builderbench_checkpoint_eval.sh 180      # daemon, 3 min
#   scripts/run_watch_mpo_builderbench_checkpoint_eval.sh once     # single scan
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm figs/builderbench/checkpoint_eval

ARG="${1:-300}"
LOG="${ROOT}/slurm/watch_mpo_builderbench_checkpoint_eval.log"
PID_FILE="${ROOT}/slurm/watch_mpo_builderbench_checkpoint_eval.pid"

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/watch_mpo_builderbench_checkpoint_eval.py --once
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
# Do not run this as a SLURM GPU watcher — login node only.
nohup python -u scripts/watch_mpo_builderbench_checkpoint_eval.py \
  --watch_interval "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE")  interval=${INTERVAL}s  log=${LOG}"
