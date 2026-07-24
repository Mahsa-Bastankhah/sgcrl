#!/usr/bin/env bash
# Login-node daemon: MPO BuilderBench videos for the four c2/c3 validation runs.
# Submits short 1-GPU jobs that render only new checkpoints (≤25 min).
#
# Usage:
#   scripts/run_watch_mpo_builderbench_video_updates.sh          # daemon, 3 min
#   scripts/run_watch_mpo_builderbench_video_updates.sh 180      # daemon, 3 min
#   scripts/run_watch_mpo_builderbench_video_updates.sh once     # single scan
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm videos/builderbench/mpo_active_runs

ARG="${1:-180}"
LOG="${ROOT}/slurm/watch_mpo_builderbench_video_updates.log"
PID_FILE="${ROOT}/slurm/watch_mpo_builderbench_video_updates.pid"

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/watch_mpo_builderbench_video_updates.py --once
fi

# Stop any previous watcher with the same pid file.
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
nohup python -u scripts/watch_mpo_builderbench_video_updates.py \
  --watch_interval "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE")  interval=${INTERVAL}s  log=${LOG}"
