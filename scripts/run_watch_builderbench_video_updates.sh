#!/usr/bin/env bash
# Login-node daemon: BuilderBench videos for *currently running* jobs only.
# Submits short 1-GPU jobs that render only new checkpoints (≤25 min).
# Cube4 task2 videos are excluded by default (eval-only for that task).
#
# Usage:
#   scripts/run_watch_builderbench_video_updates.sh          # daemon, 5 min
#   scripts/run_watch_builderbench_video_updates.sh 180      # daemon, 3 min
#   scripts/run_watch_builderbench_video_updates.sh once     # single scan
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm videos/builderbench/active_runs

ARG="${1:-300}"
LOG="${ROOT}/slurm/watch_builderbench_video_updates.log"
EXCLUDE_ARGS=(--exclude creative4_task2)

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/watch_builderbench_video_updates.py --once "${EXCLUDE_ARGS[@]}"
fi

INTERVAL="$ARG"
# Do not run this as a SLURM GPU watcher — login node only.
nohup python -u scripts/watch_builderbench_video_updates.py \
  --watch_interval "$INTERVAL" \
  "${EXCLUDE_ARGS[@]}" \
  >>"$LOG" 2>&1 &
echo $! > "${ROOT}/slurm/watch_builderbench_video_updates.pid"
echo "started pid=$(cat "${ROOT}/slurm/watch_builderbench_video_updates.pid")  interval=${INTERVAL}s  exclude=creative4_task2  log=${LOG}"
