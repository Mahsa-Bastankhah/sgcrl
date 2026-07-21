#!/usr/bin/env bash
# Login-node daemon: BuilderBench videos for *currently running* jobs only.
# Submits short 1-GPU jobs that render only new checkpoints (≤25 min).
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

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/watch_builderbench_video_updates.py --once
fi

INTERVAL="$ARG"
# Do not run this as a SLURM GPU watcher — login node only.
nohup python -u scripts/watch_builderbench_video_updates.py \
  --watch_interval "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "${ROOT}/slurm/watch_builderbench_video_updates.pid"
echo "started pid=$(cat "${ROOT}/slurm/watch_builderbench_video_updates.pid")  interval=${INTERVAL}s  log=${LOG}"
