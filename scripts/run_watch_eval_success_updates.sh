#!/usr/bin/env bash
# Login-node daemon: refresh eval-success when >2 new checkpoints appear.
# Metaworld/Sawyer → replot from logs/eval/logs.csv (CPU).
# BuilderBench → short one-shot SLURM GPU eval job.
#
# Usage:
#   scripts/run_watch_eval_success_updates.sh          # daemon, 5 min
#   scripts/run_watch_eval_success_updates.sh 180      # daemon, 3 min
#   scripts/run_watch_eval_success_updates.sh once     # single scan
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm figs/metaworld figs/builderbench/checkpoint_eval

ARG="${1:-300}"
LOG="${ROOT}/slurm/watch_eval_success_updates.log"

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/watch_eval_success_updates.py --once
fi

INTERVAL="$ARG"
# Do not run this as a SLURM GPU watcher — login node only.
nohup python -u scripts/watch_eval_success_updates.py \
  --watch_interval "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "${ROOT}/slurm/watch_eval_success_updates.pid"
echo "started pid=$(cat "${ROOT}/slurm/watch_eval_success_updates.pid")  interval=${INTERVAL}s  log=${LOG}"
