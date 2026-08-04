#!/usr/bin/env bash
# Login-node CPU watcher: refresh figs/builderbench/active_run/ from logs.
#
# Tracks scheduled c4t2 NF / c5t2 NF (mean±stderr) and c4t1 per-estimator
# overlays. No GPU / no rollouts — parses learner CSV + slurm stdout only.
#
# Usage:
#   scripts/run_watch_builderbench_active_run.sh          # daemon, 2h
#   scripts/run_watch_builderbench_active_run.sh 3600     # daemon, 1h
#   scripts/run_watch_builderbench_active_run.sh once     # single refresh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm figs/builderbench/active_run

ARG="${1:-7200}"
LOG="${ROOT}/slurm/watch_builderbench_active_run.log"
PIDFILE="${ROOT}/slurm/watch_builderbench_active_run.pid"

# Prefer the BuilderBench conda env when available (matplotlib).
if [ -f /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
  conda activate sgcrl_builderbench 2>/dev/null || true
fi

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/plot_builderbench_active_run.py
fi

INTERVAL="$ARG"
# Stop any previous watcher with the same pid file.
if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "stopping previous watcher pid=${old}"
    kill "$old" 2>/dev/null || true
    sleep 1
  fi
fi

# Do not run this as a SLURM GPU watcher — login node / CPU only.
nohup python -u scripts/plot_builderbench_active_run.py \
  --watch "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  interval=${INTERVAL}s  log=${LOG}"
echo "plots → ${ROOT}/figs/builderbench/active_run/"
