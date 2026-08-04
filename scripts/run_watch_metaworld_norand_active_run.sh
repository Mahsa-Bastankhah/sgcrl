#!/usr/bin/env bash
# Login-node CPU watcher: refresh figs/metaworld/active_run/ from eval CSVs.
#
# Plots Sawyer bin / peg / box norand CRL eval success (one PNG per env).
# No GPU / no rollouts — parses logs/*/logs/eval/logs.csv only.
#
# Usage:
#   scripts/run_watch_metaworld_norand_active_run.sh          # daemon, 1h
#   scripts/run_watch_metaworld_norand_active_run.sh 3600     # daemon, 1h
#   scripts/run_watch_metaworld_norand_active_run.sh once     # single refresh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm figs/metaworld/active_run

ARG="${1:-3600}"
LOG="${ROOT}/slurm/watch_metaworld_norand_active_run.log"
PIDFILE="${ROOT}/slurm/watch_metaworld_norand_active_run.pid"

# Prefer a conda env with matplotlib when available.
if [ -f /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
  conda activate sgcrl_flow 2>/dev/null || \
    conda activate sgcrl_builderbench 2>/dev/null || true
fi

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/plot_metaworld_norand_active_run.py
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
nohup python -u scripts/plot_metaworld_norand_active_run.py \
  --watch "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  interval=${INTERVAL}s  log=${LOG}"
echo "plots → ${ROOT}/figs/metaworld/active_run/"
