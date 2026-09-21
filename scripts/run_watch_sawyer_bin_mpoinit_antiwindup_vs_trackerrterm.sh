#!/usr/bin/env bash
# Login-node CPU watcher: refresh dedicated anti-windup vs trackerr-term plot.
#
# No GPU / no rollouts — parses logs/eval/logs.csv only.
# Does NOT touch the big baseline figure watcher.
#
# Usage:
#   scripts/run_watch_sawyer_bin_mpoinit_antiwindup_vs_trackerrterm.sh
#   scripts/run_watch_sawyer_bin_mpoinit_antiwindup_vs_trackerrterm.sh 1800
#   scripts/run_watch_sawyer_bin_mpoinit_antiwindup_vs_trackerrterm.sh once
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm figs/metaworld/final_metaworld_runs

ARG="${1:-900}"
LOG="${ROOT}/slurm/watch_sawyer_bin_mpoinit_antiwindup_vs_trackerrterm.log"
PIDFILE="${ROOT}/slurm/watch_sawyer_bin_mpoinit_antiwindup_vs_trackerrterm.pid"
PLOT="scripts/plot_sawyer_bin_mpoinit_antiwindup_vs_trackerrterm.py"
OUT="${ROOT}/figs/metaworld/final_metaworld_runs/sawyer_bin_mpoinit_selectiveantiwindup_vs_trackerrterm20cm.png"

if [ -f /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
  conda activate sgcrl_flow 2>/dev/null || \
    conda activate sgcrl_builderbench 2>/dev/null || true
fi

if [[ "$ARG" == "once" ]]; then
  exec python -u "$PLOT"
fi

INTERVAL="$ARG"
if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "stopping previous dedicated watcher pid=${old}"
    kill "$old" 2>/dev/null || true
    sleep 1
  fi
fi

# Do not run this as a SLURM GPU watcher — login node / CPU only.
nohup python -u "$PLOT" \
  --watch "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  interval=${INTERVAL}s  log=${LOG}"
echo "plot → ${OUT}"
