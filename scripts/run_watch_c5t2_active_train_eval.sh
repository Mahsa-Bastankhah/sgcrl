#!/usr/bin/env bash
# Login-node CPU watcher: refresh creative5_task2 active_train_eval success plots.
#
# Overlays pinned NF baselines + live tiny/compact/large NF norand/nopermute runs onto:
#   figs/builderbench/active_train_eval/creative5_task2_train_success1000.png
#   figs/builderbench/active_train_eval/creative5_task2_eval_success.png
#
# No GPU / no rollouts — parses learner/eval CSV + slurm stdout only.
#
# Usage:
#   scripts/run_watch_c5t2_active_train_eval.sh          # daemon, 15m
#   scripts/run_watch_c5t2_active_train_eval.sh 1800     # daemon, 30m
#   scripts/run_watch_c5t2_active_train_eval.sh once     # single refresh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm figs/builderbench/active_train_eval

ARG="${1:-900}"
LOG="${ROOT}/slurm/watch_c5t2_active_train_eval.log"
PIDFILE="${ROOT}/slurm/watch_c5t2_active_train_eval.pid"

# Prefer the BuilderBench conda env when available (matplotlib).
if [ -f /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
  conda activate sgcrl_builderbench 2>/dev/null || true
fi

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/plot_c5t2_active_train_eval.py
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
nohup python -u scripts/plot_c5t2_active_train_eval.py \
  --watch "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  interval=${INTERVAL}s  log=${LOG}"
echo "plots → ${ROOT}/figs/builderbench/active_train_eval/creative5_task2_{train_success1000,eval_success}.png"
