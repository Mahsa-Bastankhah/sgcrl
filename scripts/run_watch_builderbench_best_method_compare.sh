#!/usr/bin/env bash
# Login-node CPU watcher: refresh best NF/CRL/TD3/TD-InfoNCE compare plots.
#
# Rewrites:
#   figs/builderbench/active_train_eval/best_method_compare/
#     {task}_best_nf_crl_td3_tdinfonce_train_eval.png
#     overview_best_methods_train.png
#     summary.csv / ambiguities.md / picks.json / BEST_RUNS.md
#
# No GPU / no rollouts — parses learner/eval CSV + slurm stdout only.
#
# Usage:
#   scripts/run_watch_builderbench_best_method_compare.sh         # daemon, 30m
#   scripts/run_watch_builderbench_best_method_compare.sh 900     # daemon, 15m
#   scripts/run_watch_builderbench_best_method_compare.sh once    # single refresh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm figs/builderbench/active_train_eval/best_method_compare

ARG="${1:-1800}"
LOG="${ROOT}/slurm/watch_builderbench_best_method_compare.log"
PIDFILE="${ROOT}/slurm/watch_builderbench_best_method_compare.pid"

# Prefer the BuilderBench conda env when available (matplotlib).
if [ -f /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
  conda activate sgcrl_builderbench 2>/dev/null || true
fi

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/plot_builderbench_best_method_compare.py
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
nohup python -u scripts/plot_builderbench_best_method_compare.py \
  --watch "$INTERVAL" \
  >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")  interval=${INTERVAL}s  log=${LOG}"
echo "plots → ${ROOT}/figs/builderbench/active_train_eval/best_method_compare/"
