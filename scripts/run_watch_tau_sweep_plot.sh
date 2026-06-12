#!/usr/bin/env bash
# Auto-refresh CRL tau sweep figure as Slurm runs accumulate eval data.
# Usage: scripts/run_watch_tau_sweep_plot.sh [interval_sec]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
INTERVAL="${1:-300}"
LOG="${ROOT}/slurm/plot_watch_tau_sweep.log"
mkdir -p "${ROOT}/slurm"
exec python -u plots/plot_crl_tau_sweep.py \
  --watch \
  --watch_interval "$INTERVAL" \
  >>"$LOG" 2>&1
