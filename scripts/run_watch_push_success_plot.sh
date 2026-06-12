#!/usr/bin/env bash
# Auto-refresh Sawyer push success plots as Slurm runs accumulate eval data.
# Usage: scripts/run_watch_push_success_plot.sh [interval_sec]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
INTERVAL="${1:-300}"
LOG="${ROOT}/slurm/plot_watch_push.log"
mkdir -p "${ROOT}/slurm"
exec python -u scripts/plot_crl_gauss_nf_success.py \
  --tasks push push_nf \
  --watch \
  --watch_interval "$INTERVAL" \
  >>"$LOG" 2>&1
