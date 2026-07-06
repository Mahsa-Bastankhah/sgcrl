#!/usr/bin/env bash
# Auto-refresh user-vs-ref w_diag comparison + ref eval success.
# Usage: scripts/run_watch_w_diag_compare.sh [interval_sec]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
INTERVAL="${1:-30}"
LOG="${ROOT}/slurm/plot_watch_w_diag_compare.log"
mkdir -p "${ROOT}/slurm"
exec python -u scripts/watch_w_diag_compare_user_vs_ref.py \
  --watch \
  --poll_interval "$INTERVAL" \
  >>"$LOG" 2>&1
