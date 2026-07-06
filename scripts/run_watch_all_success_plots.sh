#!/usr/bin/env bash
# Auto-refresh all main success figures as Slurm runs accumulate eval data.
# Usage: scripts/run_watch_all_success_plots.sh [interval_sec]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
INTERVAL="${1:-300}"
LOG="${ROOT}/slurm/plot_watch_all.log"
mkdir -p "${ROOT}/slurm"
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

while true; do
  echo "=== all-plots cycle @ $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"
  python -u scripts/plot_crl_gauss_nf_success.py >>"$LOG" 2>&1 || true
  python -u plots/plot_crl_tau_sweep.py >>"$LOG" 2>&1 || true
  python -u plots/plot_td_infonce_experiments.py >>"$LOG" 2>&1 || true
  sleep "$INTERVAL"
done
