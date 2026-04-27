#!/usr/bin/env bash
# Auto-refresh FourRooms success_1000 plot (Slurm whitelist + PPO-knob grouping).
# Usage: from repo root, or:  scripts/run_watch_fourrooms_success_plot.sh [interval_sec]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
INTERVAL="${1:-90}"
exec python scripts/plot_success_1000_by_config.py \
  --output plots/success_1000_fourrooms_ppo_knob_groups.png \
  --title 'FourRooms: success_1000 vs PPO knobs (live refresh)' \
  --watch \
  --watch_interval "$INTERVAL"
