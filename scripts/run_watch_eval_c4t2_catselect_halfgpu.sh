#!/usr/bin/env bash
# Login-node daemon: checkpoint-eval watcher for c4t2 CRL/NF catselect runs
# that have eval_interval=0. Submits light 1-GPU oneshots (≤55 min, 16G).
# (Fractional --gpus-per-task=0.5 is accepted here but allocates no device.)
#
# Watched log roots:
#   logs/ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_catselect/
#   logs/ppo_builderbench_creative4_task2_e1024_pd_nf_tau05_catselect/
#
# Usage:
#   scripts/run_watch_eval_c4t2_catselect_halfgpu.sh          # daemon, 5 min
#   scripts/run_watch_eval_c4t2_catselect_halfgpu.sh 180      # daemon, 3 min
#   scripts/run_watch_eval_c4t2_catselect_halfgpu.sh once     # single scan
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm figs/builderbench/checkpoint_eval figs/builderbench/active_runs

ARG="${1:-300}"
LOG="${ROOT}/slurm/watch_eval_c4t2_catselect_halfgpu.log"
PID_FILE="${ROOT}/slurm/watch_eval_c4t2_catselect_halfgpu.pid"
STATE="figs/.watch_eval_success_c4t2_catselect_state.json"
JOB="jobs/job_bb_ckpt_eval_oneshot_halfgpu.slurm"

COMMON=(
  --threshold 0
  --state_file "$STATE"
  --bb_eval_job "$JOB"
  --include_any creative4_task2_e1024_pd_crl_tau05_catselect
  --include_any creative4_task2_e1024_pd_nf_tau05_catselect
)

if [[ "$ARG" == "once" ]]; then
  exec python -u scripts/watch_eval_success_updates.py --once "${COMMON[@]}"
fi

# Do not run this as a SLURM GPU watcher — login node only.
nohup python -u scripts/watch_eval_success_updates.py \
  --watch_interval "$ARG" \
  "${COMMON[@]}" \
  >>"$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE")  interval=${ARG}s  log=${LOG}"
