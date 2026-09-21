#!/bin/bash
# CPU-only watch for job 3812404 NF task-goal z / log_det. Resume-safe.
set -euo pipefail
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3
RUN_DIR="${RUN_DIR:-logs/final_runs/visualizations/ppo_builderbench_creative4_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep50_20m_ckpt20_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_warp_logp_s0/ppo_builderbench_creative_4_task2_0}"
JOB_ID="${JOB_ID:-3812404}"
MAX_WATCH_SEC="${MAX_WATCH_SEC:-14400}"
POLL_SEC="${POLL_SEC:-30}"
exec python -u scripts/probe_nf_ckpt_task_goal_z.py \
  --run_dir="$RUN_DIR" \
  --watch \
  --job-id="$JOB_ID" \
  --poll-sec="$POLL_SEC" \
  --max-watch-sec="$MAX_WATCH_SEC"
