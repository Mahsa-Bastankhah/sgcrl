#!/bin/bash
# Four c7t2 stability ablations × seeds 0/1 (8 jobs).
# Usage: bash jobs/final_runs/submit_c7t2_stab.sh
set -euo pipefail
cd /n/fs/mislresearch/sgcrl
mkdir -p slurm/final_runs logs/final_runs

SCRIPT=jobs/final_runs/job_c7t2_stab.slurm

submit() {
  local variant=$1
  local mem=$2
  local tag="c7t2_stab_${variant}"
  local out="slurm/final_runs/${tag}_%A_%a.log"
  echo "sbatch ${tag}  mem=${mem}  (seeds 0-1)"
  sbatch \
    --job-name="${tag}" \
    --output="${out}" \
    --error="${out}" \
    --mem="${mem}" \
    --time=2:00:00 \
    --export=ALL,STAB_VARIANT="${variant}" \
    "${SCRIPT}"
}

submit replay100m 64G
submit replay50m 16G
submit ent05 16G
submit mixtaskg 16G
submit randx 16G
submit lamlr1e2 16G
submit klpen 16G
submit klpen_replaysub 16G
submit combo 16G
submit combo_tiny 16G
submit combo_freezenf 16G
submit combo_c20 16G
submit klpen_c20 16G
