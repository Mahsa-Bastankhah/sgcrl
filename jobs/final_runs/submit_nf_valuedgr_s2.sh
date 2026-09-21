#!/bin/bash
# Third seed (array task 2) for paper NF valuedgr tasks that only have 0/1.
# 4h: the original 2:30 c4/c5 jobs stopped short of 200M.
# Usage: bash jobs/final_runs/submit_nf_valuedgr_s2.sh
set -euo pipefail
cd /n/fs/mislresearch/sgcrl
mkdir -p slurm/final_runs logs/final_runs

submit_s2() {
  local script=$1
  local tag=$2
  echo "sbatch ${tag}  array=2  time=4:00:00"
  sbatch \
    --job-name="${tag}" \
    --array=2 \
    --time=4:00:00 \
    "${script}"
}

submit_s2 jobs/final_runs/job_c3t1_dgr_valuedgr.slurm c3t1_vdgr_s2
submit_s2 jobs/final_runs/job_c4t1_dgr_valuedgr.slurm c4t1_vdgr_s2
submit_s2 jobs/final_runs/job_c4t2_dgr_valuedgr.slurm c4t2_vdgr_s2
submit_s2 jobs/final_runs/job_c5t2_dgr_valuedgr.slurm c5t2_vdgr_s2
