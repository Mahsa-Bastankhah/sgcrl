#!/bin/bash
# Blue c8t2 dgr + gated full-iter KL cancel. Seeds 0-1 via array.
# Usage: bash jobs/final_runs/submit_c8t2_dgr_klrb.sh
set -euo pipefail
cd /n/fs/mislresearch/sgcrl
mkdir -p slurm/final_runs logs/final_runs
sbatch jobs/final_runs/job_c8t2_dgr_klrb.slurm
