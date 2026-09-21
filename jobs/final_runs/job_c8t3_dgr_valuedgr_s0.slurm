#!/bin/bash
#SBATCH --job-name=c8t3_dgr_vdgr_s0
#SBATCH --output=slurm/final_runs/c8t3_dgr_valuedgr_s0_%j.log
#SBATCH --error=slurm/final_runs/c8t3_dgr_valuedgr_s0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --account=seas

# One-seed creative-8-task3 run. Reuse the final c8 NF + value-DGR recipe
# verbatim; only the task, seed, and output directory differ.
set -euo pipefail

cd /n/fs/mislresearch/sgcrl

export ENV=builderbench_creative_8_task3
export SEED="${SEED:-0}"
export NUM_STEPS="${NUM_STEPS:-300000000}"
export LOG_DIR="${LOG_DIR:-logs/final_runs/ppo_builderbench_creative8_task3_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent005_to001_ep100_300m_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_warp_3h/}"

echo "creative-8-task3 random planar spread; seed=${SEED} num_steps=${NUM_STEPS}"
echo "Reusing final c8 NF + value-DGR setup from job_c8t2_dgr_valuedgr.slurm"
echo "Fail-fast Slurm output limit: 8 MiB"

set +e
bash jobs/final_runs/job_c8t2_dgr_valuedgr.slurm 2>&1 |
  python -u scripts/cap_log_stream.py --max-mib=8
pipeline_status=("${PIPESTATUS[@]}")
set -e

if [ "${pipeline_status[1]}" -ne 0 ]; then
  exit "${pipeline_status[1]}"
fi
exit "${pipeline_status[0]}"
