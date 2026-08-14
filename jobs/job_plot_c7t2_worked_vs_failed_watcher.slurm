#!/bin/bash
#SBATCH --job-name=c7t2_wvf_plot
#SBATCH --output=slurm/c7t2_wvf_plot_%j.log
#SBATCH --error=slurm/c7t2_wvf_plot_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14:30:00
#SBATCH --account=seas

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache

INTERVAL=1800  # 30 minutes

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Started at $(date). Will re-run every ${INTERVAL}s."

while true; do
  echo "=== $(date) — running plot ==="
  python -u scripts/plot_c7t2_crl_nf_td3_worked_vs_failed.py
  echo "=== $(date) — done. Sleeping ${INTERVAL}s ==="
  sleep "${INTERVAL}"
done
