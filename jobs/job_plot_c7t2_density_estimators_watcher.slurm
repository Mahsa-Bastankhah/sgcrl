#!/bin/bash
#SBATCH --job-name=c7t2_ode_plot
#SBATCH --output=slurm/c7t2_density_estimators_plot_%j.log
#SBATCH --error=slurm/c7t2_density_estimators_plot_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --account=seas
# No GPU: CSV-only paper-plot refresh for the live c7t2 CRL / TD-InfoNCE jobs.

# Watches 3814411 (CRL dual-dgr) and 3814412 (TD-InfoNCE). Re-plots the
# paper figure every 15 min with --allow-partial while they run, then one
# final plot after both leave the queue.
unset -f python python3 2>/dev/null || true
unset LD_PRELOAD

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export TMPDIR=/n/fs/mislresearch/mb6458-tmp

INTERVAL="${INTERVAL:-900}"
TRAIN_JOBS="${TRAIN_JOBS:-3814411,3814412}"
PLOT_PY="paper material/paper plot pythons/plot_c7t2_density_estimators_success.py"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Started $(date). interval=${INTERVAL}s  train_jobs=${TRAIN_JOBS}"
echo "hostname=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

jobs_running() {
  squeue -j "${TRAIN_JOBS}" -h -o '%i %T' 2>/dev/null \
    | awk '$2 ~ /RUNNING|PENDING|CONFIGURING|COMPLETING/ { found=1 } END { exit !found }'
}

run_plot() {
  local extra=("$@")
  echo "=== $(date) — ${PLOT_PY} ${extra[*]} ==="
  python -u "${PLOT_PY}" "${extra[@]}"
}

while true; do
  if jobs_running; then
    run_plot --allow-partial
    echo "=== $(date) — train jobs still up; sleeping ${INTERVAL}s ==="
    sleep "${INTERVAL}"
  else
    echo "=== $(date) — train jobs gone; final plot ==="
    if ! run_plot; then
      echo "final plot without --allow-partial failed; retrying partial"
      run_plot --allow-partial
    fi
    echo "=== $(date) — watcher exiting ==="
    break
  fi
done
