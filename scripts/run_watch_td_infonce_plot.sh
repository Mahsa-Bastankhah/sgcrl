#!/bin/bash
# Refresh TD InfoNCE experiment plots every 5 minutes.
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow 2>/dev/null || true
exec python -u plots/plot_td_infonce_experiments.py --watch --watch_interval 300
