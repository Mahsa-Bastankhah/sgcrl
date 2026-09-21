#!/bin/bash
#SBATCH --job-name=plot_bin_safe
#SBATCH --output=slurm/final_metaworld_runs/plot_bin_floorgrasp_safe_reset_overall_%j.log
#SBATCH --error=slurm/final_metaworld_runs/plot_bin_floorgrasp_safe_reset_overall_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=seas

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow
export MPLBACKEND=Agg
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PYTHONUNBUFFERED=1

OUT=figs/sawyer_bin/floorgrasp_safe_reset_eval
mkdir -p "${OUT}" slurm/final_metaworld_runs

python -u scripts/plot_sawyer_bin_floorgrasp_safe_reset_overall.py \
  --seed_csv="${OUT}/seed0_safe_reset_checkpoint_success.csv" \
  --seed_csv="${OUT}/seed1_safe_reset_checkpoint_success.csv" \
  --seed_csv="${OUT}/seed2_safe_reset_checkpoint_success.csv" \
  --csv_output="${OUT}/seeds012_safe_reset_checkpoint_success.csv" \
  --plot_output="${OUT}/seeds012_safe_reset_checkpoint_success.png"
