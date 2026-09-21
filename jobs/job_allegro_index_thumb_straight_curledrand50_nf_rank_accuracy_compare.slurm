#!/bin/bash
#SBATCH --job-name=its_rank50_compare
#SBATCH --output=slurm/allegro_index_thumb_straight_curledrand50_nf_rank_accuracy_compare_%j.log
#SBATCH --error=slurm/allegro_index_thumb_straight_curledrand50_nf_rank_accuracy_compare_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --account=seas

set -euo pipefail
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

OUT_DIR="figs/allegro_kuka_throw/index_thumb_straight_curledrand50_rank_accuracy"
python -u scripts/compare_allegro_nf_rank_accuracy.py \
  --input="compact-small + mix50=${OUT_DIR}/compactsmall_mix50_ckpts100_500_1000_1300.csv" \
  --input="tiny + mix50=${OUT_DIR}/tiny_mix50_ckpts100_500_1000_1300.csv" \
  --input="compact-small + no mix=${OUT_DIR}/compactsmall_nomix_ckpts100_500_1000_1300.csv" \
  --output-csv="${OUT_DIR}/matched_variants.csv" \
  --output-plot="${OUT_DIR}/matched_variants.png"
