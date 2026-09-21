#!/bin/bash
#SBATCH --job-name=its_coord_rank_merge
#SBATCH --output=slurm/allegro_its_coordinate_nf_rank_consolidate_%j.log
#SBATCH --error=slurm/allegro_its_coordinate_nf_rank_consolidate_%j.log
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
export MPLBACKEND=Agg
OUT="figs/allegro_kuka_throw/coordinate_mode_nf_rank_accuracy"

python - "${OUT}" <<'PY'
import csv
import glob
import os
import sys
from scripts.eval_allegro_nf_rank_accuracy import _plot, _write_csv

out = sys.argv[1]
for mode in ("physical", "fully_scaled", "mixed"):
  rows = []
  for path in glob.glob(os.path.join(out, mode, "parts", "iter_*.csv")):
    with open(path, newline="", encoding="utf-8") as handle:
      rows.extend(dict(row) for row in csv.DictReader(handle))
  rows.sort(key=lambda row: int(row["iteration"]))
  if [int(row["iteration"]) for row in rows] != [100, 500, 1000, 1300]:
    raise RuntimeError(f"{mode}: incomplete rows")
  csv_path = os.path.join(out, mode, "rank_accuracy.csv")
  png_path = os.path.join(out, mode, "rank_accuracy.png")
  _write_csv(csv_path, rows)
  _plot(png_path, rows)
  print(f"wrote {csv_path} and {png_path}")
PY

python scripts/compare_allegro_nf_rank_accuracy.py \
  --input="physical=${OUT}/physical/rank_accuracy.csv" \
  --input="fully_scaled=${OUT}/fully_scaled/rank_accuracy.csv" \
  --input="mixed_baseline=${OUT}/mixed/rank_accuracy.csv" \
  --output-csv="${OUT}/coordinate_mode_comparison.csv" \
  --output-plot="${OUT}/coordinate_mode_comparison.png"
