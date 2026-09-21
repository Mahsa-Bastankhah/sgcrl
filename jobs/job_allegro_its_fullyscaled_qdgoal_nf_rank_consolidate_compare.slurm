#!/bin/bash
#SBATCH --job-name=its_fsqg_rank_merge
#SBATCH --output=slurm/allegro_its_fullyscaled_qdgoal_rank_consolidate_%j.log
#SBATCH --error=slurm/allegro_its_fullyscaled_qdgoal_rank_consolidate_%j.log
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

OUT="figs/allegro_kuka_throw/index_thumb_straight_fullyscaled_qdgoal_fullrange_rank_accuracy"
BASELINE="${BASELINE:-figs/allegro_kuka_throw/coordinate_mode_nf_rank_accuracy/fully_scaled/rank_accuracy.csv}"
BASELINE_PARTS="figs/allegro_kuka_throw/coordinate_mode_nf_rank_accuracy/fully_scaled/parts"
python - "${OUT}" <<'PY'
import csv
import os
import sys
from scripts.eval_allegro_nf_rank_accuracy import _plot, _write_csv

out = sys.argv[1]
rows = []
for iteration in (100, 500, 1000, 1300):
  path = os.path.join(out, f"compactsmall_mix50_ckpt{iteration}_seed0.csv")
  with open(path, newline="", encoding="utf-8") as handle:
    rows.extend(dict(row) for row in csv.DictReader(handle))
if [int(row["iteration"]) for row in rows] != [100, 500, 1000, 1300]:
  raise RuntimeError("incomplete or misordered fully-scaled qd-goal rank rows")
_write_csv(os.path.join(out, "rank_accuracy.csv"), rows)
_plot(os.path.join(out, "rank_accuracy.png"), rows)
PY

if [[ ! -f "${BASELINE}" ]]; then
  python - "${BASELINE}" "${BASELINE_PARTS}" <<'PY'
import csv
import os
import sys
from scripts.eval_allegro_nf_rank_accuracy import _plot, _write_csv

output, parts = sys.argv[1:]
rows = []
for iteration in (100, 500, 1000, 1300):
  path = os.path.join(parts, f"iter_{iteration}.csv")
  with open(path, newline="", encoding="utf-8") as handle:
    rows.extend(dict(row) for row in csv.DictReader(handle))
if [int(row["iteration"]) for row in rows] != [100, 500, 1000, 1300]:
  raise RuntimeError("incomplete or misordered fully-scaled q-only rank rows")
_write_csv(output, rows)
_plot(os.path.splitext(output)[0] + ".png", rows)
PY
fi

echo "CAUTION: binary rank accuracy across 16D q+qd and 8D q-only goal spaces is not strictly apples-to-apples."
if [[ -f "${BASELINE}" ]]; then
  python scripts/compare_allegro_nf_rank_accuracy.py \
    --input="fullyscaled_q_only_goal_8d=${BASELINE}" \
    --input="fullyscaled_qd_goal_16d=${OUT}/rank_accuracy.csv" \
    --output-csv="${OUT}/vs_fullyscaled_q_only_goal_comparison.csv" \
    --output-plot="${OUT}/vs_fullyscaled_q_only_goal_comparison.png"
else
  echo "Optional fully-scaled q-only baseline not found at ${BASELINE}; consolidation complete."
  echo "Resubmit with --export=ALL,BASELINE=/path/to/baseline.csv to compare."
fi
