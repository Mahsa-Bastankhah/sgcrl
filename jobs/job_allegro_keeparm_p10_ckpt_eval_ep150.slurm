#!/bin/bash
#SBATCH --job-name=akt_p10_ckpt_eval_ep150
#SBATCH --output=slurm/allegro_keeparm_p10_ckpt_eval_ep150_%j.log
#SBATCH --error=slurm/allegro_keeparm_p10_ckpt_eval_ep150_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=seas
#SBATCH --exclude=neu306

# Offline P10 pad-hold bucket policies at T=150 (train was T=50).
# First-episode only; fall/timeout does not start a second try.
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

OUT_DIR="figs/allegro_kuka_throw/p10_ckpt_eval_ep150"
CKPT_200="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_200m_q10shift_seed0_7h/ppo_allegro_kuka_throw_0/checkpoints"
CKPT_500="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_q10shift_seed0_14h/ppo_allegro_kuka_throw_0/checkpoints"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "T=150 E=256 deterministic  first-episode only"
echo "CKPT_200=$(ls -1 ${CKPT_200}/ckpt_iter_*.pkl | wc -l)"
echo "CKPT_500=$(ls -1 ${CKPT_500}/ckpt_iter_*.pkl | wc -l)"
nvidia-smi || true

run_eval() {
  local ckpt_dir="$1" csv="$2" plot="$3"
  local try
  for try in 1 2; do
    echo "eval_try=${try}/2 csv=${csv}"
    timeout 80m python -u scripts/eval_allegro_ckpt_success.py \
      --checkpoint-dir="${ckpt_dir}" \
      --output-csv="${csv}" \
      --output-plot="${plot}" \
      --episode-length=150 \
      --num-envs=256 \
      --seed=0 \
      --pipeline=gpu \
      --deterministic \
      --steps-per-iter=$((1024 * 50))
    local ec=$?
    if [ -f "${csv}" ]; then
      echo "wrote ${csv} exit=${ec}"
      return 0
    fi
    echo "eval failed/hung try=${try} exit=${ec}; sleep 15"
    sleep 15
  done
  return 1
}

run_eval "${CKPT_200}" "${OUT_DIR}/q10_200m.csv" "${OUT_DIR}/q10_200m.png"
run_eval "${CKPT_500}" "${OUT_DIR}/q10_500m.csv" "${OUT_DIR}/q10_500m.png"

python -u - <<'PY'
import csv, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
sys.path.insert(0, 'scripts')
import plot_builderbench_train_success1000 as base

out = 'figs/allegro_kuka_throw/p10_ckpt_eval_ep150/p10_ckpt_eval_ep150.png'
runs = (
    ('figs/allegro_kuka_throw/p10_ckpt_eval_ep150/q10_200m.csv',
     'P10 200M (3870369)', base.ACCENT_COLORS[2]),
    ('figs/allegro_kuka_throw/p10_ckpt_eval_ep150/q10_500m.csv',
     'P10 500M (3871188)', base.ACCENT_COLORS[8]),
)
fig, ax = plt.subplots(figsize=(10.4, 4.8))
for path, label, color in runs:
    if not os.path.isfile(path):
        continue
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    xs = [float(r['env_steps']) for r in rows]
    ys = [float(r['success']) for r in rows]
    base._plot_eval_smoothed(ax, xs, ys, color=color, label=label + '  ever')
    if rows and 'end_success' in rows[0]:
        ye = [float(r['end_success']) for r in rows]
        base._plot_eval_smoothed(
            ax, xs, ye, color=color, label=label + '  end', linestyle='--')
    peak = max(ys) if ys else 0.0
    print(f'{label}: n={len(xs)} last_ever={ys[-1]:.4f} peak_ever={peak:.4f}')
ax.set_title(
    f'P10 hide-table bucket  offline ckpt eval  T=150  '
    f'(roll mean w={base.EVAL_SMOOTH_WINDOW})',
    fontsize=11, fontweight='bold')
ax.set_ylabel(f'in-bucket ≤7.5cm (roll mean, w={base.EVAL_SMOOTH_WINDOW})')
ax.set_xlabel('Train env steps')
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='best', fontsize=8, framealpha=0.95)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
fig.tight_layout()
fig.savefig(out, dpi=160)
print('wrote', out)
PY

echo "done"
ls -lh "${OUT_DIR}"
