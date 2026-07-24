#!/bin/bash
#SBATCH --job-name=mpo_bb_eval
#SBATCH --output=slurm/mpo_bb_ckpt_eval_%j.log
#SBATCH --error=slurm/mpo_bb_ckpt_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# One-shot MPO BuilderBench checkpoint eval + success plot.
# Required: RUN_DIR, ENV_NAME, PLOT_TAG
# Optional: SEED (default 0), NUM_EVAL_EPISODES (default 20), ONLY_LABELS
unset LD_PRELOAD
module purge

try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -n "${CUDA_HOME:-}" ] && return 0
    for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
      [ -d "${_cand}/lib64" ] && export CUDA_HOME="${_cand}" && return 0
    done
    return 1
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || try_cuda cudatoolkit/12.4 || true
if [ -z "${CUDA_HOME:-}" ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

: "${RUN_DIR:?RUN_DIR required}"
: "${ENV_NAME:?ENV_NAME required}"
: "${PLOT_TAG:?PLOT_TAG required}"
SEED="${SEED:-0}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}"
ONLY_LABELS="${ONLY_LABELS:-}"
# Watcher may pass labels with ';' to avoid Slurm --export comma splitting.
ONLY_LABELS="${ONLY_LABELS//;/,}"

CSV="figs/builderbench/checkpoint_eval/${PLOT_TAG}_seed${SEED}_checkpoint_success.csv"
PNG="figs/builderbench/${PLOT_TAG}_seed${SEED}_checkpoint_success.png"
AGG_PNG="figs/builderbench/${PLOT_TAG}_checkpoint_success.png"
mkdir -p slurm figs/builderbench/checkpoint_eval

echo "RUN_DIR=${RUN_DIR}"
echo "ENV_NAME=${ENV_NAME}  PLOT_TAG=${PLOT_TAG}  SEED=${SEED}"
echo "CSV=${CSV}"
echo "PNG=${PNG}"
echo "ONLY_LABELS=${ONLY_LABELS}"
ls "${RUN_DIR}/checkpoints"/ckpt_iter_*.pkl 2>/dev/null | wc -l | awk '{print "n_ckpts="$1}'
nvidia-smi || true

EXTRA=()
if [[ -n "${ONLY_LABELS}" ]]; then
  EXTRA+=(--only_labels="${ONLY_LABELS}")
fi

python -u scripts/mpo_builderbench_checkpoint_eval.py \
  --run_dir="${RUN_DIR}" \
  --env="${ENV_NAME}" \
  --plot_tag="${PLOT_TAG}" \
  --csv_output="${CSV}" \
  --output="${PNG}" \
  --incremental \
  --num_eval_episodes="${NUM_EVAL_EPISODES}" \
  "${EXTRA[@]}"

PY_EXIT=$?

# Refresh aggregate mean±SE plot across whatever seed CSVs exist.
python -u - <<'PY' || true
import csv, glob, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plot_tag = os.environ['PLOT_TAG']
env_name = os.environ['ENV_NAME']
eval_dir = 'figs/builderbench/checkpoint_eval'
out_path = f'figs/builderbench/{plot_tag}_checkpoint_success.png'
by_x = {}
n_seeds = 0
for path in sorted(glob.glob(os.path.join(eval_dir, f'{plot_tag}_seed*_checkpoint_success.csv'))):
    rows = 0
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            try:
                x = int(float(row['global_step']))
                y = float(row['success_mean'])
            except (KeyError, TypeError, ValueError):
                continue
            by_x.setdefault(x, []).append(y)
            rows += 1
    if rows:
        n_seeds += 1
if by_x and n_seeds:
    xs = np.asarray(sorted(by_x), dtype=np.float64)
    means = np.empty_like(xs)
    ses = np.empty_like(xs)
    for i, x in enumerate(xs):
        ys = np.asarray(by_x[int(x)], dtype=np.float64)
        means[i] = float(ys.mean())
        ses[i] = float(ys.std(ddof=1) / math.sqrt(ys.size)) if ys.size > 1 else 0.0

    def fmt(v, _):
        if v >= 1e6:
            return f'{v/1e6:.1f}M'.replace('.0M', 'M')
        if v >= 1e3:
            return f'{v/1e3:.0f}K'
        return str(int(v))

    fig, ax = plt.subplots(figsize=(8, 4))
    c = '#4C9BE8'
    ax.plot(xs, means, color=c, lw=1.8, marker='o', ms=4,
            label=f'eval success (mean ± SE, n={n_seeds})')
    ax.fill_between(xs, means - ses, means + ses, color=c, alpha=0.25, lw=0)
    ax.set_title(f'MPO BuilderBench eval success — {env_name} ({plot_tag})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Global env steps'); ax.set_ylabel('Eval success rate')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))
    ax.set_ylim(-0.02, 1.02)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'[mpo_bb_eval] refreshed aggregate: {out_path} (n={n_seeds})')
PY

ls -lh "${PNG}" "${CSV}" "${AGG_PNG}" 2>&1 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
