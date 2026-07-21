#!/bin/bash
#SBATCH --job-name=bb_ckpt_eval
#SBATCH --output=slurm/bb_ckpt_eval_%j.log
#SBATCH --error=slurm/bb_ckpt_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# One-shot BuilderBench checkpoint eval + success plot.
# Required env vars: RUN_DIR, ENV_NAME, PLOT_TAG
# Optional: SEED (default 0), NUM_EVAL_EPISODES (default 20)
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

: "${RUN_DIR:?RUN_DIR required}"
: "${ENV_NAME:?ENV_NAME required}"
: "${PLOT_TAG:?PLOT_TAG required}"
SEED="${SEED:-0}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}"

CSV="figs/builderbench/checkpoint_eval/${PLOT_TAG}_seed${SEED}_checkpoint_success.csv"
PNG="figs/builderbench/${PLOT_TAG}_checkpoint_success.png"
mkdir -p slurm figs/builderbench/checkpoint_eval

echo "RUN_DIR=${RUN_DIR}"
echo "ENV_NAME=${ENV_NAME}  PLOT_TAG=${PLOT_TAG}  SEED=${SEED}"
echo "CSV=${CSV}"
echo "PNG=${PNG}"
ls "${RUN_DIR}/checkpoints"/ckpt_iter_*.pkl 2>/dev/null | wc -l | awk '{print "n_ckpts="$1}'
nvidia-smi || true

python -u scripts/ppo_builderbench_checkpoint_eval.py \
  --run_dir="${RUN_DIR}" \
  --env="${ENV_NAME}" \
  --plot_tag="${PLOT_TAG}" \
  --csv_output="${CSV}" \
  --output="${PNG}" \
  --incremental \
  --num_eval_episodes="${NUM_EVAL_EPISODES}"

PY_EXIT=$?
ls -lh "${PNG}" "${CSV}" 2>&1 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
