#!/bin/bash
#SBATCH --job-name=bb_ckpt_eval05
#SBATCH --output=slurm/bb_ckpt_eval05_%j.log
#SBATCH --error=slurm/bb_ckpt_eval05_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --account=seas

# One-shot BuilderBench checkpoint eval + success plot (light: 16G, ≤55min).
# NOTE: this cluster does not allocate a device for --gpus-per-task=0.5
# (jobs start with CUDA_ERROR_NO_DEVICE), so we request 1 full GPU.
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
ACTIVE_PNG="figs/builderbench/active_runs/${PLOT_TAG}_checkpoint_success.png"
mkdir -p slurm figs/builderbench/checkpoint_eval figs/builderbench/active_runs

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
if [ $PY_EXIT -eq 0 ] && [ -f "${PNG}" ]; then
  cp -f "${PNG}" "${ACTIVE_PNG}"
  echo "mirrored → ${ACTIVE_PNG}"
fi
ls -lh "${PNG}" "${CSV}" "${ACTIVE_PNG}" 2>&1 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
