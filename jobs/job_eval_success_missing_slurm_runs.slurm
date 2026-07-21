#!/bin/bash
#SBATCH --job-name=bb_eval_missing
#SBATCH --output=slurm/bb_eval_missing_%A_%a.log
#SBATCH --error=slurm/bb_eval_missing_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas
#SBATCH --array=0-2

# One-shot eval+plot for BuilderBench runs that have a slurm log but no
# (complete) checkpoint-success plot yet.
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

# run_dir|env|plot_tag
RUNS=(
  "logs/ppo_builderbench_creative1_task1_e1024_pd_gaussian_tau05_log2pi/ppo_builderbench_creative_1_task1_0|builderbench_creative_1_task1|c1t1_gauss_tau05_log2pi"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05/ppo_builderbench_creative_3_task1_0|builderbench_creative_3_task1|c3t1_td3_logq_tau05"
  "logs/ppo_builderbench_creative3_task2_e1024_pd_crl_tau05_mid/ppo_builderbench_creative_3_task2_0|builderbench_creative_3_task2|c3t2_crl_tau05_mid"
)

IFS='|' read -r RUN_DIR ENV PLOT_TAG <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
CSV="figs/builderbench/checkpoint_eval/${PLOT_TAG}_seed0_checkpoint_success.csv"
PNG="figs/builderbench/${PLOT_TAG}_checkpoint_success.png"
mkdir -p slurm figs/builderbench/checkpoint_eval

echo "ARRAY=${SLURM_ARRAY_TASK_ID} RUN_DIR=${RUN_DIR} TAG=${PLOT_TAG}"
ls "${RUN_DIR}/checkpoints"/ckpt_iter_*.pkl 2>/dev/null | wc -l | awk '{print "n_ckpts="$1}'
nvidia-smi || true

python -u scripts/ppo_builderbench_checkpoint_eval.py \
  --run_dir="${RUN_DIR}" \
  --env="${ENV}" \
  --plot_tag="${PLOT_TAG}" \
  --csv_output="${CSV}" \
  --output="${PNG}" \
  --incremental \
  --num_eval_episodes=20

PY_EXIT=$?
ls -lh "${PNG}" "${CSV}" 2>&1 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
