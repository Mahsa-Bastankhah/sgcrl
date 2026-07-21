#!/bin/bash
#SBATCH --job-name=bb_eval_lastnight
#SBATCH --output=slurm/bb_eval_lastnight_%A_%a.log
#SBATCH --error=slurm/bb_eval_lastnight_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=seas
#SBATCH --array=0-12

# One-shot deterministic checkpoint eval + success plot for last-night
# BuilderBench sweep runs (entanneal / rollout200 / r150_crl128 / r50_crl64 /
# r25_crl64). Array index selects the run; each task is independent.
unset LD_PRELOAD

module purge

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

CUDA_LOADED=0
try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    if [ -z "${CUDA_HOME:-}" ]; then
      for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
        if [ -d "${_cand}/lib64" ]; then
          export CUDA_HOME="${_cand}"
          break
        fi
      done
    fi
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || \
  try_cuda cudatoolkit/12.5 || \
  try_cuda cudatoolkit/12.4 || true

if [ "$CUDA_LOADED" != 1 ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1
      echo "Fallback CUDA_HOME=${CUDA_HOME} (module load failed)"
      break
    fi
  done
fi

if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit and no /usr/local/cuda* found."
  module avail cudatoolkit 2>&1 || true
  ls -d /usr/local/cuda* 2>&1 || true
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# log_dir|env|plot_tag
RUNS=(
  "logs/ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05_entanneal|builderbench_creative_3_task1|c3t1_td3_logq_tau05_entanneal"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05_rollout200|builderbench_creative_3_task1|c3t1_td3_logq_tau05_rollout200"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_gaussian_tau05_entanneal|builderbench_creative_3_task1|c3t1_gauss_tau05_entanneal"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_gaussian_tau05_rollout200|builderbench_creative_3_task1|c3t1_gauss_tau05_rollout200"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05_r150_crl128|builderbench_creative_3_task1|c3t1_td3_logq_tau05_r150_crl128"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_gaussian_tau05_r150_crl128|builderbench_creative_3_task1|c3t1_gauss_tau05_r150_crl128"
  "logs/ppo_builderbench_creative3_task2_e1024_pd_nf_tau05_r150_crl128|builderbench_creative_3_task2|c3t2_nf_tau05_r150_crl128"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05_r50_crl64|builderbench_creative_3_task1|c3t1_td3_logq_tau05_r50_crl64"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_gaussian_tau05_r50_crl64|builderbench_creative_3_task1|c3t1_gauss_tau05_r50_crl64"
  "logs/ppo_builderbench_creative3_task2_e1024_pd_nf_tau05_r50_crl64|builderbench_creative_3_task2|c3t2_nf_tau05_r50_crl64"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05_r25_crl64|builderbench_creative_3_task1|c3t1_td3_logq_tau05_r25_crl64"
  "logs/ppo_builderbench_creative3_task1_e1024_pd_gaussian_tau05_r25_crl64|builderbench_creative_3_task1|c3t1_gauss_tau05_r25_crl64"
  "logs/ppo_builderbench_creative3_task2_e1024_pd_nf_tau05_r25_crl64|builderbench_creative_3_task2|c3t2_nf_tau05_r25_crl64"
)

IDX=${SLURM_ARRAY_TASK_ID}
IFS='|' read -r LOG_DIR ENV PLOT_TAG <<< "${RUNS[$IDX]}"
RUN_DIR=$(ls -d "${LOG_DIR}"/ppo_builderbench_creative_*_* 2>/dev/null | head -1)

if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}/checkpoints" ]; then
  echo "ERROR: no run dir / checkpoints for LOG_DIR=${LOG_DIR}"
  ls -la "${LOG_DIR}" 2>&1 || true
  exit 1
fi

CSV="figs/builderbench/checkpoint_eval/${PLOT_TAG}_seed0_checkpoint_success.csv"
PNG="figs/builderbench/${PLOT_TAG}_checkpoint_success.png"
mkdir -p slurm figs/builderbench/checkpoint_eval

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  ARRAY=${SLURM_ARRAY_TASK_ID}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "ENV=${ENV}  PLOT_TAG=${PLOT_TAG}"
echo "CSV=${CSV}"
echo "PNG=${PNG}"
ls "${RUN_DIR}/checkpoints"/ckpt_iter_*.pkl 2>/dev/null | wc -l | awk '{print "n_ckpts="$1}'
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u scripts/ppo_builderbench_checkpoint_eval.py \
  --run_dir="${RUN_DIR}" \
  --env="${ENV}" \
  --plot_tag="${PLOT_TAG}" \
  --csv_output="${CSV}" \
  --output="${PNG}" \
  --incremental \
  --num_eval_episodes=20

PY_EXIT=$?
echo "done exit=${PY_EXIT}  png=${PNG}"
ls -lh "${PNG}" "${CSV}" 2>&1 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
