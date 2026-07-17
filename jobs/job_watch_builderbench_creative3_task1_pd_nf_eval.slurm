#!/bin/bash
#SBATCH --job-name=bb_eval_c3t1_nf
#SBATCH --output=slurm/watch_bb_c3t1_pd_nf_eval_%j.log
#SBATCH --error=slurm/watch_bb_c3t1_pd_nf_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=seas

# Incremental checkpoint eval for creative-3-task1 PD + NF.
# Watches logs/ppo_builderbench_creative3_task1_e1024_pd_nf/ and updates:
#   figs/builderbench/creative3_task1_pd_nf_checkpoint_success.png
#   figs/builderbench/checkpoint_eval/ppo_builderbench_creative_3_task1_0_checkpoint_success.csv
#     (under the NF run dir basename — actually per run_dir name)
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

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p slurm figs/builderbench/checkpoint_eval

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "watching logs/ppo_builderbench_creative3_task1_e1024_pd_nf/ (interval=300s)"
echo "plot -> figs/builderbench/creative3_task1_pd_nf_checkpoint_success.png"
echo "csv  -> figs/builderbench/checkpoint_eval/ppo_builderbench_creative3_task1_e1024_pd_nf_seed0_checkpoint_success.csv"

python -u scripts/watch_builderbench_checkpoint_eval.py \
  --log_dir=logs/ppo_builderbench_creative3_task1_e1024_pd_nf \
  --env=builderbench_creative_3_task1 \
  --plot_tag=creative3_task1_pd_nf \
  --output=figs/builderbench/creative3_task1_pd_nf_checkpoint_success.png \
  --csv_output=figs/builderbench/checkpoint_eval/ppo_builderbench_creative3_task1_e1024_pd_nf_seed0_checkpoint_success.csv \
  --num_eval_episodes=20 \
  --eval_seed=0 \
  --state_file=figs/builderbench/checkpoint_eval/.watch_eval_state_c3t1_pd_nf.json \
  --watch_interval=300
