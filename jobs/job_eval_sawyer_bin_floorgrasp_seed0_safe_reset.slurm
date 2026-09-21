#!/bin/bash
#SBATCH --job-name=eval_bin_safe
#SBATCH --output=slurm/final_metaworld_runs/eval_bin_floorgrasp_seed0_safe_reset_%j.log
#SBATCH --error=slurm/final_metaworld_runs/eval_bin_floorgrasp_seed0_safe_reset_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# Ten deterministic episodes from ten rendered, healthy reset states per ckpt.
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
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || \
  try_cuda cudatoolkit/12.5 || \
  try_cuda cudatoolkit/12.4 || true

if [ "${CUDA_LOADED}" != 1 ]; then
  for candidate in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${candidate}/lib64" ]; then
      export CUDA_HOME="${candidate}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1
      break
    fi
  done
fi

if [ "${CUDA_LOADED}" != 1 ]; then
  echo "ERROR: no usable CUDA toolkit found"
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN=logs/final_metaworld_runs/ppo_bin_nf_pegrecipe_floorgrasp_g04_rand_mixtaskg/ppo_sawyer_bin_0
OUT=figs/sawyer_bin/floorgrasp_safe_reset_eval
mkdir -p "${OUT}" slurm/final_metaworld_runs

echo "seed=0 checkpoints=${RUN}/checkpoints deterministic=true episodes=10"
echo "safe reset: d<=0.06 xy<=0.04 dz=[0,0.06] grip=[0.35,0.45] contacts>=1"
nvidia-smi || true

python -u scripts/eval_sawyer_bin_floorgrasp_checkpoints_safe_reset.py \
  --checkpoint_dir="${RUN}/checkpoints" \
  --csv_output="${OUT}/seed0_safe_reset_checkpoint_success.csv" \
  --plot_output="${OUT}/seed0_safe_reset_checkpoint_success.png" \
  --init_csv_output="${OUT}/seed0_safe_reset_initializations.csv" \
  --init_plot_output="${OUT}/seed0_safe_reset_initializations.png" \
  --episodes=10 \
  --run_seed=0 \
  --max_reset_attempts=100
