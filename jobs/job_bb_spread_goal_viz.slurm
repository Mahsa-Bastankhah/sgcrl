#!/bin/bash
#SBATCH --job-name=bb_spread_viz
#SBATCH --output=/n/fs/mislresearch/sgcrl/slurm/bb_spread_goal_viz_%j.log
#SBATCH --error=/n/fs/mislresearch/sgcrl/slurm/bb_spread_goal_viz_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --account=seas

# Render the compact irregular creative-8 planar goal.
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

try_cuda() {
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    echo "Loaded CUDA modules: $*"
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || true
if [ -z "${CUDA_HOME:-}" ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      echo "Fallback CUDA_HOME=${CUDA_HOME}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export BUILDERBENCH_MJX_IMPL=warp
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache

OUT=figs/builderbench/c8_random_planar_goal.png
SCHEMATIC=figs/builderbench/c8_random_planar_goal_schematic.png
mkdir -p slurm figs/builderbench

echo "=== BuilderBench c8 random planar goal viz ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "MUJOCO_GL=${MUJOCO_GL}  BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
nvidia-smi -L || true

python -u scripts/builderbench_spread_goal_viz.py \
  --output="${OUT}" \
  --schematic-output="${SCHEMATIC}"
ls -lh "${OUT}" "${SCHEMATIC}"
