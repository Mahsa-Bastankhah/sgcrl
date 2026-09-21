#!/bin/bash
#SBATCH --job-name=vid_bin_g04_md
#SBATCH --output=slurm/final_metaworld_runs/vid_bin_floorgrasp_seeds_stoch_multidraw_%j.log
#SBATCH --error=slurm/final_metaworld_runs/vid_bin_floorgrasp_seeds_stoch_multidraw_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas

# 5 stochastic latest-checkpoint rollouts for each floor-grasp seed (0/1/2).
# Different --seed values vary both policy noise and env init (np.random).
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
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

ROOT=/n/fs/mislresearch/sgcrl/logs/final_metaworld_runs/ppo_bin_nf_pegrecipe_floorgrasp_g04_rand_mixtaskg
OUT=/n/fs/mislresearch/sgcrl/videos/final_metaworld_runs/bin_floorgrasp_g04_seed_diagnostics
mkdir -p "${OUT}" slurm/final_metaworld_runs

N_DRAWS=5
DRAW_SEEDS=(200 201 202 203 204)

latest_concrete_ckpt() {
  python - "$1" <<'PY'
import glob
import os
import re
import sys

paths = glob.glob(os.path.join(sys.argv[1], 'ckpt_iter_*.pkl'))
if not paths:
  raise SystemExit(f'No concrete checkpoints under {sys.argv[1]}')
paths.sort(key=lambda p: int(re.search(r'ckpt_iter_(\d+)', p).group(1)))
print(paths[-1])
PY
}

echo "Rendering ${N_DRAWS} stochastic floor-grasp draws per seed (0/1/2)"
nvidia-smi || true

for SEED in 0 1 2; do
  CKPT_DIR="${ROOT}/ppo_sawyer_bin_${SEED}/checkpoints"
  CKPT="$(latest_concrete_ckpt "${CKPT_DIR}")" || exit 1
  echo "seed=${SEED} checkpoint=${CKPT}"
  for DI in $(seq 0 $((N_DRAWS - 1))); do
    RSEED="${DRAW_SEEDS[$DI]}"
    OUT_MP4="${OUT}/floorgrasp_g04_seed${SEED}_draw${DI}_stoch.mp4"
    echo "  draw=${DI} rollout_seed=${RSEED} → ${OUT_MP4}"
    python -u scripts/ppo_rollout_video.py \
      --checkpoint="${CKPT}" \
      --env=sawyer_bin \
      --stochastic \
      --seed="${RSEED}" \
      --output="${OUT_MP4}" || exit 1
  done
done

ls -lh "${OUT}"/floorgrasp_g04_seed*_draw*_stoch.mp4
echo FLOORGRASP_MULTIDRAW_VIDEO_JOB_DONE
