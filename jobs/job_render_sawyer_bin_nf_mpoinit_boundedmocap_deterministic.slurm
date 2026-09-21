#!/bin/bash
#SBATCH --job-name=vid_nf_mpo_bm
#SBATCH --output=slurm/final_metaworld_runs/vid_nf_mpoinit_boundedmocap_%j.log
#SBATCH --error=slurm/final_metaworld_runs/vid_nf_mpoinit_boundedmocap_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# Required exports: POLICY_SEED, CKPT_ITER, INIT_SEED.
# One deterministic rollout with the bounded-controller run's randomized
# safe MPO init (object XY random, TCP XY +/-5cm, z U(3cm, 6cm)).
set -euo pipefail
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
    echo "Loaded CUDA modules: $*"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || \
  try_cuda cudatoolkit/12.5 || \
  try_cuda cudatoolkit/12.4 || true

if [ "${CUDA_LOADED}" != 1 ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1
      echo "Fallback CUDA_HOME=${CUDA_HOME}"
      break
    fi
  done
fi

if [ "${CUDA_LOADED}" != 1 ]; then
  echo "ERROR: no usable CUDA toolkit found"
  exit 1
fi

: "${POLICY_SEED:?POLICY_SEED is required}"
: "${CKPT_ITER:?CKPT_ITER is required}"
: "${INIT_SEED:?INIT_SEED is required}"

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

RUN_ROOT=logs/final_metaworld_runs/ppo_bin_nf_floorgrasp_safeinit_mpoinit_boundedmocap_mixtaskg_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6
printf -v ITER_PAD "%07d" "${CKPT_ITER}"
CKPT="${RUN_ROOT}/ppo_sawyer_bin_${POLICY_SEED}/checkpoints/ckpt_iter_${ITER_PAD}.pkl"
OUT_DIR="videos/final_metaworld_runs/bin_nf_mpoinit_boundedmocap_deterministic/seed${POLICY_SEED}"
OUT="${OUT_DIR}/sawyer_bin_iter_${ITER_PAD}_init${INIT_SEED}.mp4"

mkdir -p slurm/final_metaworld_runs "${OUT_DIR}"
echo "policy_seed=${POLICY_SEED} checkpoint_iter=${CKPT_ITER} init_seed=${INIT_SEED}"
echo "policy=deterministic"
echo "init=SAFE MPO randomized; control=bounded per-step mocap"
echo "checkpoint=${CKPT}"
echo "output=${OUT}"
nvidia-smi || true

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint missing: ${CKPT}"
  exit 2
fi

python -u scripts/ppo_rollout_video.py \
  --checkpoint="${CKPT}" \
  --env=sawyer_bin \
  --output="${OUT}" \
  --width=640 \
  --height=480 \
  --fps=30 \
  --rotate=180 \
  --camera=corner \
  --seed="${INIT_SEED}" \
  --sawyer_bin_safe_grasp_reset \
  --bin_randomize_gripper_init \
  --sawyer_bin_bounded_step_mocap

echo "VIDEO_READY ${OUT}"
