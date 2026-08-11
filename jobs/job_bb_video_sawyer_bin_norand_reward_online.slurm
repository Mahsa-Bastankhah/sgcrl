#!/bin/bash
#SBATCH --job-name=bin_crlrew_vid
#SBATCH --output=slurm/bb_vid_sawyer_bin_norand_reward_online_%j.log
#SBATCH --error=slurm/bb_vid_sawyer_bin_norand_reward_online_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=seas

# Online φ·ψ reward-strip videos for the most recent sawyer_bin run
# (no hit-indicator): logs/ppo_bin_crl_crl10_40m_minstd1e5_extrew1_norand
# seed 0. Evenly subsample 24 milestones (87 total would be too slow).
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

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

CKPT_DIR=logs/ppo_bin_crl_crl10_40m_minstd1e5_extrew1_norand/ppo_sawyer_bin_0/checkpoints
OUT_DIR=figs/sawyer_bin/frozen_crl_reward_probe/bin_norand_extrew1_s0_online_all
mkdir -p slurm "${OUT_DIR}"

echo "Rendering sawyer_bin norand seed0 φ·ψ reward-strip videos → ${OUT_DIR}"
nvidia-smi || true

python -u scripts/render_sawyer_crl_traj_reward_video.py \
  --checkpoint_dir="${CKPT_DIR}" \
  --env=sawyer_bin \
  --out_dir="${OUT_DIR}" \
  --tag_prefix=bin_norand_extrew1_s0_online \
  --allow_no_success \
  --max_tries=8 \
  --max_ckpts=24 \
  --seed=0 \
  --fps=15 \
  --camera=corner \
  --rotate=180 \
  --skip_existing

ls -lh "${OUT_DIR}"/*.mp4 2>&1 | tail -40 || true
pkill -P $$ 2>/dev/null || true
