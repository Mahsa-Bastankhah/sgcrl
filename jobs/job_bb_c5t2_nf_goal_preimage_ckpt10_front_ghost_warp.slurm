#!/bin/bash
#SBATCH --job-name=c5t2_ghost_front
#SBATCH --output=slurm/bb_c5t2_nf_goal_preimage_ckpt10_front_ghost_warp_%j.log
#SBATCH --error=slurm/bb_c5t2_nf_goal_preimage_ckpt10_front_ghost_warp_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas

# Front-view zoomed ghost blends. Drop 10-100; later ckpts fill two rows of 6.
# Azimuth 0 = looking along +x at the pyramid face (not the side azimuth 125).
unset LD_PRELOAD
module purge
set -euo pipefail

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
export BUILDERBENCH_MJX_IMPL=warp
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

RUN_DIR=logs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep50_40m_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_ckpt10_warp_1h/ppo_builderbench_creative_5_task2_0
OUT_DIR=figs/builderbench/creative5_task2_nf_goal_preimage_ckpt10_front_ghost
ITERS=200,300,400,500,600,650,700,720,740,760,770,780

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== c5t2 NF front-view ghost blends (ckpt10 seed 0, zoom=5, az=0) ==="
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "run_dir=${RUN_DIR}"
echo "out_dir=${OUT_DIR}"
echo "iters=${ITERS}"
nvidia-smi || true
python -u -c "import os, warp, jax; print('warp', warp.__version__); print('JAX', jax.default_backend(), jax.devices())"

python -u scripts/builderbench_goal_preimage_composite.py \
    --repr_mode=nf \
    --env=builderbench_creative_5_task2 \
    --run_dir="${RUN_DIR}" \
    --output="${OUT_DIR}" \
    --iters="${ITERS}" \
    --num_trajs=10 \
    --top_k=5 \
    --seed=0 \
    --width=960 \
    --height=720 \
    --cam_zoom=5.0 \
    --cam_lookat=0.27,0.0,0.06 \
    --cam_azimuth=0 \
    --cam_elevation=-25 \
    --montage_ncols=6 \
    --gif_ms=700 \
    --skip_topk_gif

echo "=== done ==="
ls -lh "${OUT_DIR}/ghosts_montage.png" "${OUT_DIR}/blends_montage.png"
