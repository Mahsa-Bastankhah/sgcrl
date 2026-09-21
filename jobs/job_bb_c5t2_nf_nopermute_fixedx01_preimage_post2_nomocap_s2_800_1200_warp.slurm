#!/bin/bash
#SBATCH --job-name=c5t2_post2_s2
#SBATCH --output=slurm/bb_c5t2_nf_preimage_post2_nomocap_s2_800_1200_warp_%j.log
#SBATCH --error=slurm/bb_c5t2_nf_preimage_post2_nomocap_s2_800_1200_warp_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas

# Probe: post_success_steps=2 for seed-2 800/1200 (does not touch paper ghost fig).
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

RUN_DIR=logs/final_runs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_warp_2h30/ppo_builderbench_creative_5_task2_2
OUT_DIR=figs/builderbench/creative5_task2_nf_nopermute_fixedx01_goal_preimage_post2_nomocap_s2_front
ITERS=800,1200

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== C5T2 seed-2 post2 nomocap top-5 probe ==="
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "out_dir=${OUT_DIR}"
echo "iters=${ITERS} post_success_steps=2"
nvidia-smi || true
python -u -c "import warp, jax; print('warp', warp.__version__); print('JAX', jax.default_backend(), jax.devices())"

python -u scripts/builderbench_goal_preimage_composite.py \
    --repr_mode=nf \
    --env=builderbench_creative_5_task2 \
    --run_dir="${RUN_DIR}" \
    --output="${OUT_DIR}" \
    --iters="${ITERS}" \
    --num_trajs=10 \
    --top_k=5 \
    --post_success_steps=2 \
    --hide_mocap \
    --seed=0 \
    --width=960 \
    --height=720 \
    --cam_zoom=5.0 \
    --cam_lookat=0.27,0.0,0.06 \
    --cam_azimuth=0 \
    --cam_elevation=-25 \
    --montage_ncols=2 \
    --gif_ms=700 \
    --skip_topk_gif

echo "=== done ==="
ls -lh "${OUT_DIR}/strips_vertical.png"
