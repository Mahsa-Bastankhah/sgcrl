#!/bin/bash
#SBATCH --job-name=c7t2_grad_vid
#SBATCH --output=slurm/bb_vid_c7t2_catwp_phi_psi_grad_450_600_%j.log
#SBATCH --error=slurm/bb_vid_c7t2_catwp_phi_psi_grad_450_600_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas

# Re-render c7t2 catwp iters 450 and 600 with ||∇_s (φ·ψ)|| / ||∇_a (φ·ψ)|| strips.
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
export JAX_PLATFORMS=cuda

CKPT_DIR=logs/ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m/ppo_builderbench_creative_7_task2_0/checkpoints
OUT_DIR=figs/builderbench/frozen_crl_reward_probe/c7t2_crl_catwp_ms_ea_s0_online_all
TAG_PREFIX=c7t2_crl_catwp_ms_ea_s0_online

mkdir -p slurm "${OUT_DIR}"
echo "c7t2 catwp phi-psi grad videos → ${OUT_DIR}"
nvidia-smi || true

for ITER in 0000450 0000600; do
  echo "=== ckpt_iter_${ITER} ==="
  python -u scripts/render_frozen_crl_traj_reward_video.py \
    --checkpoint="${CKPT_DIR}/ckpt_iter_${ITER}.pkl" \
    --env=builderbench_creative_7_task2 \
    --out_dir="${OUT_DIR}" \
    --tag_prefix="${TAG_PREFIX}" \
    --allow_no_success \
    --max_tries=1 \
    --seed=0 \
    --fps=8 \
    --show_phi_psi_grad
done

ls -lh "${OUT_DIR}"/${TAG_PREFIX}_iter_0000450* "${OUT_DIR}"/${TAG_PREFIX}_iter_0000600* || true
pkill -P $$ 2>/dev/null || true
