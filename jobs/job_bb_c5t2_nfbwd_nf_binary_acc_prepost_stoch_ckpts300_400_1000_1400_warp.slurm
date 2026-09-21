#!/bin/bash
#SBATCH --job-name=c5t2_nfbwd_binacc
#SBATCH --output=slurm/bb_c5t2_nfbwd_nf_binary_acc_prepost_stoch_ckpts300_400_1000_1400_warp_%j.log
#SBATCH --error=slurm/bb_c5t2_nfbwd_nf_binary_acc_prepost_stoch_ckpts300_400_1000_1400_warp_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Stochastic NF future-goal binary rank accuracy before vs after success.
# Same run as paper plot c5t2_nfbwd_iter1000_ppo_r_grad_s (seed 0).
# Ckpts: 300, 400, 1000, 1400.
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
export BUILDERBENCH_MJX_IMPL=warp
export PYTHONPATH="/n/fs/mislresearch/builderbench${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

RUN_DIR=logs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect_extrew1_minstd1e5_ent05to001_ep70_T50_100m_crl10_nfbwd/ppo_builderbench_creative_5_task2_0
OUT_DIR=figs/builderbench/c5t2_nfbwd_nf_binary_acc_prepost_stoch
OUT_STEM="${OUT_DIR}/c5t2_nfbwd_s0_stoch_ckpts300_400_1000_1400"
ITERS=300,400,1000,1400

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== c5t2 nfbwd NF binary acc before/after success (stochastic) ==="
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "run_dir=${RUN_DIR}"
echo "iterations=${ITERS}"
echo "policy=stochastic  pairing=future achieved goal  num_envs=32"
nvidia-smi || true
python -u -c "import os, warp, jax; print('warp', warp.__version__); print('JAX', jax.default_backend(), jax.devices())"

python -u scripts/eval_builderbench_nf_binary_acc_prepost.py \
  --run-dir="${RUN_DIR}" \
  --iterations="${ITERS}" \
  --seed=0 \
  --num-envs=32 \
  --min-samples=32 \
  --future-discount=0.99 \
  --output-csv="${OUT_STEM}.csv" \
  --output-plot="${OUT_STEM}.png"

echo "done"
ls -lh "${OUT_STEM}".csv "${OUT_STEM}".png 2>/dev/null || true
