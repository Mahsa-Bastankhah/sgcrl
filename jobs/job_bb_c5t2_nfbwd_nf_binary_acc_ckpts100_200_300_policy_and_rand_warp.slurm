#!/bin/bash
#SBATCH --job-name=c5t2_nfbwd_binacc_early
#SBATCH --output=slurm/bb_c5t2_nfbwd_nf_binary_acc_ckpts100_200_300_policy_and_rand_warp_%j.log
#SBATCH --error=slurm/bb_c5t2_nfbwd_nf_binary_acc_ckpts100_200_300_policy_and_rand_warp_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Binary rank acc at ckpts 100/200 (pre-lock) and 300 (first success lock).
# Each ckpt uses its own policy + NF + learner goal mean/std.
# Then the same NFs scored on uniform[-1,1] random rollouts.
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
OUT_DIR=figs/builderbench/c5t2_nfbwd_nf_binary_acc_ckpts100_200_300
ITERS=100,200,300

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== c5t2 nfbwd NF binary acc ckpts 100/200/300 (policy + random) ==="
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "run_dir=${RUN_DIR}"
echo "iterations=${ITERS}"
echo "goal_norm=learner nf/goal_mean_* nf/goal_std_* at that iteration"
nvidia-smi || true
python -u -c "import os, warp, jax; print('warp', warp.__version__); print('JAX', jax.default_backend(), jax.devices())"

python -u scripts/eval_builderbench_nf_binary_acc_prepost.py \
  --run-dir="${RUN_DIR}" \
  --iterations="${ITERS}" \
  --seed=0 \
  --num-envs=32 \
  --min-samples=32 \
  --future-discount=0.99 \
  --policy=checkpoint \
  --output-csv="${OUT_DIR}/c5t2_nfbwd_s0_policy_ckpts100_200_300.csv" \
  --output-plot="${OUT_DIR}/c5t2_nfbwd_s0_policy_ckpts100_200_300.png"

python -u scripts/eval_builderbench_nf_binary_acc_prepost.py \
  --run-dir="${RUN_DIR}" \
  --iterations="${ITERS}" \
  --seed=0 \
  --num-envs=32 \
  --min-samples=32 \
  --future-discount=0.99 \
  --policy=random \
  --output-csv="${OUT_DIR}/c5t2_nfbwd_s0_rand_ckpts100_200_300.csv" \
  --output-plot="${OUT_DIR}/c5t2_nfbwd_s0_rand_ckpts100_200_300.png"

echo "done"
ls -lh "${OUT_DIR}"/*.csv "${OUT_DIR}"/*.png 2>/dev/null || true
