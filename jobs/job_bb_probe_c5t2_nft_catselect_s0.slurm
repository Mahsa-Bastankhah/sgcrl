#!/bin/bash
#SBATCH --job-name=probe_c5t2_nft
#SBATCH --output=slurm/bb_probe_c5t2_nft_catselect_s0_%j.log
#SBATCH --error=slurm/bb_probe_c5t2_nft_catselect_s0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas

# Probe watcher: renders log-p reward + ‖∇_s log p‖ probe videos at 10 evenly
# spaced checkpoints of the c5t2 NF tiny catselect run (seed 0).
# Polls every 5 min for each target checkpoint; times out after 90 min/ckpt.
# Produces: raw log-p strip + normalised (r/σ) strip + grad-norm strip.
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

# ---- config ----------------------------------------------------------------
ENV_NAME=builderbench_creative_5_task2
CKPT_DIR=logs/ppo_builderbench_creative5_task2_e1024_pd_nf_tiny_sa2x128_r32_b4_w128_tau05_nopermute_catselect_spert001/ppo_builderbench_creative_5_task2_0/checkpoints
OUT_DIR=figs/builderbench/nf_logp_reward_probe/c5t2_nft_catselect_s0_online_all
TAG_PREFIX=c5t2_nft_catselect_s0_online

TARGET_ITERS="300 900 1200 1500 1800 2400 2700 3000 3600 3900"
POLL_INTERVAL_SECS=300
TIMEOUT_POLLS=18

mkdir -p slurm "${OUT_DIR}"
echo "=== c5t2 NF tiny catselect probe watcher (seed 0) ==="
echo "ckpt_dir=${CKPT_DIR}"
echo "out_dir=${OUT_DIR}"
echo "targets=${TARGET_ITERS}"
nvidia-smi || true

PY_EXIT=0
for ITER in $TARGET_ITERS; do
  CKPT=$(printf '%s/ckpt_iter_%07d.pkl' "${CKPT_DIR}" "${ITER}")
  echo ""
  echo "[probe] waiting for iter=${ITER}  path=${CKPT}"

  polls=0
  while [[ ! -f "${CKPT}" ]]; do
    if [[ $polls -ge $TIMEOUT_POLLS ]]; then
      echo "[probe] TIMEOUT after $((polls * POLL_INTERVAL_SECS / 60)) min — skipping iter ${ITER}"
      break
    fi
    sleep "${POLL_INTERVAL_SECS}"
    polls=$((polls + 1))
    echo "[probe]   still waiting... (${polls}/${TIMEOUT_POLLS})"
  done

  if [[ ! -f "${CKPT}" ]]; then
    continue
  fi

  echo "[probe] rendering iter=${ITER}"
  python -u scripts/render_nf_traj_reward_video.py \
    --checkpoint="${CKPT}" \
    --env="${ENV_NAME}" \
    --out_dir="${OUT_DIR}" \
    --tag_prefix="${TAG_PREFIX}" \
    --allow_no_success \
    --max_tries=1 \
    --seed=0 \
    --fps=8 \
    --normalize_reward \
    --show_logp_grad \
    --skip_existing || PY_EXIT=$?

  echo "[probe] done iter=${ITER}  exit=${PY_EXIT}"
done

echo ""
echo "=== probe watcher finished (PY_EXIT=${PY_EXIT}) ==="
ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null | tail -20 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
