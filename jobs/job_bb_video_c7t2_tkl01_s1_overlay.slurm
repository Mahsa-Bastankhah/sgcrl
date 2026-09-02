#!/bin/bash
#SBATCH --job-name=c7t2_tkl_vid
#SBATCH --output=slurm/bb_vid_c7t2_tkl01_s1_overlay_%j.log
#SBATCH --error=slurm/bb_vid_c7t2_tkl01_s1_overlay_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --account=seas
#SBATCH --exclude=neu303

# Re-render c7t2 tKL seed1 ckpts 2800/3200. Deterministic. Top bar = SUCCESS=1.
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
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

ENV_NAME=builderbench_creative_7_task2
RUN=logs/final_runs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep70_300m_crl10_dualgradreg_c100_lamlr1e6_tkl01_warp_4h/ppo_builderbench_creative_7_task2_1
OUT_DIR=videos/builderbench/c7t2_tkl01_s1_succ_overlay
mkdir -p slurm "${OUT_DIR}"

echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
nvidia-smi || true
python -u -c "import os, warp, jax; print('warp', warp.__version__); print('JAX', jax.default_backend(), jax.devices())"

PY_EXIT=0
for it in 2800 3200; do
  ckpt=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "${RUN}" "${it}")
  echo "[overlay] iter=${it} ckpt=${ckpt}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing ${ckpt}"
    PY_EXIT=1
    continue
  fi
  python -u scripts/ppo_builderbench_rollout_video.py \
    --checkpoint="${ckpt}" \
    --env="${ENV_NAME}" \
    --output="${OUT_DIR}/" \
    --run_tag="c7t2_tkl01_s1" \
    --seed=0 \
    --max_seed_attempts=16 \
    --require_success || PY_EXIT=$?
done

ls -lh "${OUT_DIR}" || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
