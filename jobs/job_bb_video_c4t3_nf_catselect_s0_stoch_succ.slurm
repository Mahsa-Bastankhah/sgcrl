#!/bin/bash
#SBATCH --job-name=c4t3_nf_s0_succ
#SBATCH --output=slurm/bb_vid_c4t3_nf_catselect_s0_stoch_succ_%j.log
#SBATCH --error=slurm/bb_vid_c4t3_nf_catselect_s0_stoch_succ_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Stochastic success-search video for creative4_task3 NF compact catselect
# (no extrew) seed 0 — the only no-reward seed with sustained success.
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

ENV_NAME=builderbench_creative_4_task3
RUN=logs/ppo_builderbench_creative4_task3_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_norand_evalvid_catselect/ppo_builderbench_creative_4_task3_0
OUT_DIR=videos/builderbench/c4t3_nf_catselect_s0_stoch_succ
mkdir -p slurm "${OUT_DIR}"

echo "Searching stochastic successes → ${OUT_DIR}"
nvidia-smi || true

PY_EXIT=0

render_until_succ() {
  local it="$1" max_attempts="$2"
  local ckpt tag
  ckpt=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "$RUN" "$it")
  tag=$(printf 'c4t3_nf_cat_s0_stoch_it%07d' "$it")
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing ${ckpt}"
    PY_EXIT=1
    return
  fi
  echo "[c4t3_nf_vid] stochastic require_success tag=${tag} attempts=${max_attempts}"
  python -u scripts/ppo_builderbench_rollout_video.py \
    --checkpoint="${ckpt}" \
    --env="${ENV_NAME}" \
    --output="${OUT_DIR}/" \
    --run_tag="${tag}" \
    --seed=0 \
    --stochastic \
    --max_seed_attempts="${max_attempts}" \
    --require_success \
    --skip_existing || PY_EXIT=$?
}

# Prefer late eval-success iters (eval=1.0 @ 3400/3600), then peak-train window.
for it in 3600 3300 3900 2400 2100; do
  # Stop early once we have at least one success video.
  if ls "${OUT_DIR}"/*succ*.mp4 >/dev/null 2>&1; then
    echo "already have success video(s); skipping remaining iters"
    break
  fi
  render_until_succ "${it}" 48
done

echo "=== successful videos ==="
ls -lh "${OUT_DIR}"/*succ*.mp4 2>&1 || echo '(none)'
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
