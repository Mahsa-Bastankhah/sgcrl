#!/bin/bash
#SBATCH --job-name=c4t3_nf_ms_succ
#SBATCH --output=slurm/bb_vid_c4t3_nf_minstd_entanneal_stoch_succ_%j.log
#SBATCH --error=slurm/bb_vid_c4t3_nf_minstd_entanneal_stoch_succ_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Stochastic success-search for creative4_task3 NF compact + minstd/entanneal
# (no extrew) — seed 1 intermittent successes, seed 0 brief spike @3200.
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
ROOT=logs/ppo_builderbench_creative4_task3_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_norand_minstd1e5_entanneal_evalvid_catselect
OUT_DIR=videos/builderbench/c4t3_nf_minstd_entanneal_stoch_succ
mkdir -p slurm "${OUT_DIR}"

echo "Searching stochastic successes → ${OUT_DIR}"
nvidia-smi || true

PY_EXIT=0

render_until_succ() {
  local run="$1" seed_tag="$2" it="$3" max_attempts="$4"
  local ckpt tag
  ckpt=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "$run" "$it")
  tag=$(printf 'c4t3_nf_ms_ea_%s_stoch_it%07d' "$seed_tag" "$it")
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing ${ckpt}"
    return
  fi
  if ls "${OUT_DIR}"/*succ*.mp4 >/dev/null 2>&1; then
    echo "already have success video(s); skip ${tag}"
    return
  fi
  echo "[c4t3_nf_ms_vid] stochastic require_success tag=${tag} attempts=${max_attempts}"
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

RUN1="${ROOT}/ppo_builderbench_creative_4_task3_1"
RUN0="${ROOT}/ppo_builderbench_creative_4_task3_0"

# Seed 1: eval success @600/1600/2600/2800 → nearest ckpts
for it in 2700 3000 1800 1500 600; do
  render_until_succ "${RUN1}" "s1" "${it}" 48
done

# Seed 0 fallback: brief eval success @3200
for it in 3300 3000; do
  render_until_succ "${RUN0}" "s0" "${it}" 48
done

echo "=== successful videos ==="
ls -lh "${OUT_DIR}"/*succ*.mp4 2>&1 || echo '(none)'
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
