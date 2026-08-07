#!/bin/bash
#SBATCH --job-name=c7t2_vid_nfs2
#SBATCH --output=slurm/bb_vid_c7t2_nf_succ_find_%j.log
#SBATCH --error=slurm/bb_vid_c7t2_nf_succ_find_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Search seeds until we get metrics["success"]=1 videos for the NF runs
# that reached eval success (peak train success windows).
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

ENV_NAME=builderbench_creative_7_task2
OUT_DIR=videos/builderbench/c7t2_nf_succ_verified
mkdir -p slurm "${OUT_DIR}"

echo "Searching for successful NF videos → ${OUT_DIR}"
nvidia-smi || true

PY_EXIT=0

render_until_succ() {
  local run="$1" tag="$2" it="$3" max_attempts="$4"
  local ckpt
  ckpt=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "$run" "$it")
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing ${ckpt}"
    PY_EXIT=1
    return
  fi
  echo "[c7t2_nf_vid] require_success tag=${tag} iter=${it} attempts=${max_attempts}"
  # Deterministic policy, varied env seeds (matches train eval style).
  python -u scripts/ppo_builderbench_rollout_video.py \
    --checkpoint="${ckpt}" \
    --env="${ENV_NAME}" \
    --output="${OUT_DIR}/" \
    --run_tag="${tag}" \
    --seed=0 \
    --max_seed_attempts="${max_attempts}" \
    --require_success \
    --skip_existing || PY_EXIT=$?
}

# NF extrew1 minstd+entanneal — peak train ~1669, eval success 1400/1600/1800
NF_MS=logs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catselect_extrew1_minstd1e5_entanneal_ep70_300m/ppo_builderbench_creative_7_task2_0
for it in 1650 1500 1800; do
  render_until_succ "${NF_MS}" "c7t2_nf_ext1_ms_ea" "${it}" 64
done

# NF extrew1 — peak train ~2201, eval success 2200
NF_BASE=logs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catselect_extrew1_ep70_300m_NOnormobs/ppo_builderbench_creative_7_task2_0
for it in 2250 2100; do
  render_until_succ "${NF_BASE}" "c7t2_nf_ext1" "${it}" 64
done

echo "=== successful videos ==="
ls -lh "${OUT_DIR}"/*succ*.mp4 2>&1 || echo '(none)'
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
