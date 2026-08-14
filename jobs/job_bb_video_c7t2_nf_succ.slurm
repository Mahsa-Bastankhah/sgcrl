#!/bin/bash
#SBATCH --job-name=c7t2_vid_nfs
#SBATCH --output=slurm/bb_vid_c7t2_nf_succ_%j.log
#SBATCH --error=slurm/bb_vid_c7t2_nf_succ_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Render creative7 task2 NF runs that reached eval success=1:
#   - NF extrew1 minstd+entanneal (success from eval iter 1400)
#   - NF extrew1 catselect ep70 (success from eval iter 2200)
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
OUT_DIR=videos/builderbench/c7t2_nf_succ
mkdir -p slurm "${OUT_DIR}"

echo "Rendering c7t2 NF success videos → ${OUT_DIR}"
nvidia-smi || true

PY_EXIT=0

render_det() {
  local run="$1" tag="$2" it="$3"
  local ckpt
  ckpt=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "$run" "$it")
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing ${ckpt}"
    PY_EXIT=1
    return
  fi
  echo "[c7t2_nf_vid] det tag=${tag} iter=${it}"
  python -u scripts/ppo_builderbench_rollout_video.py \
    --checkpoint="${ckpt}" \
    --env="${ENV_NAME}" \
    --output="${OUT_DIR}/" \
    --run_tag="${tag}" \
    --seed=0 \
    --skip_existing || PY_EXIT=$?
}

# NF extrew1 minstd+entanneal — eval success from 1400
NF_MS=logs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catselect_extrew1_minstd1e5_entanneal_ep70_300m/ppo_builderbench_creative_7_task2_0
for it in 1350 1500 1650 1800 1950; do
  render_det "${NF_MS}" "c7t2_nf_ext1_ms_ea_det" "${it}"
done

# NF extrew1 catselect ep70 — eval success from 2200
NF_BASE=logs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catselect_extrew1_ep70_300m/ppo_builderbench_creative_7_task2_0
for it in 2100 2250; do
  render_det "${NF_BASE}" "c7t2_nf_ext1_det" "${it}"
done

ls -lh "${OUT_DIR}"/*.mp4 2>&1 | tail -40 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
