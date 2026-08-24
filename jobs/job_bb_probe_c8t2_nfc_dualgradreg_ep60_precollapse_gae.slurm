#!/bin/bash
#SBATCH --job-name=c8t2_ep60_precol_gae
#SBATCH --output=slurm/bb_probe_c8t2_nfc_dualgradreg_ep60_precollapse_gae_%j.log
#SBATCH --error=slurm/bb_probe_c8t2_nfc_dualgradreg_ep60_precollapse_gae_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:59:00
#SBATCH --account=seas

# 3744740 c8t2 ep60 dualgradreg: reward + GAE A_t + V around the collapse.
# Train takeoff ~60M (iter 976), peak ~65M (1056), collapse ~69M (1125).
# Five ckpts (interval 100): 800, 900, 1000, 1100, 1200.
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

ENV_NAME=builderbench_creative_8_task2
CKPT_DIR=logs/ppo_builderbench_creative8_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent005_to001_ep60_300m_crl10_dualgradreg_c100_lamlr1e6/ppo_builderbench_creative_8_task2_0/checkpoints
OUT_DIR=figs/builderbench/nf_logp_reward_probe/c8t2_nfc_catwp_dualgradreg_c100_lamlr1e6_ent005to001_ep60_s0_precollapse_gae
TAG_PREFIX=c8t2_nfc_catwp_dualgradreg_c100_lamlr1e6_ent005to001_ep60_s0_precollapse_gae

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== c8t2 ep60 dualgradreg reward+V/GAE (5 ckpts around collapse) ==="
echo "ckpt_dir=${CKPT_DIR}"
echo "out_dir=${OUT_DIR}"
nvidia-smi || true

CKPTS=()
for IT in 800 900 1000 1100 1200; do
  P=$(printf '%s/ckpt_iter_%07d.pkl' "${CKPT_DIR}" "${IT}")
  if [ ! -f "${P}" ]; then
    echo "ERROR: missing ${P}"
    exit 1
  fi
  CKPTS+=("${P}")
done

echo "selected ${#CKPTS[@]} checkpoint(s):"
printf '  %s\n' "${CKPTS[@]}"

PY_EXIT=0
for CKPT in "${CKPTS[@]}"; do
  echo ""
  echo "[probe] rendering ${CKPT}"
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
    --show_gae \
    --skip_existing || PY_EXIT=$?
done

echo ""
echo "=== probe finished (PY_EXIT=${PY_EXIT}) ==="
ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null || true
pkill -P $$ 2>/dev/null || true
exit ${PY_EXIT}
