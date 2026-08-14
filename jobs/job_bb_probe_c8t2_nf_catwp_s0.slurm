#!/bin/bash
#SBATCH --job-name=probe_c8t2_nf
#SBATCH --output=slurm/bb_probe_c8t2_nf_catwp_s0_%j.log
#SBATCH --error=slurm/bb_probe_c8t2_nf_catwp_s0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=seas

# Online log-p reward + r/σ + ‖∇_s log p‖ probe videos for current c8t2 NF
# compact catwp checkpoints (skip iter 0). One compile via a symlink farm.
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
CKPT_DIR=logs/ppo_builderbench_creative8_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl10/ppo_builderbench_creative_8_task2_0/checkpoints
OUT_DIR=figs/builderbench/nf_logp_reward_probe/c8t2_nf_catwp_ms_ea_s0_online_all
TAG_PREFIX=c8t2_nf_catwp_ms_ea_s0_online

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}"
STAGE=$(mktemp -d "${TMPDIR}/c8t2_nf_probe_XXXX")
trap 'rm -rf "${STAGE}"' EXIT

echo "=== c8t2 NF catwp probe (skip iter 0) ==="
echo "ckpt_dir=${CKPT_DIR}"
echo "out_dir=${OUT_DIR}"
nvidia-smi || true

shopt -s nullglob
n=0
for f in "${CKPT_DIR}"/ckpt_iter_*.pkl; do
  base=$(basename "${f}")
  iter_num=$(echo "${base}" | sed -n 's/ckpt_iter_0*\([0-9]*\)\.pkl/\1/p')
  if [[ "${iter_num}" == "0" || -z "${iter_num}" ]]; then
    echo "[probe] skip ${base}"
    continue
  fi
  ln -s "$(realpath "${f}")" "${STAGE}/${base}"
  echo "[probe] stage ${base}"
  n=$((n + 1))
done
echo "[probe] ${n} checkpoint(s) staged in ${STAGE}"
ls -l "${STAGE}"

python -u scripts/render_nf_traj_reward_video.py \
  --checkpoint_dir="${STAGE}" \
  --env="${ENV_NAME}" \
  --out_dir="${OUT_DIR}" \
  --tag_prefix="${TAG_PREFIX}" \
  --allow_no_success \
  --max_tries=1 \
  --seed=0 \
  --fps=8 \
  --normalize_reward \
  --show_logp_grad \
  --skip_existing
PY_EXIT=$?

echo ""
echo "=== probe finished (PY_EXIT=${PY_EXIT}) ==="
ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null | tail -20 || true
pkill -P $$ 2>/dev/null || true
exit ${PY_EXIT}
