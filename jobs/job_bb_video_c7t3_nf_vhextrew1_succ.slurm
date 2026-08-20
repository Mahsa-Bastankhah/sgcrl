#!/bin/bash
#SBATCH --job-name=c7t3_nf_succ
#SBATCH --output=slurm/bb_vid_c7t3_nf_vhextrew1_succ_%A_%a.log
#SBATCH --error=slurm/bb_vid_c7t3_nf_vhextrew1_succ_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas
#SBATCH --array=0-1

# Deterministic-policy videos from the c7t3 NF vhextrew1 success window
# (eval success=1 at 1800–1920; ckpts every 100). Search seeds until success=1.
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

ENV_NAME=builderbench_creative_7_task3
OUT_DIR=videos/builderbench/c7t3_nf_vhextrew1_succ
mkdir -p slurm "${OUT_DIR}"

RUN=logs/ppo_builderbench_creative7_task3_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl10_vhextrew1/ppo_builderbench_creative_7_task3_0
TAG=c7t3_nf_catwp_vhextrew1_s0

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # first eval success=1 (hard only; very_hard=0); train_success_1000=0.384
    IT=1800
    ;;
  1)
    # peak of the window: eval hard=1 and very_hard=1 at 1890/1920
    # train_success_1000=0.861, train_very_hard_1000=0.651
    IT=1900
    ;;
  *)
    echo "bad array id ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

CKPT=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "${RUN}" "${IT}")
echo "c7t3 NF vhextrew1 succ video tag=${TAG} iter=${IT}"
echo "CKPT=${CKPT}"
nvidia-smi || true
if [[ ! -f "${CKPT}" ]]; then
  echo "missing ${CKPT}"
  exit 1
fi

python -u scripts/ppo_builderbench_rollout_video.py \
  --checkpoint="${CKPT}" \
  --env="${ENV_NAME}" \
  --output="${OUT_DIR}/" \
  --run_tag="${TAG}" \
  --seed=0 \
  --max_seed_attempts=32 \
  --require_success \
  --skip_existing

ls -lh "${OUT_DIR}/${TAG}"*iter*$(printf '%07d' "${IT}")* 2>/dev/null || ls -lh "${OUT_DIR}" | tail
pkill -P $$ 2>/dev/null || true
