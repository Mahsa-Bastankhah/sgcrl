#!/bin/bash
#SBATCH --job-name=c7t2_crl_bestv
#SBATCH --output=slurm/bb_vid_c7t2_crl_best_succ_%A_%a.log
#SBATCH --error=slurm/bb_vid_c7t2_crl_best_succ_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas
#SBATCH --array=0-4

# Deterministic-policy videos from the strongest c7t2 CRL checkpoints
# (peak train_success_1000 ~0.90–0.96). Search env seeds until success=1.
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
OUT_DIR=videos/builderbench/c7t2_crl_best_succ
mkdir -p slurm "${OUT_DIR}"

LOGS=logs
C7=ppo_builderbench_creative7_task2_e1024_pd
SEED0=ppo_builderbench_creative_7_task2_0

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # peak train 0.958@1264; ckpt 1350 train=0.933; eval=1 at 1400
    RUN="${LOGS}/${C7}_crl_tau05_nopermute_fixedx01_catselect_extrew1_minstd1e5_entanneal_ep70_300m/${SEED0}"
    TAG=c7t2_crl_catselect_ms_ea_ep70_s0
    IT=1350
    ;;
  1)
    # same run; ckpt 1200 train=0.908; eval=1 at 1200
    RUN="${LOGS}/${C7}_crl_tau05_nopermute_fixedx01_catselect_extrew1_minstd1e5_entanneal_ep70_300m/${SEED0}"
    TAG=c7t2_crl_catselect_ms_ea_ep70_s0
    IT=1200
    ;;
  2)
    # catwp crl10; peak 0.943@725; ckpt 600 train=0.899; eval=1 at 600
    RUN="${LOGS}/${C7}_crl_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl10/${SEED0}"
    TAG=c7t2_crl_catwp_ep60_crl10_s0
    IT=600
    ;;
  3)
    # catwp ep70; peak 0.925@515; ckpt 600 train=0.874; eval=1 at 400/600
    RUN="${LOGS}/${C7}_crl_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m/${SEED0}"
    TAG=c7t2_crl_catwp_ep70_s0
    IT=600
    ;;
  4)
    # catwp crl8; peak 0.907@594; ckpt 600 train=0.897; eval=1 at 600
    RUN="${LOGS}/${C7}_crl_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl8/${SEED0}"
    TAG=c7t2_crl_catwp_ep60_crl8_s0
    IT=600
    ;;
  *)
    echo "bad array id ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

CKPT=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "${RUN}" "${IT}")
echo "c7t2 CRL best-succ video tag=${TAG} iter=${IT}"
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
