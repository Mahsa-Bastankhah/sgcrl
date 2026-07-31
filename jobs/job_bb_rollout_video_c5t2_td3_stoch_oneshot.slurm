#!/bin/bash
#SBATCH --job-name=bb_vid_c5t2_td3_stoch
#SBATCH --output=slurm/bb_vid_c5t2_td3_stoch_%j.log
#SBATCH --error=slurm/bb_vid_c5t2_td3_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas

# Stochastic-policy videos for c5t2 TD3 logQ seed1 (job 3603869).
# Renders two late checkpoints only.
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

RUN_DIR=logs/ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_actorreset_nopermute_normobs_evalvid/ppo_builderbench_creative_5_task2_1
CKPT_DIR="${RUN_DIR}/checkpoints"
OUT_DIR=videos/builderbench/ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_actorreset_nopermute_normobs_evalvid_seed1_stoch
RUN_TAG=c5t2_td3_seed1_stoch
STAGE_DIR="${TMPDIR}/bb_vid_stoch_${SLURM_JOB_ID}"
mkdir -p slurm "${OUT_DIR}" "${STAGE_DIR}"

# A couple of late checkpoints.
for ckpt in ckpt_iter_0003600.pkl ckpt_iter_0003905.pkl; do
  if [ ! -f "${CKPT_DIR}/${ckpt}" ]; then
    echo "ERROR: missing ${CKPT_DIR}/${ckpt}"
    exit 2
  fi
  ln -sfn "$(realpath "${CKPT_DIR}/${ckpt}")" "${STAGE_DIR}/${ckpt}"
done
# So run_config.json is discoverable via realpath of symlink targets.
echo "RUN_DIR=${RUN_DIR}"
echo "OUT_DIR=${OUT_DIR}  RUN_TAG=${RUN_TAG}  stochastic=true"
ls -l "${STAGE_DIR}"
nvidia-smi || true

python -u scripts/ppo_builderbench_rollout_video.py \
  --checkpoint="${STAGE_DIR}" \
  --env=builderbench_creative_5_task2 \
  --output="${OUT_DIR}/" \
  --run_tag="${RUN_TAG}" \
  --seed=1 \
  --stochastic \
  --fps=10

PY_EXIT=$?
rm -rf "${STAGE_DIR}"
ls -lh "${OUT_DIR}/${RUN_TAG}"_*.mp4 2>&1 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
