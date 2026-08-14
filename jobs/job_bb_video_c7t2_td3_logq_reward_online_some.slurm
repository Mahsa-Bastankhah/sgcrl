#!/bin/bash
#SBATCH --job-name=c7t2_td3rew
#SBATCH --output=slurm/bb_vid_c7t2_td3_logq_reward_online_some_%j.log
#SBATCH --error=slurm/bb_vid_c7t2_td3_logq_reward_online_some_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# Online TD3 log-Q reward-strip videos for a subset of milestones of the
# best-method c7t2 TD3 run (actorreset + catwp + tol0013, seed0).
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

CKPT_DIR=logs/ppo_builderbench_creative7_task2_e1024_pd_td3_logq_tau05_actorreset_nopermute_normobs_evalvid_catwp_tol0013/ppo_builderbench_creative_7_task2_0/checkpoints
OUT_DIR=figs/builderbench/td3_logq_reward_probe/c7t2_td3_catwp_tol0013_s0_online_some
TAG_PREFIX=c7t2_td3_catwp_tol0013_s0_online
# sparse subset across training (run only reached ~2857 iters / 0 success)
ITERS=(0000000 0000600 0001200 0001800 0002400 0002700)

mkdir -p slurm "${OUT_DIR}"

echo "Rendering ${#ITERS[@]} c7t2 TD3 logq reward-strip videos → ${OUT_DIR}"
nvidia-smi || true

for it in "${ITERS[@]}"; do
  ckpt="${CKPT_DIR}/ckpt_iter_${it}.pkl"
  if [ ! -f "${ckpt}" ]; then
    echo "[skip] missing ${ckpt}"
    continue
  fi
  echo "=== ckpt_iter_${it} ==="
  python -u scripts/render_td3_traj_reward_video.py \
    --checkpoint="${ckpt}" \
    --env=builderbench_creative_7_task2 \
    --out_dir="${OUT_DIR}" \
    --tag="${TAG_PREFIX}_iter_${it}" \
    --allow_no_success \
    --max_tries=5 \
    --seed=0 \
    --fps=8 \
    --skip_existing
done

ls -lh "${OUT_DIR}"/*.mp4 2>&1 || true
pkill -P $$ 2>/dev/null || true
