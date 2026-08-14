#!/bin/bash
#SBATCH --job-name=lp_bin_rew_eo
#SBATCH --output=slurm/lp_vid_cpc_sawyer_bin_reward_everyother_%j.log
#SBATCH --error=slurm/lp_vid_cpc_sawyer_bin_reward_everyother_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=seas

# Online φ·ψ reward-strip videos for EVERY OTHER retained learner ckpt of
# lp_contrastive sawyer_bin contrastive_cpc seed42 (job 3677974).
# Acme keeps max_to_keep=10, so this is every other of ckpt-62..71.
unset LD_PRELOAD
module purge

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

try_cuda() {
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    echo "Loaded CUDA modules: $*"
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || true
if [ -z "${CUDA_HOME:-}" ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      echo "Fallback CUDA_HOME=${CUDA_HOME}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

RUN_DIR=logs/lp_contrastive_sawyer_bin/contrastive_cpc_sawyer_bin_42
OUT_DIR=figs/sawyer_bin/frozen_crl_reward_probe/lp_cpc_sawyer_bin_s42_everyother
mkdir -p slurm "${OUT_DIR}"

echo "Rendering EVERY OTHER LP contrastive_cpc sawyer_bin s42 φ·ψ videos → ${OUT_DIR}"
echo "reward: r = phi(s,a)·psi(g)  ckpt_stride=2  run_dir=${RUN_DIR}"
nvidia-smi || true

python -u scripts/render_lp_crl_traj_reward_video.py \
  --run_dir="${RUN_DIR}" \
  --env=sawyer_bin \
  --ckpt_stride=2 \
  --out_dir="${OUT_DIR}" \
  --tag_prefix=lp_cpc_sawyer_bin_s42_eo \
  --title='online CRL  $r=\varphi(s,a)\cdot\psi(g)$' \
  --allow_no_success \
  --max_tries=8 \
  --seed=0 \
  --fps=15 \
  --camera=corner \
  --rotate=180 \
  --skip_existing

ls -lh "${OUT_DIR}"/*.mp4 2>&1 | tail -40 || true
pkill -P $$ 2>/dev/null || true
