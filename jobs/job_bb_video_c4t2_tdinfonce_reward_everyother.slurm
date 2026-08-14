#!/bin/bash
#SBATCH --job-name=c4t2_tdi_rew_eo
#SBATCH --output=slurm/bb_vid_c4t2_tdinfonce_reward_everyother_%j.log
#SBATCH --error=slurm/bb_vid_c4t2_tdinfonce_reward_everyother_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=seas

# Online φ·ψ reward-strip videos for EVERY OTHER milestone of
# c4t2 TD-InfoNCE catselect (NO external reward) seed0.
# Reward probe: r = φ(s,a)·ψ(g) from online q_params (same as CRL strip).
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

CKPT_DIR=logs/ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_catselect/ppo_builderbench_creative_4_task2_0/checkpoints
OUT_DIR=figs/builderbench/tdinfonce_reward_probe/c4t2_tdinfonce_catselect_s0_everyother
mkdir -p slurm "${OUT_DIR}"

echo "Rendering EVERY OTHER c4t2 TD-InfoNCE (no extrew) seed0 online φ·ψ videos → ${OUT_DIR}"
echo "reward: r = phi(s,a)·psi(g)  ckpt_stride=2"
nvidia-smi || true

python -u scripts/render_frozen_crl_traj_reward_video.py \
  --checkpoint_dir="${CKPT_DIR}" \
  --ckpt_stride=2 \
  --env=builderbench_creative_4_task2 \
  --out_dir="${OUT_DIR}" \
  --tag_prefix=c4t2_tdinfonce_catselect_s0_eo \
  --title='online TD-InfoNCE  $r=\varphi(s,a)\cdot\psi(g)$' \
  --allow_no_success \
  --max_tries=20 \
  --seed=0 \
  --fps=8 \
  --skip_existing

ls -lh "${OUT_DIR}"/*.mp4 2>&1 | tail -50 || true
pkill -P $$ 2>/dev/null || true
