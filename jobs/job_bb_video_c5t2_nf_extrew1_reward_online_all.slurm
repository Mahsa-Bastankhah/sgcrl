#!/bin/bash
#SBATCH --job-name=c5t2_nf_e1_rew
#SBATCH --output=slurm/bb_vid_c5t2_nf_extrew1_reward_online_all_%j.log
#SBATCH --error=slurm/bb_vid_c5t2_nf_extrew1_reward_online_all_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=seas

# Online-only NF log-prob reward-strip videos for ALL milestones of
# c5t2 NF (full SA) extrew1 seed1 (best success among seeds 0/1/2).
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

CKPT_DIR=logs/ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_evalvid_catselect_extrew1/ppo_builderbench_creative_5_task2_1/checkpoints
OUT_DIR=figs/builderbench/nf_logp_reward_probe/c5t2_nf_extrew1_s1_online_all
mkdir -p slurm "${OUT_DIR}"

echo "Rendering ALL c5t2 NF extrew1 seed1 online logp reward-strip videos → ${OUT_DIR}"
nvidia-smi || true

python -u scripts/render_nf_traj_reward_video.py \
  --checkpoint_dir="${CKPT_DIR}" \
  --env=builderbench_creative_5_task2 \
  --out_dir="${OUT_DIR}" \
  --tag_prefix=c5t2_nf_extrew1_s1_online \
  --allow_no_success \
  --max_tries=20 \
  --seed=0 \
  --fps=8 \
  --skip_existing

ls -lh "${OUT_DIR}"/*.mp4 2>&1 | tail -50 || true
pkill -P $$ 2>/dev/null || true
