#!/bin/bash
#SBATCH --job-name=c4t2_hit_sf_t03
#SBATCH --output=slurm/bb_vid_c4t2_hitbonus_sf_tol003_reward_online_all_%j.log
#SBATCH --error=slurm/bb_vid_c4t2_hitbonus_sf_tol003_reward_online_all_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# Online CRL reward-strip videos for c4t2 hitbonus_sf tol=0.03 seed0:
#   r = φ(s,a)·ψ(g) + 1{‖s_g − g‖ < 0.03}
# Reads hit mode/tol from the run's run_config.json; --show_hit adds the
# indicator (+ distance) strip under the reward strip.
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

CKPT_DIR=logs/ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_catselect_hitbonus_sf_tol003/ppo_builderbench_creative_4_task2_0/checkpoints
OUT_DIR=figs/builderbench/frozen_crl_reward_probe/c4t2_hitbonus_sf_tol003_s0_online_all
mkdir -p slurm "${OUT_DIR}"

echo "Rendering ALL c4t2 hitbonus_sf tol=0.03 seed0 videos WITH hit strip → ${OUT_DIR}"
echo "reward: r = phi(s,a)·psi(g) + 1{||sg-g|| < 0.03}"
nvidia-smi || true

rm -f "${OUT_DIR}"/c4t2_hitbonus_sf_tol003_s0_online_iter_*.mp4 \
      "${OUT_DIR}"/c4t2_hitbonus_sf_tol003_s0_online_iter_*_still.png \
      "${OUT_DIR}"/c4t2_hitbonus_sf_tol003_s0_online_iter_*.csv

python -u scripts/render_frozen_crl_traj_reward_video.py \
  --checkpoint_dir="${CKPT_DIR}" \
  --env=builderbench_creative_4_task2 \
  --out_dir="${OUT_DIR}" \
  --tag_prefix=c4t2_hitbonus_sf_tol003_s0_online \
  --allow_no_success \
  --max_tries=20 \
  --seed=0 \
  --fps=8 \
  --show_hit

ls -lh "${OUT_DIR}"/*.mp4 2>&1 | tail -50 || true
pkill -P $$ 2>/dev/null || true
