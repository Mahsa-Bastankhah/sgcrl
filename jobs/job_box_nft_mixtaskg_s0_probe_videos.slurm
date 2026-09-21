#!/bin/bash
#SBATCH --job-name=box_nft_vid
#SBATCH --output=slurm/box_nft_mixtaskg_s0_probe_%j.log
#SBATCH --error=slurm/box_nft_mixtaskg_s0_probe_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --account=seas

# NF reward-probe videos for the latest box tiny run (peg-winner recipe):
#   logs/ppo_box_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_rand_minstd1e5_ent0005_mixtaskg
# 38 ckpts (iter 0..29600). log p + r/σ + ‖∇_s log p‖. stochastic policy.
# allow_no_success (train/eval success stayed 0). Skip iter 0.
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

CKPT_DIR=logs/ppo_box_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_rand_minstd1e5_ent0005_mixtaskg/ppo_sawyer_box_0/checkpoints
OUT_DIR=figs/sawyer_box/nf_logp_reward_probe/box_nft_mixtaskg_s0_online_all
TAG_PREFIX=box_nft_mixtaskg_s0_online

mkdir -p slurm "${OUT_DIR}"
echo "=== box NF tiny mixtaskg probe seed=0  38 ckpts  log p + r/σ + grad  stochastic ==="
echo "ckpt_dir=${CKPT_DIR}"
echo "out_dir=${OUT_DIR}"
nvidia-smi || true
ls "${CKPT_DIR}"/ckpt_iter_*.pkl | wc -l

python -u scripts/render_sawyer_nf_traj_reward_video.py \
  --checkpoint_dir="${CKPT_DIR}" \
  --env=sawyer_box \
  --out_dir="${OUT_DIR}" \
  --tag_prefix="${TAG_PREFIX}" \
  --allow_no_success \
  --max_tries=1 \
  --seed=0 \
  --fps=15 \
  --camera=corner \
  --rotate=180 \
  --min_iter=1 \
  --normalize_reward \
  --show_logp_grad \
  --stochastic \
  --skip_existing
PY_EXIT=$?

echo ""
echo "=== probe finished (PY_EXIT=${PY_EXIT}) ==="
ls "${OUT_DIR}"/*.mp4 2>/dev/null | wc -l
ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null | tail -10 || true
pkill -P $$ 2>/dev/null || true
exit ${PY_EXIT}
