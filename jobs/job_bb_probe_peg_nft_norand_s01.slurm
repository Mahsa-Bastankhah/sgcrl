#!/bin/bash
#SBATCH --job-name=probe_peg_nft
#SBATCH --output=slurm/bb_probe_peg_nft_norand_s%a_%j.log
#SBATCH --error=slurm/bb_probe_peg_nft_norand_s%a_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-1

# Online NF log-p + r/σ + ‖∇_s log p‖ probe videos for the norand peg tiny
# evalvid seeds (array 2 → seed 0, array 3 → seed 1). Skip iter 0.
# ~90 current milestones per seed; skip_existing so a rerun continues.
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

SEED=${SLURM_ARRAY_TASK_ID:-0}
CKPT_DIR=logs/ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_norand_minstd1e5_evalvid/ppo_sawyer_peg_${SEED}/checkpoints
OUT_DIR=figs/sawyer_peg/nf_logp_reward_probe/peg_nft_norand_ms_s${SEED}_online_all
TAG_PREFIX=peg_nft_norand_ms_s${SEED}_online

mkdir -p slurm "${OUT_DIR}"
echo "=== peg NF tiny norand probe seed=${SEED} (skip iter 0) ==="
echo "ckpt_dir=${CKPT_DIR}"
echo "out_dir=${OUT_DIR}"
nvidia-smi || true
ls "${CKPT_DIR}"/ckpt_iter_*.pkl | wc -l

python -u scripts/render_sawyer_nf_traj_reward_video.py \
  --checkpoint_dir="${CKPT_DIR}" \
  --env=sawyer_peg \
  --out_dir="${OUT_DIR}" \
  --tag_prefix="${TAG_PREFIX}" \
  --allow_no_success \
  --max_tries=1 \
  --seed="${SEED}" \
  --fps=15 \
  --camera=corner \
  --rotate=180 \
  --min_iter=1 \
  --normalize_reward \
  --show_logp_grad \
  --skip_existing
PY_EXIT=$?

echo ""
echo "=== probe finished seed=${SEED} (PY_EXIT=${PY_EXIT}) ==="
ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null | tail -20 || true
pkill -P $$ 2>/dev/null || true
exit ${PY_EXIT}
