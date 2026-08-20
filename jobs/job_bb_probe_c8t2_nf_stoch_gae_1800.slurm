#!/bin/bash
#SBATCH --job-name=c8t2_nf_stoch
#SBATCH --output=slurm/bb_probe_c8t2_nf_stoch_gae_1800_%j.log
#SBATCH --error=slurm/bb_probe_c8t2_nf_stoch_gae_1800_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas

# Stochastic c8t2 NF catwp probes at iter 1800 (seeds 0,1,2) with GAE strip.
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

CKPT=logs/ppo_builderbench_creative8_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl10/ppo_builderbench_creative_8_task2_0/checkpoints/ckpt_iter_0001800.pkl
OUT_DIR=figs/builderbench/nf_logp_reward_probe/c8t2_nf_catwp_ms_ea_stoch_1800
TAG_PREFIX=c8t2_nf_catwp_ms_ea

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}"

echo "=== c8t2 NF stochastic+GAE probe iter 1800 seeds 0,1,2 ==="
nvidia-smi || true

python -u scripts/render_nf_traj_reward_video.py \
  --checkpoint="${CKPT}" \
  --env=builderbench_creative_8_task2 \
  --out_dir="${OUT_DIR}" \
  --tag_prefix="${TAG_PREFIX}" \
  --allow_no_success \
  --max_tries=1 \
  --stochastic \
  --seeds=0,1,2 \
  --fps=8 \
  --normalize_reward \
  --show_logp_grad \
  --show_policy_loc_scale \
  --show_gae

echo "=== done ==="
ls -lh "${OUT_DIR}"/*.mp4 || true
pkill -P $$ 2>/dev/null || true
