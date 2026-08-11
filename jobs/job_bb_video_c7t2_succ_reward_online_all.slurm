#!/bin/bash
#SBATCH --job-name=c7t2_rew_%a
#SBATCH --output=slurm/bb_vid_c7t2_succ_reward_online_all_%A_%a.log
#SBATCH --error=slurm/bb_vid_c7t2_succ_reward_online_all_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=seas
#SBATCH --array=0-4

# Online reward-strip + traj videos for ALL milestones of the c7t2 runs that
# reached high train_success_1000 (~0.88–0.98). skip_existing keeps NF probes
# that are already on disk; CRL catselect is re-rendered (dir was wiped).
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

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # best≈0.958 (collapsed); probe dir was deleted → full re-render
    KIND=crl
    CKPT_DIR=logs/ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_nopermute_fixedx01_catselect_extrew1_minstd1e5_entanneal_ep70_300m/ppo_builderbench_creative_7_task2_0/checkpoints
    OUT_DIR=figs/builderbench/frozen_crl_reward_probe/c7t2_crl_ms_ea_s0_online_all
    TAG_PREFIX=c7t2_crl_ms_ea_s0_online
    ;;
  1)
    # best≈0.968 (collapsed); probes mostly present
    KIND=nf
    CKPT_DIR=logs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catselect_extrew1_minstd1e5_entanneal_ep70_300m/ppo_builderbench_creative_7_task2_0/checkpoints
    OUT_DIR=figs/builderbench/nf_logp_reward_probe/c7t2_nf_ms_ea_s0_online_all
    TAG_PREFIX=c7t2_nf_ms_ea_s0_online
    ;;
  2)
    # best≈0.881 (collapsed); probes mostly present
    KIND=nf
    CKPT_DIR=logs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catselect_extrew1_ep70_300m_NOnormobs/ppo_builderbench_creative_7_task2_0/checkpoints
    OUT_DIR=figs/builderbench/nf_logp_reward_probe/c7t2_nf_base_s0_online_all
    TAG_PREFIX=c7t2_nf_base_s0_online
    ;;
  3)
    # best≈0.925 (collapsed); no probes yet
    KIND=crl
    CKPT_DIR=logs/ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m/ppo_builderbench_creative_7_task2_0/checkpoints
    OUT_DIR=figs/builderbench/frozen_crl_reward_probe/c7t2_crl_catwp_ms_ea_s0_online_all
    TAG_PREFIX=c7t2_crl_catwp_ms_ea_s0_online
    ;;
  4)
    # best≈0.976 last≈0.949 (sustained!); no probes yet
    KIND=nf
    CKPT_DIR=logs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m/ppo_builderbench_creative_7_task2_0/checkpoints
    OUT_DIR=figs/builderbench/nf_logp_reward_probe/c7t2_nf_catwp_ms_ea_s0_online_all
    TAG_PREFIX=c7t2_nf_catwp_ms_ea_s0_online
    ;;
  *)
    echo "bad array id ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

mkdir -p slurm "${OUT_DIR}"
echo "c7t2 successful-run reward videos kind=${KIND} → ${OUT_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"
nvidia-smi || true
ls "${CKPT_DIR}"/ckpt_iter_*.pkl | wc -l

if [ "${KIND}" = crl ]; then
  python -u scripts/render_frozen_crl_traj_reward_video.py \
    --checkpoint_dir="${CKPT_DIR}" \
    --env=builderbench_creative_7_task2 \
    --out_dir="${OUT_DIR}" \
    --tag_prefix="${TAG_PREFIX}" \
    --allow_no_success \
    --max_tries=1 \
    --seed=0 \
    --fps=8 \
    --skip_existing
else
  python -u scripts/render_nf_traj_reward_video.py \
    --checkpoint_dir="${CKPT_DIR}" \
    --env=builderbench_creative_7_task2 \
    --out_dir="${OUT_DIR}" \
    --tag_prefix="${TAG_PREFIX}" \
    --allow_no_success \
    --max_tries=1 \
    --seed=0 \
    --fps=8 \
    --skip_existing
fi

ls -lh "${OUT_DIR}"/*.mp4 2>&1 | tail -40 || true
pkill -P $$ 2>/dev/null || true
