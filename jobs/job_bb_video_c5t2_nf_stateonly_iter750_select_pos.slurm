#!/bin/bash
#SBATCH --job-name=c5t2_nf_s750p
#SBATCH --output=slurm/bb_vid_c5t2_nf_stateonly_iter750_select_pos_%j.log
#SBATCH --error=slurm/bb_vid_c5t2_nf_stateonly_iter750_select_pos_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=seas

# Single deterministic-policy rollout + select + pos-delta strips.
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

CKPT=logs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_norand_evalvid_catselect_stateonly/ppo_builderbench_creative_5_task2_3/checkpoints/ckpt_iter_0000750.pkl
OUT_DIR=figs/builderbench/nf_logp_reward_probe/c5t2_nf_stateonly_s3_online_all
mkdir -p slurm "${OUT_DIR}"

# overwrite prior multi-try render
rm -f "${OUT_DIR}/c5t2_nf_stateonly_s3_online_iter_0000750_select_pos.mp4" \
      "${OUT_DIR}/c5t2_nf_stateonly_s3_online_iter_0000750_select_pos_still.png" \
      "${OUT_DIR}/c5t2_nf_stateonly_s3_online_iter_0000750_select_pos.csv"

echo "Single det-policy rollout iter750 + select + pos-delta → ${OUT_DIR}"
nvidia-smi || true

python -u scripts/render_nf_traj_reward_video.py \
  --checkpoint="${CKPT}" \
  --env=builderbench_creative_5_task2 \
  --out_dir="${OUT_DIR}" \
  --tag=c5t2_nf_stateonly_s3_online_iter_0000750_select_pos \
  --show_select \
  --show_pos_delta \
  --allow_no_success \
  --max_tries=1 \
  --seed=0 \
  --fps=8

ls -lh "${OUT_DIR}"/*iter_0000750_select_pos* 2>&1 || true
pkill -P $$ 2>/dev/null || true
