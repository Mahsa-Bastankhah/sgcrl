#!/bin/bash
#SBATCH --job-name=c5t2_nf_fwdvid
#SBATCH --output=slurm/bb_video_c5t2_nf_flow_forward_probe_%j.log
#SBATCH --error=slurm/bb_video_c5t2_nf_flow_forward_probe_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas

# Four flow-probe rollout videos:
# pre-success checkpoints 600/1050; post-success checkpoints 1500/3905.
# Live strips compare task vs NF-training-positive mean ||z|| and log|det|.
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

CKPT_DIR=logs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_norand_evalvid_catselect_extrew1/ppo_builderbench_creative_5_task2_1/checkpoints
OUT_DIR=figs/builderbench/nf_flow_forward/c5t2_nf_compact_extrew1_s1_videos

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

python -u scripts/render_nf_flow_forward_video.py \
  --checkpoint_dir="${CKPT_DIR}" \
  --iters=600,1050,1500,3905 \
  --env=builderbench_creative_5_task2 \
  --out_dir="${OUT_DIR}" \
  --tag_prefix=c5t2_nf_compact_extrew1_s1 \
  --n_pos=8 \
  --seed=0 \
  --fps=8 \
  --skip_existing
PY_EXIT=$?

echo "=== flow-probe videos finished (exit=${PY_EXIT}) ==="
ls -lh "${OUT_DIR}"/*.{mp4,png,csv} 2>/dev/null || true
pkill -P $$ 2>/dev/null || true
exit ${PY_EXIT}
