#!/bin/bash
#SBATCH --job-name=c3t1_rewvid
#SBATCH --output=slurm/bb_vid_c3t1_crl_stateonly_reward_strip_%j.log
#SBATCH --error=slurm/bb_vid_c3t1_crl_stateonly_reward_strip_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Reward-strip videos (φ·ψ timeline + MuJoCo) for creative3 task1 CRL stateonly.
# Source log: slurm/ppo_builderbench_creative3_task1_e1024_pd_crl_tau05_catselect_stateonly_3638389_0.log
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

RUN=logs/ppo_builderbench_creative3_task1_e1024_pd_crl_tau05_catselect_stateonly/ppo_builderbench_creative_3_task1_0
OUT_DIR=figs/builderbench/frozen_crl_reward_probe
mkdir -p slurm "${OUT_DIR}"

echo "Rendering c3t1 CRL stateonly reward-strip videos"
nvidia-smi || true

PY_EXIT=0
# Mid-train near peak train_success (~1916) + late ckpt. Eval success was 0,
# so allow_no_success keeps the highest φ·ψ return traj among tries.
for it in 1800 3900; do
  ckpt=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "$RUN" "$it")
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing ${ckpt}"
    PY_EXIT=1
    continue
  fi
  tag="c3t1_crl_stateonly_ckpt${it}_phi_psi_traj"
  echo "[rewvid] iter=${it}"
  python -u scripts/render_frozen_crl_traj_reward_video.py \
    --checkpoint="${ckpt}" \
    --tag="${tag}" \
    --title="CRL state-only reward  r=φ(s)·ψ(g)  ·  c3t1 ckpt ${it}" \
    --allow_no_success \
    --max_tries=40 \
    --seed=0 \
    --fps=8 || PY_EXIT=$?
done

ls -lh "${OUT_DIR}"/c3t1_crl_stateonly_ckpt*_phi_psi_traj* 2>&1 | tail -40 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
