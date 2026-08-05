#!/bin/bash
#SBATCH --job-name=c7t2_vid_e10s
#SBATCH --output=slurm/bb_vid_c7t2_extrew10_succ_%j.log
#SBATCH --error=slurm/bb_vid_c7t2_extrew10_succ_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Render creative7 task2 extrew10 ckpts nearest the rare train_success_1000
# spikes (iters 511/516/523 @ ~37M). Closest saved ckpts: 450 (~32M) and 600 (~43M).
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

ENV_NAME=builderbench_creative_7_task2
OUT_DIR=videos/builderbench/c7t2_extrew10_succ
RUN=logs/ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_nopermute_fixedx01_catselect_extrew10_ep70_300m/ppo_builderbench_creative_7_task2_2
mkdir -p slurm "${OUT_DIR}"

echo "Rendering c7t2 extrew10 success-neighborhood videos → ${OUT_DIR}"
nvidia-smi || true

PY_EXIT=0
# Deterministic: before (450) and after (600) the ~37M success window
for it in 450 600; do
  ckpt=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "$RUN" "$it")
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing ${ckpt}"
    PY_EXIT=1
    continue
  fi
  tag="c7t2_e10_succ_det"
  echo "[c7t2_vid] det iter=${it} ~$((it * 1024 * 70 / 1000000))M"
  python -u scripts/ppo_builderbench_rollout_video.py \
    --checkpoint="${ckpt}" \
    --env="${ENV_NAME}" \
    --output="${OUT_DIR}/" \
    --run_tag="${tag}" \
    --seed=0 \
    --skip_existing || PY_EXIT=$?
done

# Stochastic rollouts at both neighborhood ckpts (success was ~0.1%)
for it in 450 600; do
  ckpt=$(printf '%s/checkpoints/ckpt_iter_%07d.pkl' "$RUN" "$it")
  [[ -f "${ckpt}" ]] || continue
  for seed in 0 1 2 3 4; do
    tag="c7t2_e10_succ_stoch_s${seed}"
    echo "[c7t2_vid] stoch iter=${it} seed=${seed}"
    python -u scripts/ppo_builderbench_rollout_video.py \
      --checkpoint="${ckpt}" \
      --env="${ENV_NAME}" \
      --output="${OUT_DIR}/" \
      --run_tag="${tag}" \
      --seed="${seed}" \
      --stochastic \
      --skip_existing || PY_EXIT=$?
  done
done

ls -lh "${OUT_DIR}"/*.mp4 2>&1 | tail -40 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
