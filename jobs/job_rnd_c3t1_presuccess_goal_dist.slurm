#!/bin/bash
#SBATCH --job-name=rnd_c3t1_gdist
#SBATCH --output=slurm/rnd_c3t1_presuccess_goal_dist_%j.log
#SBATCH --error=slurm/rnd_c3t1_presuccess_goal_dist_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas

# PPO+RND c3t1 seed 1: 5 stochastic rollouts per checkpoint (full 200M).
# Resumes the existing CSV (4M-156M already done) and replots with
# first-nonzero and first eval>=0.1 markers.
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

CUDA_LOADED=0
try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    if [ -z "${CUDA_HOME:-}" ]; then
      for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
        if [ -d "${_cand}/lib64" ]; then export CUDA_HOME="${_cand}"; break; fi
      done
    fi
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1; echo "Loaded CUDA: $*"; echo "CUDA_HOME=${CUDA_HOME}"; return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || try_cuda cudatoolkit/12.4 || true
if [ "$CUDA_LOADED" != 1 ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1; echo "Fallback CUDA_HOME=${CUDA_HOME}"; break
    fi
  done
fi
if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit."; exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=warp
export PYTHONPATH="/n/fs/mislresearch/builderbench${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache

mkdir -p slurm "${MPLCONFIGDIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  task=c3t1 seed=1"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "full 200M; mark first nonzero and first eval_success >= 0.1"
nvidia-smi -L || true
python -u -c "import warp, jax, os; print('warp', warp.__version__); print('MJX_IMPL', os.environ.get('BUILDERBENCH_MJX_IMPL')); print('JAX backend:', jax.default_backend()); print('devices:', jax.devices())"

python -u scripts/ppo_rnd_builderbench_goal_distance.py \
  --task_key=c3t1 \
  --seed=1 \
  --num_traj=5 \
  --success_threshold=0.1

PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
