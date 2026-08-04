#!/bin/bash
#SBATCH --job-name=ppo_rnd_bb_c3t1_pd
#SBATCH --output=slurm/ppo_rnd_builderbench_creative3_task1_pd_%A_%a.log
#SBATCH --error=slurm/ppo_rnd_builderbench_creative3_task1_pd_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# PPO+RND for creative-3-task1 with PD waypoint control (duration=5),
# matching job_ppo_rnd_builderbench_creative4_task2_pd.slurm.
# Seed 0, 200M env steps (macro-steps under PD).
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
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || true

if [ "$CUDA_LOADED" != 1 ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1
      echo "Fallback CUDA_HOME=${CUDA_HOME} (module load failed)"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/mb6458-tmp/envs/builderbench/bin/activate

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export PYTHONPATH="/n/fs/mislresearch/builderbench${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
PD_DURATION=${PD_DURATION:-5}
LOG_DIR="${LOG_DIR:-logs/ppo_rnd_builderbench_creative3_task1_pd/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} seed=${SEED}"
echo "env_id=creative-3-task1 reward=hard_success_only mjx_impl=${BUILDERBENCH_MJX_IMPL}"
echo "use_pd=true pd_duration=${PD_DURATION}"
echo "num_timesteps=${NUM_STEPS} wandb_dir=${LOG_DIR}"
nvidia-smi || true
python -u -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u baseline-agents/ppo-rnd.py \
  --env-id creative-3-task1 \
  --seed "${SEED}" \
  --num-timesteps "${NUM_STEPS}" \
  --use-pd \
  --pd-duration "${PD_DURATION}" \
  --wandb-dir "${LOG_DIR}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
