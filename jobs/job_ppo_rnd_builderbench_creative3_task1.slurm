#!/bin/bash
#SBATCH --job-name=ppo_rnd_bb_c3t1
#SBATCH --output=slurm/ppo_rnd_builderbench_creative3_task1_%A_%a.log
#SBATCH --error=slurm/ppo_rnd_builderbench_creative3_task1_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=seas
#SBATCH --array=0-1

# Two fresh PPO+RND seeds for three-cube Task 1, each trained for 200M steps.
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

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/mb6458-tmp/envs/builderbench/bin/activate

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export PYTHONPATH="/n/fs/mislresearch/builderbench${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p slurm

echo "SLURM_JOB_ID=${SLURM_JOB_ID} seed=${SLURM_ARRAY_TASK_ID}"
echo "env_id=creative-3-task1 reward=hard_success_only mjx_impl=${BUILDERBENCH_MJX_IMPL}"
nvidia-smi
python -u -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u baseline-agents/ppo-rnd.py \
  --env-id creative-3-task1 \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --num-timesteps 200000000
