#!/bin/bash
#SBATCH --job-name=rnd_c5t2_ef50
#SBATCH --output=slurm/ppo_rnd_bb_c5t2_pd_res_e1024_ep50_200m_warp_evalfix_%A_%a.log
#SBATCH --error=slurm/ppo_rnd_bb_c5t2_pd_res_e1024_ep50_200m_warp_evalfix_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --account=seas
#SBATCH --array=0-1

# PPO+RND baseline: creative-5-task2 (pyramid), PD control, residual 6x256,
# E=1024, T=ep=50 macro-steps, 200M env steps, 2 seeds.
# Fixed init: permute_start_boxes=false, fixed_start_x=0.1, fixed goal.
# Eval uses the same fixed target as train (FixedGoalWrapper).
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

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
EP_LEN=50
PD_DURATION=5
MJ_EP_LEN=$((EP_LEN * PD_DURATION))   # 250 simulator steps (c5 default is 350)
LOG_DIR="${LOG_DIR:-logs/ppo_rnd_bb_c5t2_pd_cs_res_e1024_ep50_200m_warp_evalfix/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  seed=${SEED}"
echo "env=creative-5-task2 num_steps=${NUM_STEPS} num_envs=1024"
echo "pd_duration=${PD_DURATION} ep_len=${EP_LEN} mj_ep_len=${MJ_EP_LEN}"
echo "arch=residual_6x256 permute_start_boxes=false fixed_start_x=0.1 fix_goal=true"
echo "eval uses the same fixed target_goal as train"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"

python -u -c "import warp, jax, os; print('warp', warp.__version__); print('MJX_IMPL', os.environ.get('BUILDERBENCH_MJX_IMPL')); print('JAX backend:', jax.default_backend()); print('devices:', jax.devices())"

python -u baseline-agents/ppo-rnd.py \
  --env-id creative-5-task2 \
  --seed "${SEED}" \
  --num-timesteps "${NUM_STEPS}" \
  --num-envs 1024 \
  --use-pd \
  --pd-duration "${PD_DURATION}" \
  --env-episode-length "${MJ_EP_LEN}" \
  --rollout-length "${EP_LEN}" \
  --use-residual-mlp \
  --no-permute-start-boxes \
  --fixed-start-x 0.1 \
  --fix-goal \
  --categorical-select \
  --wandb-dir "${LOG_DIR}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
