#!/bin/bash
#SBATCH --job-name=bb_vid_rnd_c5t2_res6
#SBATCH --output=slurm/bb_vid_rnd_c5t2_pd_res6_catselect_stoch_%j.log
#SBATCH --error=slurm/bb_vid_rnd_c5t2_pd_res6_catselect_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas

# Stochastic-policy video for PPO+RND creative-5-task2 PD + residual6 + CatSelect
# (training job 3689005). Renders the latest params_*.pkl only.
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
      echo "Fallback CUDA_HOME=${CUDA_HOME}"
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
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

export SGCRL_ROOT=/n/fs/mislresearch/sgcrl
CKPT_ROOT="${SGCRL_ROOT}/logs/ppo_rnd_builderbench_creative5_task2_pd_res6_catselect_hardsuccess_200m/checkpoints"
CKPT_DIR="$(ls -dt "${CKPT_ROOT}"/creative-5-task2__0__ppo-rnd__* 2>/dev/null | head -1)"
OUT_DIR="${SGCRL_ROOT}/videos/builderbench/ppo_rnd_builderbench_creative5_task2_pd_res6_catselect_seed0_stoch"

mkdir -p slurm "${OUT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL} use_pd=true residual=true catselect=true"
ls -la "${CKPT_DIR}" || { echo "ERROR: no checkpoint dir"; exit 2; }
LATEST="$(ls -1 "${CKPT_DIR}"/params_*.pkl 2>/dev/null | sort -V | tail -1)"
echo "latest_ckpt=${LATEST}"
[ -n "${LATEST}" ] || { echo "ERROR: no params_*.pkl yet"; exit 3; }
nvidia-smi || true
python -u -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u "${SGCRL_ROOT}/scripts/ppo_rnd_builderbench_rollout_video.py" \
  --checkpoint_dir="${CKPT_DIR}" \
  --env_id=creative-5-task2 \
  --output="${OUT_DIR}" \
  --seed=0 \
  --stochastic \
  --num_episodes=1 \
  --max_ckpts=1 \
  --skip_existing \
  --use_pd \
  --pd_duration=5 \
  --use_residual_mlp \
  --categorical_select \
  --fps=10

PY_EXIT=$?
echo "==== outputs ===="
ls -lh "${OUT_DIR}"/*.mp4 2>&1 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
