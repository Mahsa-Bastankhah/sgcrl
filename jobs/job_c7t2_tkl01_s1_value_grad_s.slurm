#!/bin/bash
#SBATCH --job-name=c7t2_vgrads
#SBATCH --output=slurm/final_runs/c7t2_tkl01_s1_value_grad_s_%j.log
#SBATCH --error=slurm/final_runs/c7t2_tkl01_s1_value_grad_s_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:45:00
#SBATCH --account=seas

# Probe ||∇_s V(s)|| on 5 stochastic rollouts per c7t2 tKL seed-1 ckpt.
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
        if [ -d "${_cand}/lib64" ]; then
          export CUDA_HOME="${_cand}"
          break
        fi
      done
    fi
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    return 0
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
      CUDA_LOADED=1
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=warp
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache

RUN_DIR=logs/final_runs/ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep70_300m_crl10_dualgradreg_c100_lamlr1e6_tkl01_warp_4h/ppo_builderbench_creative_7_task2_1

echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "RUN_DIR=${RUN_DIR}"
nvidia-smi || true
python -c "import warp; print('warp', warp.__version__)"

mkdir -p slurm/final_runs figs/builderbench/final_runs

python -u scripts/probe_value_grad_s_ckpts.py \
  --run_dir="${RUN_DIR}" \
  --env=builderbench_creative_7_task2 \
  --stride=200 \
  --num_eval_episodes=5 \
  --eval_seed=0

PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
