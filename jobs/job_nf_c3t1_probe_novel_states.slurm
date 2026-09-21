#!/bin/bash
#SBATCH --job-name=nf_c3t1_novel
#SBATCH --output=slurm/nf_c3t1_probe_novel_states_%j.log
#SBATCH --error=slurm/nf_c3t1_probe_novel_states_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --account=seas

# PPO+NF c3t1: dense probe (seed 0, ckpt every 10) then long-run seed 0
# ckpts through ~40M. Bin cube xyz at 2cm; accumulate never-seen states.
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
export JAX_PLATFORMS=cuda

mkdir -p slurm "${MPLCONFIGDIR}"

PROBE_DIR=logs/ppo_builderbench_creative3_task1_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep50_ckpt10_iter200_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_warp/ppo_builderbench_creative_3_task1_0
TAIL_DIR=logs/final_runs/ppo_builderbench_creative3_task1_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_warp_4h/ppo_builderbench_creative_3_task1_0
BIN_SIZE=${BIN_SIZE:-0.02}
PLOT_OUT="${PLOT_OUT:-paper material/paper plots/bb_nf_c3t1_novel_states}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  NF c3t1 seed=0  novel states through 40M"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "run_dir=${PROBE_DIR}"
echo "tail_run_dir=${TAIL_DIR}"
echo "bin_size=${BIN_SIZE} max_env_steps=40000000"
echo "plot_out=${PLOT_OUT}"
nvidia-smi -L || true
python -u -c "import warp, jax, os; print('warp', warp.__version__); print('MJX_IMPL', os.environ.get('BUILDERBENCH_MJX_IMPL')); print('JAX backend:', jax.default_backend()); print('devices:', jax.devices())"

python -u scripts/ppo_nf_builderbench_novel_states.py \
  --seed=0 \
  --run_dir="${PROBE_DIR}" \
  --tail_run_dir="${TAIL_DIR}" \
  --num_traj=5 \
  --success_threshold=0.1 \
  --bin_size="${BIN_SIZE}" \
  --after_success_ckpts=-1 \
  --max_env_steps=40000000 \
  --plot_output="${PLOT_OUT}"

PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
