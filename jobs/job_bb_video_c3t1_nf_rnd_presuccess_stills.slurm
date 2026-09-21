#!/bin/bash
#SBATCH --job-name=c3t1_gdist_vid
#SBATCH --output=slurm/bb_vid_c3t1_nf_rnd_presuccess_stills_%j.log
#SBATCH --error=slurm/bb_vid_c3t1_nf_rnd_presuccess_stills_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas

# Last-frame stills for the c3t1 NF vs RND goal-distance figure:
# NF probe iter 110 (~5.6M, just before first success) and RND params_39 (~156M).
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
export SGCRL_ROOT=/n/fs/mislresearch/sgcrl

NF_CKPT=logs/ppo_builderbench_creative3_task1_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep50_ckpt10_iter200_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_warp/ppo_builderbench_creative_3_task1_0/checkpoints/ckpt_iter_0000110.pkl
RND_CKPT_DIR=logs/final_baselines/rnd_bb_c3t1/checkpoints/creative-3-task1__1__ppo-rnd__1788316941
OUT_DIR=videos/builderbench/c3t1_goal_dist_stills

mkdir -p slurm "${OUT_DIR}" "${MPLCONFIGDIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  c3t1 NF/RND pre-success stills"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
nvidia-smi -L || true
python -u -c "import warp, jax, os; print('warp', warp.__version__); print('MJX_IMPL', os.environ.get('BUILDERBENCH_MJX_IMPL')); print('JAX backend:', jax.default_backend()); print('devices:', jax.devices())"

PY_EXIT=0
echo "[vid] NF iter 110 stochastic, no mocap targets"
python -u scripts/ppo_builderbench_rollout_video.py \
  --checkpoint="${NF_CKPT}" \
  --env=builderbench_creative_3_task1 \
  --output="${OUT_DIR}/" \
  --run_tag=nf_c3t1_presuccess \
  --seed=0 \
  --stochastic \
  --hide_targets || PY_EXIT=$?

echo "[vid] RND params_39 stochastic, no mocap targets"
python -u scripts/ppo_rnd_builderbench_rollout_video.py \
  --checkpoint_dir="${RND_CKPT_DIR}" \
  --env_id=creative-3-task1 \
  --output="${OUT_DIR}" \
  --seed=0 \
  --stochastic \
  --num_episodes=1 \
  --ckpt_glob=params_39.pkl \
  --use_pd \
  --pd_duration=5 \
  --use-residual-mlp \
  --categorical-select \
  --hide_targets \
  --fps=10 || PY_EXIT=$?

echo "==== outputs ===="
ls -lh "${OUT_DIR}"/*.mp4 2>&1 || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
