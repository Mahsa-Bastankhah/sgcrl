#!/bin/bash
#SBATCH --job-name=vid_peg_s0ck
#SBATCH --output=slurm/final_metaworld_runs/vid_peg_nf_tiny_s0_ckpts_stoch_%j.log
#SBATCH --error=slurm/final_metaworld_runs/vid_peg_nf_tiny_s0_ckpts_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas

# Stochastic training-init rollouts for final Sawyer-peg NF seed 0.
# Same np.random seed 0 on every ckpt → same peg XY + hole spawn.
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

try_cuda() {
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    echo "Loaded CUDA modules: $*"
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || true

if [ -z "${CUDA_HOME:-}" ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      echo "Fallback CUDA_HOME=${CUDA_HOME}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

ENV=sawyer_peg
SEED=0
N_PICK=${N_PICK:-8}
CKPT_SRC=logs/final_metaworld_runs/ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_rand_minstd1e5_ent0005_mixtaskg/ppo_sawyer_peg_0/checkpoints
OUT_DIR=videos/final_metaworld_runs/peg_nf_tiny_rand_mixtaskg_s0_traininit_stoch
STAGE_DIR="/tmp/vid_peg_s0ck_${SLURM_JOB_ID}"

mkdir -p slurm/final_metaworld_runs "${OUT_DIR}" "${STAGE_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "env=${ENV} run_seed=0 rollout_seed=${SEED} n_pick=${N_PICK} stochastic=1"
echo "ckpt_src=${CKPT_SRC}"
echo "out_dir=${OUT_DIR}"
echo "init: MetaWorld randomize_init (peg XY + hole; hand at (0,0.6,0.2) open)"
nvidia-smi || true

if [ ! -d "${CKPT_SRC}" ]; then
  echo "ERROR: missing checkpoints dir: ${CKPT_SRC}"
  exit 2
fi

python -u - <<PY
import glob, os, re, sys
src = "${CKPT_SRC}"
stage = "${STAGE_DIR}"
n_pick = int("${N_PICK}")
ckpts = sorted(
    glob.glob(os.path.join(src, "ckpt_iter_*.pkl")),
    key=lambda p: int(re.search(r"ckpt_iter_(\d+)", p).group(1)),
)
if not ckpts:
    print(f"ERROR: no ckpt_iter_*.pkl in {src}", file=sys.stderr)
    sys.exit(2)
if len(ckpts) <= n_pick:
    picks = ckpts
else:
    idxs = [round(i * (len(ckpts) - 1) / (n_pick - 1)) for i in range(n_pick)]
    picks = [ckpts[i] for i in sorted(set(idxs))]
for p in picks:
    dst = os.path.join(stage, os.path.basename(p))
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(p), dst)
    print("pick", os.path.basename(p))
print(f"staged {len(picks)} -> {stage}")
PY

python -u scripts/ppo_rollout_video.py \
  --checkpoint="${STAGE_DIR}" \
  --env="${ENV}" \
  --output="${OUT_DIR}/" \
  --width=640 \
  --height=480 \
  --fps=30 \
  --rotate=180 \
  --camera=corner \
  --stochastic \
  --sawyer_randomize_init \
  --seed="${SEED}"

rm -rf "${STAGE_DIR}"
echo "Done. Videos:"
ls -lh "${OUT_DIR}"
echo PEG_S0_CKPT_VIDEO_JOB_DONE
