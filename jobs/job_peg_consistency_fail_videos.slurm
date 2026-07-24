#!/bin/bash
#SBATCH --job-name=peg_fail_vid
#SBATCH --output=slurm/peg_consistency_fail_vid_%A_%a.log
#SBATCH --error=slurm/peg_consistency_fail_vid_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:45:00
#SBATCH --account=seas
#SBATCH --array=0-2

# Render a few evenly spaced checkpoints for the *unsuccessful*
# ppo_peg_tau0p85_40m_consistency runs (seeds 42, 101, 202; seed 17
# learned and is skipped).  Six ckpts from early → late training plus
# latest.pkl when present.
#
#   sbatch jobs/job_peg_consistency_fail_videos.slurm
#
# Outputs:
#   videos/sawyer_peg/ppo_peg_tau0p85_40m_consistency_fail/seed<SEED>/

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
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || true

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export TMPDIR=/n/fs/mislresearch/mb6458-tmp

SEEDS=(42 101 202)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
ENV=sawyer_peg
LOG_SUBDIR=ppo_peg_tau0p85_40m_consistency
N_PICK=${N_PICK:-6}
RUN_DIR="logs/${LOG_SUBDIR}/ppo_${ENV}_${SEED}"
CKPT_SRC="${RUN_DIR}/checkpoints"
OUT_DIR="videos/sawyer_peg/${LOG_SUBDIR}_fail/seed${SEED}"
STAGE_DIR="/tmp/peg_fail_vid_${SLURM_JOB_ID}_${SEED}"

mkdir -p slurm "${OUT_DIR}" "${STAGE_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  array=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} n_pick=${N_PICK}"
echo "ckpt_src=${CKPT_SRC}"
echo "out_dir=${OUT_DIR}"

if [ ! -d "${CKPT_SRC}" ]; then
  echo "ERROR: missing checkpoints dir: ${CKPT_SRC}"
  exit 2
fi

# Symlink N_PICK evenly spaced ckpt_iter_*.pkl (+ latest.pkl) into a
# staging dir so ppo_rollout_video renders only those, not all ~170.
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
latest = os.path.join(src, "latest.pkl")
if os.path.isfile(latest):
    picks.append(latest)
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
    --seed="${SEED}" \
    --camera=corner

rm -rf "${STAGE_DIR}"
echo "Done. Videos:"
ls -lh "${OUT_DIR}"
