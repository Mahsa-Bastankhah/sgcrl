#!/bin/bash
#SBATCH --job-name=vid_trackerr
#SBATCH --output=slurm/final_metaworld_runs/vid_trackerr_%j.log
#SBATCH --error=slurm/final_metaworld_runs/vid_trackerr_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# Generic deterministic tracking-error overlay. Required exports:
#   POLICY_SEED CKPT_ITER INIT_SEED RUN_ROOT OUT_DIR
# Optional: CTRL=vanilla|antiwindup|terminate|bounded
#           SAWYER_MAX_EPISODE_STEPS (default unset / 150)
set -euo pipefail
unset LD_PRELOAD
module purge
module load cudatoolkit/12.6 2>/dev/null || module load cudatoolkit/12.5 2>/dev/null || true

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow
export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-/usr/local/cuda}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export TMPDIR=/n/fs/mislresearch/mb6458-tmp

: "${POLICY_SEED:?}"
: "${CKPT_ITER:?}"
: "${INIT_SEED:?}"
: "${RUN_ROOT:?}"
: "${OUT_DIR:?}"
CTRL="${CTRL:-vanilla}"
printf -v ITER_PAD "%07d" "${CKPT_ITER}"
CKPT="${RUN_ROOT}/ppo_sawyer_bin_${POLICY_SEED}/checkpoints/ckpt_iter_${ITER_PAD}.pkl"
OUT="${OUT_DIR}/sawyer_bin_iter_${ITER_PAD}_init${INIT_SEED}_trackerr.mp4"
mkdir -p slurm/final_metaworld_runs "${OUT_DIR}"

EXTRA=()
case "${CTRL}" in
  vanilla) ;;
  antiwindup) EXTRA+=(--sawyer_bin_selective_antiwindup) ;;
  terminate) EXTRA+=(--sawyer_bin_trackerr_terminate_20cm) ;;
  bounded) EXTRA+=(--sawyer_bin_bounded_step_mocap) ;;
  *) echo "ERROR: unknown CTRL=${CTRL}"; exit 2 ;;
esac
if [ -n "${SAWYER_MAX_EPISODE_STEPS:-}" ] && [ "${SAWYER_MAX_EPISODE_STEPS}" -gt 0 ]; then
  EXTRA+=(--sawyer_max_episode_steps="${SAWYER_MAX_EPISODE_STEPS}")
fi

echo "policy_seed=${POLICY_SEED} checkpoint_iter=${CKPT_ITER} init_seed=${INIT_SEED}"
echo "run_root=${RUN_ROOT}"
echo "ctrl=${CTRL} deterministic=true randomized_mpo_init=true"
echo "checkpoint=${CKPT}"
echo "output=${OUT}"
test -s "${CKPT}"
nvidia-smi || true
python -u scripts/render_sawyer_mocap_tracking_error_video.py \
  --checkpoint="${CKPT}" \
  --env=sawyer_bin \
  --output="${OUT}" \
  --width=640 \
  --height=480 \
  --fps=30 \
  --rotate=180 \
  --camera=corner \
  --seed="${INIT_SEED}" \
  --ymax_cm=20 \
  --sawyer_bin_safe_grasp_reset \
  --bin_randomize_gripper_init \
  "${EXTRA[@]+"${EXTRA[@]}"}"

echo "VIDEO_READY ${OUT}"
