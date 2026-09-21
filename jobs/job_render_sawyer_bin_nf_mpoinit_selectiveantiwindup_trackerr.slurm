#!/bin/bash
#SBATCH --job-name=vid_selaw
#SBATCH --output=slurm/final_metaworld_runs/vid_mpoinit_selectiveantiwindup_trackerr_s%a_i%j.log
#SBATCH --error=slurm/final_metaworld_runs/vid_mpoinit_selectiveantiwindup_trackerr_s%a_i%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# One deterministic policy rollout from a randomized MPO-safe initial state.
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

: "${POLICY_SEED:?POLICY_SEED must be exported by watcher}"
: "${CKPT_ITER:?CKPT_ITER must be exported by watcher}"
: "${INIT_SEED:?INIT_SEED must be exported by watcher}"
printf -v ITER_PAD "%07d" "${CKPT_ITER}"
RUN_ROOT="logs/final_metaworld_runs/ppo_bin_nf_floorgrasp_safeinit_mpoinit_selectiveantiwindup_mixtaskg_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6"
CKPT="${RUN_ROOT}/ppo_sawyer_bin_${POLICY_SEED}/checkpoints/ckpt_iter_${ITER_PAD}.pkl"
OUT_DIR="videos/final_metaworld_runs/bin_nf_mpoinit_selectiveantiwindup_trackerr_overlay/seed${POLICY_SEED}"
OUT="${OUT_DIR}/sawyer_bin_iter_${ITER_PAD}_init${INIT_SEED}_trackerr.mp4"
mkdir -p slurm/final_metaworld_runs "${OUT_DIR}"

echo "policy_seed=${POLICY_SEED} checkpoint_iter=${CKPT_ITER} init_seed=${INIT_SEED}"
echo "deterministic=true randomized_mpo_init=true selectiveantiwindup=true"
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
  --sawyer_bin_selective_antiwindup

echo "VIDEO ${OUT}"
