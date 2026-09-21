#!/bin/bash
#SBATCH --job-name=nfbin_binacc_ep50
#SBATCH --output=slurm/final_metaworld_runs/eval_sawyer_bin_nf_binary_acc_nogradreg_ep50_%j.log
#SBATCH --error=slurm/final_metaworld_runs/eval_sawyer_bin_nf_binary_acc_nogradreg_ep50_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas

# Latest-checkpoint NF binary rank accuracy for the T=50 no-gradreg bin run.
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export JAX_PLATFORMS=cuda

ROOT=logs/final_metaworld_runs
OUT=figs/metaworld/final_metaworld_runs/sawyer_bin_nf_binary_acc_nogradreg_ep50.jsonl
mkdir -p slurm/final_metaworld_runs "$(dirname "${OUT}")"
: > "${OUT}"

runs=(
  "${ROOT}/ppo_bin_nf_floorgrasp_safeinit_mpoinit_nogradreg_ep50_mixtaskg/ppo_sawyer_bin_0"
)

for run in "${runs[@]}"; do
  if [ ! -d "${run}/checkpoints" ]; then
    echo "SKIP missing ${run}"
    continue
  fi
  echo "EVAL ${run}"
  python -u scripts/eval_sawyer_nf_binary_accuracy.py --run_dir="${run}" --latest --episodes=20 \
    | tee -a "${OUT}"
done
echo "WROTE ${OUT}"
