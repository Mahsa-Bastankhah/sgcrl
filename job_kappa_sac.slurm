#!/bin/bash
#SBATCH --job-name=kappa_sac
#SBATCH --output=slurm/kappa_sac_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-2          # 4 seeds; change range to run more

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load cudatoolkit/12.6

# Make CUDA libraries visible to JAX/XLA.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa          # headless rendering for MuJoCo (if needed)
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # set in your shell or ~/.bashrc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR=/n/fs/mislresearch/sgcrl
cd "$REPO_DIR"
source .venv/bin/activate

# Seed = array task ID (0, 1, 2, 3 …).
SEED=${SLURM_ARRAY_TASK_ID}

# ---------------------------------------------------------------------------
# Config — edit these to taste
# ---------------------------------------------------------------------------
ENV=point_FourRooms
ALG=kappa_sac
NUM_STEPS=8000000
LOG_DIR=logs/kappa_sac/
WANDB_PROJECT=sgcrl
WANDB_ENTITY=""          # leave empty to use your default wandb entity
WANDB_GROUP="${ALG}_${ENV}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV}  alg=${ALG}  seed=${SEED}"
nvidia-smi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python -u lp_contrastive.py \
    --lp_launch_type=local_mt \
    --env="${ENV}" \
    --alg="${ALG}" \
    --seed="${SEED}" \
    --num_steps="${NUM_STEPS}" \
    --log_dir_path="${LOG_DIR}" \
    --use_wandb=True \
    --wandb_project="${WANDB_PROJECT}" \
    --wandb_entity="${WANDB_ENTITY}" \
    --wandb_group="${WANDB_GROUP}"
