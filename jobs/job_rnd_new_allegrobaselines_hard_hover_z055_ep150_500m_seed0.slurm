#!/bin/bash
#SBATCH --job-name=akt_rnd_hardhover_z055_ep150_500m
#SBATCH --output=slurm/newbl_rnd_hard_hover_z055_ep150_500m_%j.log
#SBATCH --error=slurm/newbl_rnd_hard_hover_z055_ep150_500m_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --account=seas

# PPO+RND twin of working hard-hover NF. T=150 z=0.55 500M, 10 in-train videos.
#   sbatch jobs/job_rnd_new_allegrobaselines_hard_hover_z055_ep150_500m_seed0.slurm
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
CONDA_PREFIX="/n/fs/mislresearch/miniconda3/envs/sgcrl_isaacgym"
ISAAC_PY="${CONDA_PREFIX}/bin/python"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/n/fs/mislresearch/isaacgym/python:/n/fs/mislresearch/IsaacGymEnvs:/n/fs/mislresearch/mb6458-tmp/ppo-rnd-flax-deps:/n/fs/mislresearch/sgcrl:/n/fs/mislresearch/builderbench${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SEED=0
ENV_ID=allegro_kuka_throw
NUM_STEPS=500000000
NUM_ENVS=1024
EP_LEN=150
NUM_EVAL_STEPS=326
ISAACGYM_PIPELINE=gpu
LOG_DIR="logs/new_allegrobaselines/rnd_hard_hover_near10_z055_ep150_500m_seed0_6h/"
mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  host=$(hostname)"
echo "RND seed=${SEED} E=${NUM_ENVS} ep=${EP_LEN} steps=${NUM_STEPS} goal_z=0.55"
echo "init: hover near10  bucket=(0.45,-0.60)  hard 0.73m"
echo "success=in-bucket  num_eval_steps=${NUM_EVAL_STEPS}  num_train_videos=10"
echo "LOG_DIR=${LOG_DIR}"
nvidia-smi || true
"${ISAAC_PY}" -u -c "
import isaacgym
import torch, flax, jax
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('flax', flax.__version__, 'jax', jax.default_backend(), jax.devices())
" || { echo 'ERROR: import failed'; exit 1; }

"${ISAAC_PY}" -u baseline-agents/ppo-rnd.py \
  --env-id="${ENV_ID}" \
  --seed="${SEED}" \
  --num-timesteps="${NUM_STEPS}" \
  --num-envs="${NUM_ENVS}" \
  --rollout-length="${EP_LEN}" \
  --num-eval-steps="${NUM_EVAL_STEPS}" \
  --num-train-videos=10 \
  --isaacgym-episode-length="${EP_LEN}" \
  --isaacgym-pipeline="${ISAACGYM_PIPELINE}" \
  --no-isaacgym-table-push \
  --isaacgym-fixed-target-xyz=0.45,-0.60,0.40 \
  --isaacgym-goal-z=0.55 \
  --isaacgym-throw-success=in_bucket \
  --isaacgym-table-spawn \
  --isaacgym-table-spawn-object-xy=0.20,0.08 \
  --isaacgym-table-spawn-behind \
  --isaacgym-table-spawn-behind-dy=0.10 \
  --isaacgym-table-spawn-behind-above=0.10 \
  --isaacgym-table-spawn-correlated-xy=0.03 \
  --isaacgym-table-spawn-finger-curl-scale=0.50 \
  --isaacgym-table-spawn-finger-noise=0 \
  --isaacgym-table-spawn-arm-noise=0 \
  --no-isaacgym-randomize-init \
  --no-isaacgym-randomize-object-xyz \
  --no-isaacgym-randomize-object-shape \
  --wandb-dir="${LOG_DIR}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
