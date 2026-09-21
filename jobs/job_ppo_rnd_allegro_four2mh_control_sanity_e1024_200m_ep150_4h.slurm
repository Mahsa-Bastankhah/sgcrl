#!/bin/bash
#SBATCH --job-name=allegro_four2mh_sanity_rnd
#SBATCH --output=slurm/ppo_rnd_allegro_four2mh_control_sanity_e1024_200m_ep150_4h_%A_%a.log
#SBATCH --error=slurm/ppo_rnd_allegro_four2mh_control_sanity_e1024_200m_ep150_4h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# PPO+RND on the between four2m (RND easy@60M) and four2h (both fail):
# base+prox for index/middle/ring/thumb.

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

ENV_ID=${ENV_ID:-allegro_kuka_throw}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
NUM_ENVS=${NUM_ENVS:-1024}
EP_LEN=${EP_LEN:-150}
NUM_EVAL_STEPS=${NUM_EVAL_STEPS:-10}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}

LOG_DIR="${LOG_DIR:-logs/ppo_rnd_allegro_four2mh_control_sanity_e1024_200m_ep150_4h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env_id=${ENV_ID} seed=${SEED} num_timesteps=${NUM_STEPS}"
echo "task=four2mh_control_sanity object_free=true state_dim=46 goal_dim=8"
echo "2 joints/finger MED-HARD: idx/mid/ring/thumb=(.42,1.00)(.37,.92)(.32,.87)(.50,.82)"
echo "success max_abs_error: hard<=0.10 (env.success())"
echo "num_envs=${NUM_ENVS} episode_length=${EP_LEN} rollout_length=${EP_LEN}"
echo "num_eval_steps=${NUM_EVAL_STEPS} isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "steps_per_iter=$((NUM_ENVS * EP_LEN))"
echo "num_iterations=$((NUM_STEPS / (NUM_ENVS * EP_LEN)))"

echo "python=${ISAAC_PY} CONDA_PREFIX=${CONDA_PREFIX}"
"${ISAAC_PY}" -c "import sys; print('sys.executable', sys.executable)"
nvidia-smi || true
"${ISAAC_PY}" -u -c "
import isaacgym  # must precede torch
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import flax, jax
print('flax', flax.__version__)
print('JAX backend:', jax.default_backend(), 'devices:', jax.devices())
" || { echo 'ERROR: isaacgym/flax import failed'; exit 1; }

"${ISAAC_PY}" -u baseline-agents/ppo-rnd.py \
  --env-id="${ENV_ID}" \
  --seed="${SEED}" \
  --num-timesteps="${NUM_STEPS}" \
  --num-envs="${NUM_ENVS}" \
  --rollout-length="${EP_LEN}" \
  --num-eval-steps="${NUM_EVAL_STEPS}" \
  --isaacgym-episode-length="${EP_LEN}" \
  --isaacgym-pipeline="${ISAACGYM_PIPELINE}" \
  --isaacgym-control-sanity-mode=four2mh \
  --isaacgym-large-table \
  --no-isaacgym-randomize-init \
  --no-isaacgym-randomize-object-xyz \
  --no-isaacgym-randomize-object-shape \
  --wandb-dir="${LOG_DIR}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
