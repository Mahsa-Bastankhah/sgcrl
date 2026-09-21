#!/bin/bash
#SBATCH --job-name=akt_rnd_near10_045m060_inbkt_e70_200m
#SBATCH --output=slurm/ppo_rnd_allegro_kuka_throw_e1024_near10_045m060_inbkt_ep70_200m_seed0_4h_%j.log
#SBATCH --error=slurm/ppo_rnd_allegro_kuka_throw_e1024_near10_045m060_inbkt_ep70_200m_seed0_4h_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas

# Auto-spawned 200M RND twin of an NF run whose eval stayed >0.2.
#   sbatch jobs/job_ppo_rnd_allegro_kuka_throw_e1024_near10_045m060_inbkt_ep70_200m_seed0_4h.slurm
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
NUM_STEPS=200000000
NUM_ENVS=1024
EP_LEN=70
NUM_EVAL_STEPS=279
ISAACGYM_PIPELINE=gpu

LOG_DIR="logs/ppo_rnd_allegro_kuka_throw_e1024_near10_045m060_inbkt_ep70_200m_seed0_4h/"
mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  host=$(hostname)"
echo "env_id=${ENV_ID} seed=${SEED} num_timesteps=${NUM_STEPS}"
echo "auto RND twin  T=70  target=0.45,-0.6,0.4  success=in_bucket"
echo "ppo_eval_interval~10 via num_eval_steps=${NUM_EVAL_STEPS}"
echo "LOG_DIR=${LOG_DIR}"
"${ISAAC_PY}" -c "import sys; print('sys.executable', sys.executable)"
nvidia-smi || true
"${ISAAC_PY}" -u -c "
import isaacgym
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
  --wandb-dir="${LOG_DIR}" \
  --no-isaacgym-table-push \
  --no-isaacgym-randomize-init \
  --no-isaacgym-randomize-object-xyz \
  --no-isaacgym-randomize-object-shape \
  --no-isaacgym-palm-goal \
  --no-isaacgym-palm-and-object-success \
  --no-isaacgym-large-table \
  --isaacgym-table-spawn \
  --isaacgym-table-spawn-behind \
  --no-isaacgym-table-spawn-in-hand \
  --no-isaacgym-table-spawn-in-hand-keep-arm \
  --no-isaacgym-hide-table \
  --isaacgym-table-push-xyz=['0.20', '-0.15', '0.555'] \
  --isaacgym-episode-length=70 \
  --isaacgym-pipeline=gpu \
  --isaacgym-palm-goal-xyz=['0.17', '0.08', '0.57'] \
  --isaacgym-throw-success=in_bucket \
  --isaacgym-table-spawn-object-xy=0.2,0.08 \
  --isaacgym-table-spawn-behind-dy=0.1 \
  --isaacgym-table-spawn-behind-above=0.1 \
  --isaacgym-table-spawn-correlated-xy=0.03 \
  --isaacgym-table-spawn-finger-curl-scale=0.5 \
  --isaacgym-table-spawn-finger-noise=0.0 \
  --isaacgym-table-spawn-arm-noise=0.0 \
  --isaacgym-table-spawn-in-hand-offset=0.042 \
  --isaacgym-table-spawn-in-hand-obj-noise=0.012 \
  --isaacgym-table-spawn-in-hand-wrist-offset=-3.141592653589793 \
  --isaacgym-table-spawn-in-hand-wrist-noise=0.1 \
  --isaacgym-fixed-target-xyz=0.45,-0.6,0.4
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
