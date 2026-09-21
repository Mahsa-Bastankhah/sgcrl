#!/bin/bash
#SBATCH --job-name=akt_rnd_near10_050045_ep70_200m
#SBATCH --output=slurm/ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03_nvbucket_xy050045_noshape_ep70_200m_inbucket_seed0_%j.log
#SBATCH --error=slurm/ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03_nvbucket_xy050045_noshape_ep70_200m_inbucket_seed0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --account=seas

# PPO+RND seed 0, same hover + mid bucket as NF 3884293 (0.50,-0.45). 200M only.
#   sbatch jobs/job_ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03_nvbucket_xy050045_noshape_ep70_200m_inbucket_seed0_3h.slurm
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
# 2790 iters; 279 evals ≈ every 10 iters
NUM_EVAL_STEPS=279
ISAACGYM_PIPELINE=gpu

LOG_DIR="logs/ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03_nvbucket_xy050045_noshape_ep70_200m_inbucket_seed${SEED}_3h/"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  host=$(hostname)"
echo "env_id=${ENV_ID} seed=${SEED} num_timesteps=${NUM_STEPS}"
echo "table_push=OFF  bucket=(0.50,-0.45,0.40)  palm_goal=OFF"
echo "success=in-bucket cylinder (r=0.12 h=0.198)"
echo "init: cube on table (0.20,0.08)  palm +y 10cm / +z 10cm  curl=0.5"
echo "  correlated_xy=±0.03  NO NVIDIA joint/pose rand  no shape rand"
echo "table=NVIDIA table_narrow  (no hide_table, no large_table)"
echo "RND knobs=defaults"
echo "num_envs=${NUM_ENVS} episode_length=${EP_LEN} rollout_length=${EP_LEN}"
echo "num_eval_steps=${NUM_EVAL_STEPS} (~every 10 iters)"
echo "LOG_DIR=${LOG_DIR}"

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
  --no-isaacgym-table-push \
  --isaacgym-fixed-target-xyz=0.50,-0.45,0.40 \
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
