#!/bin/bash
#SBATCH --job-name=akt_sac_push
#SBATCH --output=slurm/sac_allegro_kuka_throw_e1024_cpc_tablepush_norand_ep300_50m_2h_%A_%a.log
#SBATCH --error=slurm/sac_allegro_kuka_throw_e1024_cpc_tablepush_norand_ep300_50m_2h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# SAC contrastive_cpc smoke: nudge the cube 10 cm toward the robot on the desk
# (just outside the 7.5 cm success ball). Bucket parked off the table. Frozen init.
# Bucket is parked at the throw pose (not on the table). Frozen init.
#   sbatch jobs/job_sac_allegro_kuka_throw_e1024_cpc_tablepush_norand_ep300_50m_2h.slurm

unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

ENV=${ENV:-allegro_kuka_throw}
ALG=${ALG:-contrastive_cpc}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-50000000}
SAC_NUM_ENVS=${SAC_NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-1000000}
EP_LEN=${EP_LEN:-300}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}
DISCOUNT_FACTOR=${DISCOUNT_FACTOR:-0.99}
MIN_STD=${MIN_STD:-1e-6}

LOG_DIR="${LOG_DIR:-logs/sac_allegro_kuka_throw_e1024_cpc_tablepush_norand_ep300_50m_2h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} alg=${ALG} seed=${SEED} num_steps=${NUM_STEPS}"
echo "sac_num_envs=${SAC_NUM_ENVS} episode_length=${EP_LEN}"
echo "isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "table_push=true  object_goal=(0.0, 0.10, 0.555)  ~10cm nudge  bucket parked"
echo "isaacgym_randomize_init=false (frozen spawn at table center)"
echo "actor_min_std=${MIN_STD}  eval=off  checkpoint_interval=20  walltime=2h"

nvidia-smi || true
python -u -c "
import isaacgym  # must precede torch
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import sgcrl_jax_acme_compat, jax
print('JAX backend:', jax.default_backend(), 'devices:', jax.devices())
"

python -u sac_contrastive.py \
  --env="${ENV}" \
  --alg="${ALG}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --sac_num_envs="${SAC_NUM_ENVS}" \
  --isaacgym_episode_length="${EP_LEN}" \
  --isaacgym_pipeline="${ISAACGYM_PIPELINE}" \
  --noisaacgym_randomize_init \
  --noisaacgym_palm_goal \
  --isaacgym_table_push=true \
  --actor_min_std="${MIN_STD}" \
  --discount="${DISCOUNT_FACTOR}" \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --checkpoint_replay_max="${MAX_REPLAY_SIZE}" \
  --checkpoint_interval=20 \
  --num_sgd_steps_per_step=1 \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
