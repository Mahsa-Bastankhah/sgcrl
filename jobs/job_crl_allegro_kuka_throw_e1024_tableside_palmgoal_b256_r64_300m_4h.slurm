#!/bin/bash
#SBATCH --job-name=akt_crl_tside_pg
#SBATCH --output=slurm/crl_allegro_kuka_throw_e1024_tableside_palmgoal_b256_r64_300m_4h_%A_%a.log
#SBATCH --error=slurm/crl_allegro_kuka_throw_e1024_tableside_palmgoal_b256_r64_300m_4h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# CRL baseline (crl_isaacgym.py) on AllegroKukaThrow, tableside palm-goal task.
# Actor collects data conditioned on fixed task goal only (single-goal setup).
# Critic + actor gradient updates use HER future goals (standard CRL).
# 6-D goal: [palm xyz=(0.20,-0.15,0.620) | object xyz=(0.20,-0.15,0.555)].
# Object init: xyz noise ON (±0.1 m xy, ±0.02 m z); shape/joints/quat OFF.
# batch=256, rep_size=64, updates_per_env_step=3, 300M env steps.
#
#   sbatch jobs/job_crl_allegro_kuka_throw_e1024_tableside_palmgoal_b256_r64_300m_4h.slurm

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

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-300000000}
NUM_ENVS=${NUM_ENVS:-1024}
EP_LEN=${EP_LEN:-300}
BATCH_SIZE=${BATCH_SIZE:-256}
REP_SIZE=${REP_SIZE:-64}
UPDATES_PER_STEP=${UPDATES_PER_STEP:-3}
DISCOUNT=${DISCOUNT:-0.99}
ENTROPY_COST=${ENTROPY_COST:-0.1}
LOGSUMEXP_COST=${LOGSUMEXP_COST:-0.1}
MAX_REPLAY=${MAX_REPLAY:-1000000}
MIN_REPLAY=${MIN_REPLAY:-10000}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}

LOG_DIR="${LOG_DIR:-logs/crl_allegro_kuka_throw_e1024_tableside_palmgoal_b256_r64_300m_4h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "seed=${SEED} num_steps=${NUM_STEPS} num_envs=${NUM_ENVS} ep_len=${EP_LEN}"
echo "batch_size=${BATCH_SIZE} rep_size=${REP_SIZE} updates_per_env_step=${UPDATES_PER_STEP}"
echo "entropy_cost=${ENTROPY_COST} logsumexp_cost=${LOGSUMEXP_COST} discount=${DISCOUNT}"
echo "table_push=true  object_goal=(0.20,-0.15,0.555)"
echo "palm_goal=true   6-D goal [palm=(0.20,-0.15,0.620), object=(0.20,-0.15,0.555)]"
echo "isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "randomize_init=true randomize_object_shape=true"
echo "actor conditioned on task goal during data collection only"
echo "critic + actor trained with HER future goals"
echo "log_dir=${LOG_DIR}"

nvidia-smi || true
python -u -c "
import isaacgym  # must precede torch
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import jax
print('JAX backend:', jax.default_backend(), 'devices:', jax.devices())
"

python -u crl_isaacgym.py \
  --seed="${SEED}" \
  --num_envs="${NUM_ENVS}" \
  --episode_length="${EP_LEN}" \
  --pipeline="${ISAACGYM_PIPELINE}" \
  --table_push=true \
  --table_push_xyz="0.20,-0.15,0.555" \
  --palm_goal=true \
  --palm_goal_xyz="0.20,-0.15,0.620" \
  --num_timesteps="${NUM_STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --rep_size="${REP_SIZE}" \
  --updates_per_env_step="${UPDATES_PER_STEP}" \
  --discount="${DISCOUNT}" \
  --entropy_cost="${ENTROPY_COST}" \
  --logsumexp_cost="${LOGSUMEXP_COST}" \
  --max_replay_size="${MAX_REPLAY}" \
  --min_replay_size="${MIN_REPLAY}" \
  --checkpoint_interval=100 \
  --track=true \
  --wandb_mode=offline \
  --wandb_dir="${LOG_DIR}" \
  --wandb_group="crl_allegro_baseline" \
  --wandb_project_name="sgcrl"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
