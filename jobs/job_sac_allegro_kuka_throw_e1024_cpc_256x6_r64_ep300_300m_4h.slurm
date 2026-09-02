#!/bin/bash
#SBATCH --job-name=akt_sac_cpc
#SBATCH --output=slurm/sac_allegro_kuka_throw_e1024_cpc_256x6_r64_ep300_300m_4h_%A_%a.log
#SBATCH --error=slurm/sac_allegro_kuka_throw_e1024_cpc_256x6_r64_ep300_300m_4h_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=seas
#SBATCH --array=0-0

# Stock contrastive_cpc SAC on AllegroKukaThrow (no Launchpad).
# Single-process: Isaac Gym E=1024 + EpisodeReplay + ContrastiveLearner.
# Actor maximizes φ·ψ (not q_sac). 3-D object-xyz goal, residual 256x6.
#   sbatch jobs/job_sac_allegro_kuka_throw_e1024_cpc_256x6_r64_ep300_300m_4h.slurm

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
NUM_STEPS=${NUM_STEPS:-300000000}
SAC_NUM_ENVS=${SAC_NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-1000000}
EP_LEN=${EP_LEN:-300}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}
DISCOUNT_FACTOR=${DISCOUNT_FACTOR:-0.99}
MIN_STD=${MIN_STD:-1e-6}

LOG_DIR="${LOG_DIR:-logs/sac_allegro_kuka_throw_e1024_cpc_256x6_r64_ep300_300m_4h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} alg=${ALG} seed=${SEED} num_steps=${NUM_STEPS}"
echo "sac_num_envs=${SAC_NUM_ENVS} episode_length=${EP_LEN}"
echo "isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "actor_min_std=${MIN_STD} (stock SAC 1e-6)"
echo "init randomization=ON; goal FIXED; packing=49+3 throw (no palm)"
echo "eval=off  checkpoint_interval=100  walltime=4h"
echo "num_sgd_steps_per_step=1  sac_updates_per_env_step=auto (SPI mapping)"

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
  --isaacgym_randomize_init=true \
  --noisaacgym_palm_goal \
  --actor_min_std="${MIN_STD}" \
  --discount="${DISCOUNT_FACTOR}" \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --checkpoint_replay_max="${MAX_REPLAY_SIZE}" \
  --checkpoint_interval=100 \
  --num_sgd_steps_per_step=1 \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
