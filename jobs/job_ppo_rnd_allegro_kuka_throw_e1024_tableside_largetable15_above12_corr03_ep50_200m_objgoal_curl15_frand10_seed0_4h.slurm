#!/bin/bash
#SBATCH --job-name=akt_rnd_ab12_c15
#SBATCH --output=slurm/ppo_rnd_allegro_kuka_throw_e1024_tableside_largetable15_above12_corr03_ep50_200m_objgoal_curl15_frand10_seed0_%j.log
#SBATCH --error=slurm/ppo_rnd_allegro_kuka_throw_e1024_tableside_largetable15_above12_corr03_ep50_200m_objgoal_curl15_frand10_seed0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas

# PPO+RND on the hover init that had NF signal (job 3859890):
# 12cm palm-down hover, curl=0.15, finger ±10%, no arm noise, large table.
# Same env as the NF curl15_frand10 run; RND knobs stay at ppo-rnd.py defaults.
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
SEED=${SEED:-0}
NUM_STEPS=${NUM_STEPS:-200000000}
NUM_ENVS=${NUM_ENVS:-1024}
EP_LEN=${EP_LEN:-50}
# 3906 iters; 390 evals ≈ every 10 iters (matches ppo_eval_interval=10)
NUM_EVAL_STEPS=${NUM_EVAL_STEPS:-390}
ISAACGYM_PIPELINE=${ISAACGYM_PIPELINE:-gpu}
TABLE_PUSH_XYZ=${TABLE_PUSH_XYZ:-0.20,-0.15,0.555}

LOG_DIR="${LOG_DIR:-logs/ppo_rnd_allegro_kuka_throw_e1024_tableside_largetable15_above12_corr03_ep50_200m_objgoal_curl15_frand10_seed0_4h/}"
mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "env_id=${ENV_ID} seed=${SEED} num_timesteps=${NUM_STEPS}"
echo "table_push object_goal=${TABLE_PUSH_XYZ}  palm_goal=OFF"
echo "init: palm-down hover dy=0 above=0.12 curl=0.15 finger_noise=0.10 correlated_xy=0.03"
echo "table=1.5x1.5  isaacgym_large_table  no arm_noise"
echo "randomize_init=true randomize_object_xyz=false randomize_object_shape=false"
echo "RND knobs=defaults (entropy_cost/int_loss/etc from ppo-rnd.py Args)"
echo "num_envs=${NUM_ENVS} episode_length=${EP_LEN} rollout_length=${EP_LEN}"
echo "num_eval_steps=${NUM_EVAL_STEPS} (~every 10 iters) isaacgym_pipeline=${ISAACGYM_PIPELINE}"
echo "steps_per_iter=$((NUM_ENVS * EP_LEN))"
echo "num_iterations=$((NUM_STEPS / (NUM_ENVS * EP_LEN)))"
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
  --isaacgym-table-push \
  --isaacgym-table-push-xyz="${TABLE_PUSH_XYZ}" \
  --isaacgym-table-spawn \
  --isaacgym-table-spawn-behind \
  --isaacgym-table-spawn-behind-dy=0.0 \
  --isaacgym-table-spawn-behind-above=0.12 \
  --isaacgym-table-spawn-correlated-xy=0.03 \
  --isaacgym-table-spawn-finger-curl-scale=0.15 \
  --isaacgym-table-spawn-finger-noise=0.10 \
  --isaacgym-large-table \
  --isaacgym-randomize-init \
  --no-isaacgym-randomize-object-xyz \
  --no-isaacgym-randomize-object-shape \
  --wandb-dir="${LOG_DIR}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
