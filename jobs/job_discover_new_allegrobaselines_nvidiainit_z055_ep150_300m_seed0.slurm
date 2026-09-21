#!/bin/bash
#SBATCH --job-name=akt_disc_nvidia_z055_ep150
#SBATCH --output=slurm/newbl_discover_nvidiainit_z055_ep150_%j.log
#SBATCH --error=slurm/newbl_discover_nvidiainit_z055_ep150_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas

# DISCOVER twin of NVIDIA-init NF / RND / MPO. Same env as the main exp.
# BB DISCOVER hyps: T_replay=25000 batch=128 UTD=32/128 UCB std=0 HER future=0.7
# 32G host RAM: replay T=25000 x E=1024 stays on CPU (~13GB). GPU OOM'd at 16G.
#   sbatch jobs/job_discover_new_allegrobaselines_nvidiainit_z055_ep150_300m_seed0.slurm
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
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Replay T=25000 lives in host RAM (~13GB). JAX only holds nets + SGD.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SEED=0
EP_LEN=150
NUM_ENVS=1024
NUM_STEPS=300000000
BATCH=128
MULT=32
REPLAY_T=25000
LOG_DIR="logs/new_allegrobaselines/discover_nvidiainit_z055_ep150_300m_seed0_14h/"
mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  host=$(hostname)"
echo "DISCOVER seed=${SEED} E=${NUM_ENVS} ep=${EP_LEN} steps=${NUM_STEPS} goal_z=0.55"
echo "init: NVIDIA default pose  norand  bucket=(0.50,-0.30)"
echo "success=in-bucket  discover_eval_interval=10 discover_checkpoint_interval=50"
echo "discover: TD3+HER+UCB weights=[mean_ini=0,mean_tar=1,std_ini=0,std_tar=0]"
echo "replay_T=${REPLAY_T} batch=${BATCH} train_step_multiplier=${MULT} UTD=32/128=0.25"
echo "goal_reach_thresh=0.1 (DISCOVER sparse ball); eval is in-bucket cylinder"
echo "LOG_DIR=${LOG_DIR}"
nvidia-smi || true
"${ISAAC_PY}" -u -c "
import isaacgym
import torch, flax, jax
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('flax', flax.__version__, 'jax', jax.default_backend(), jax.devices())
" || { echo 'ERROR: import failed'; exit 1; }

"${ISAAC_PY}" -u baseline-agents/discover/train_allegro.py \
  --env=allegro_kuka_throw \
  --seed="${SEED}" \
  --num-timesteps="${NUM_STEPS}" \
  --num-envs="${NUM_ENVS}" \
  --isaacgym_episode_length="${EP_LEN}" \
  --isaacgym_pipeline=gpu \
  --noisaacgym_randomize_init \
  --noisaacgym_randomize_object_xyz \
  --noisaacgym_randomize_object_shape \
  --isaacgym_fixed_target_xyz=0.50,-0.30,0.40 \
  --isaacgym_goal_z=0.55 \
  --isaacgym_throw_success=in_bucket \
  --log-dir="${LOG_DIR}" \
  --batch-size="${BATCH}" \
  --train-step-multiplier="${MULT}" \
  --max-replay-size="${REPLAY_T}" \
  --ucb-std-ini=0 \
  --ucb-std-tar=0 \
  --eval-interval=10 \
  --checkpoint-interval=50 \
  --early-stop-eval-success=0.9 \
  --goal-reach-thresh=0.1
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
