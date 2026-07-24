#!/bin/bash
#SBATCH --job-name=ppo_sm_peg_mpo
#SBATCH --output=slurm/ppo_state_match_peg_mpo_compute_40m_%A_%a.log
#SBATCH --error=slurm/ppo_state_match_peg_mpo_compute_40m_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=seas
#SBATCH --array=0-3

# PPO state-distribution matching comparison against
# job_mpo_crl_sawyer_peg_tau0p85_40m.slurm.
#
# Matched per collection iteration:
#   E=4, T=256                                  -> 1,024 env transitions
#   policy: 32 epochs * 4 minibatches           -> 128 updates
#   policy batch: 1,024 / 4                     -> 256
#   CRL: 128 updates * batch 256
#   policy and CRL sample consumption ratio     -> 32 per env transition
#
# Seeds, total env steps, CRL reward EMA, discount, replay warmup, and network
# widths also match the MPO Sawyer Peg run.
module purge

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

CUDA_LOADED=0
try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    if [ -z "${CUDA_HOME:-}" ]; then
      for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
        if [ -d "${_cand}/lib64" ]; then
          export CUDA_HOME="${_cand}"
          break
        fi
      done
    fi
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || \
  try_cuda cudatoolkit/12.5 || \
  try_cuda cudatoolkit/12.4 || true

if [ "$CUDA_LOADED" != 1 ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1
      echo "Fallback CUDA_HOME=${CUDA_HOME} (module load failed)"
      break
    fi
  done
fi

if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit and no /usr/local/cuda* found."
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEEDS=(17 42 101 202)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}

ENV=sawyer_peg
NUM_STEPS=${NUM_STEPS:-40000000}
LOG_DIR="${LOG_DIR:-logs/ppo_state_match_peg_mpo_compute_40m/}"
NUM_ENVS=4
ROLLOUT_T=256
PPO_EPOCHS=32
PPO_MINIBATCHES=4
POLICY_BATCH=$((NUM_ENVS * ROLLOUT_T / PPO_MINIBATCHES))
POLICY_UPDATES=$((PPO_EPOCHS * PPO_MINIBATCHES))
CRL_UPDATES=128
CRL_BATCH=256
MIN_REPLAY=10000
TAU=0.85
DISCOUNT=0.99
ENT_COEF=0.05
MIN_STD=0.01
PPO_CLIP_COEF=0.2

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} log_dir=${LOG_DIR}"
echo "E=${NUM_ENVS} T=${ROLLOUT_T} transitions_per_iter=$((NUM_ENVS * ROLLOUT_T))"
echo "policy_epochs=${PPO_EPOCHS} minibatches=${PPO_MINIBATCHES} policy_updates=${POLICY_UPDATES} policy_batch=${POLICY_BATCH}"
echo "crl_updates=${CRL_UPDATES} crl_batch=${CRL_BATCH}"
echo "policy_SPI=$((POLICY_UPDATES * POLICY_BATCH / (NUM_ENVS * ROLLOUT_T)))"
echo "crl_SPI=$((CRL_UPDATES * CRL_BATCH / (NUM_ENVS * ROLLOUT_T)))"
echo "uniform_sampling=false tau=${TAU} discount=${DISCOUNT} min_replay=${MIN_REPLAY}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${NUM_ENVS}" \
  --ppo_rollout_length="${ROLLOUT_T}" \
  --ppo_num_epochs="${PPO_EPOCHS}" \
  --ppo_num_minibatches="${PPO_MINIBATCHES}" \
  --ppo_crl_steps_per_iter="${CRL_UPDATES}" \
  --ppo_crl_batch_size="${CRL_BATCH}" \
  --ppo_min_replay_size="${MIN_REPLAY}" \
  --ppo_repr_mode=crl \
  --uniform_sampling=false \
  --ppo_crl_loss_direction=forward \
  --ppo_crl_repr_tau="${TAU}" \
  --discount="${DISCOUNT}" \
  --ppo_discount="${DISCOUNT}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --hidden_layer_sizes="256,256,256,256,256,256" \
  --ppo_eval_interval=10 \
  --ppo_eval_episodes=5 \
  --ppo_checkpoint_interval=200 \
  --ppo_checkpoint_keep_last=5
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
