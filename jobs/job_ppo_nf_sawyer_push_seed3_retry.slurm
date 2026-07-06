#!/bin/bash
#SBATCH --job-name=nf_push_s3
#SBATCH --output=slurm/ppo_nf_push_seed3_retry_%x_%A_%j.log
#SBATCH --error=slurm/ppo_nf_push_seed3_retry_%x_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --account=seas

# Retry seed-3 NF push until JAX/GPU init succeeds and training starts.
# Resubmits on early failure (e.g. CUDA ECC); stops once learner logs exist.
#
# Required exports:
#   LOG_DIR, NF_GRAD_CLIP
# Optional: NF_REP_SIZE, NF_NUM_BLOCKS, NF_COUPLING_WIDTH, RETRY_COUNT

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
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    CUDA_LOADED=1
    echo "Loaded CUDA modules: $*"
    echo "CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

try_cuda cudatoolkit/12.6 || true
if [ "$CUDA_LOADED" != 1 ]; then
  echo "ERROR: Could not load cudatoolkit."
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

ENV=sawyer_push
SEED=3
NUM_STEPS=${NUM_STEPS:-15000000}
RETURN_NORM_WINDOW=50000
LOG_DIR="${LOG_DIR:?LOG_DIR required}"
PPO_NUM_ENVS=${PPO_NUM_ENVS:-4}
CRL_STEPS=${CRL_STEPS:-128}
ENT_COEF=0.01
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
NF_REP_SIZE=${NF_REP_SIZE:-64}
NF_NUM_BLOCKS=${NF_NUM_BLOCKS:-8}
NF_COUPLING_WIDTH=${NF_COUPLING_WIDTH:-256}
NF_GRAD_CLIP=${NF_GRAD_CLIP:-1.0}
NF_NOISE_STD=${NF_NOISE_STD:-0.05}
RETRY_COUNT=${RETRY_COUNT:-0}
RETRY_TAG=${RETRY_TAG:-push_small}

LEARN_CSV="${LOG_DIR}/ppo_sawyer_push_${SEED}/logs/learner/logs.csv"
MIN_LEARNER_LINES=20

mkdir -p slurm

echo "RETRY_COUNT=${RETRY_COUNT} RETRY_TAG=${RETRY_TAG}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=nf"
echo "log_dir=${LOG_DIR} nf_grad_clip=${NF_GRAD_CLIP}"
echo "nf_rep_size=${NF_REP_SIZE} nf_num_blocks=${NF_NUM_BLOCKS} nf_coupling_width=${NF_COUPLING_WIDTH}"
nvidia-smi || true

python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"
JAX_EXIT=$?
if [ "$JAX_EXIT" -ne 0 ]; then
  PY_EXIT=$JAX_EXIT
else
  python -u ppo_contrastive.py \
    --env="${ENV}" \
    --seed="${SEED}" \
    --num_steps="${NUM_STEPS}" \
    --log_dir_path="${LOG_DIR}" \
    --ppo_repr_mode="nf" \
    --nf_rep_size="${NF_REP_SIZE}" \
    --nf_num_blocks="${NF_NUM_BLOCKS}" \
    --nf_coupling_width="${NF_COUPLING_WIDTH}" \
    --nf_grad_clip="${NF_GRAD_CLIP}" \
    --nf_noise_std="${NF_NOISE_STD}" \
    --ppo_num_envs="${PPO_NUM_ENVS}" \
    --ppo_crl_steps_per_iter="${CRL_STEPS}" \
    --ppo_ent_coef="${ENT_COEF}" \
    --ppo_actor_min_std="${MIN_STD}" \
    --ppo_discount="${DISCOUNT_FACTOR}" \
    --ppo_clip_coef="${PPO_CLIP_COEF}" \
    --hidden_layer_sizes="256,256,256,256,256,256" \
    --nf_goal_std_min=0.01 \
    --nf_mix_env_goal_stats=true \
    --ppo_return_norm_window="${RETURN_NORM_WINDOW}"
  PY_EXIT=$?
fi

pkill -P $$ 2>/dev/null || true

learn_lines=0
if [ -f "$LEARN_CSV" ]; then
  learn_lines=$(wc -l < "$LEARN_CSV")
fi

if [ "$PY_EXIT" -eq 0 ]; then
  echo "[seed3-retry] training finished successfully (learner lines=${learn_lines})"
  exit 0
fi

if [ "$learn_lines" -ge "$MIN_LEARNER_LINES" ]; then
  echo "[seed3-retry] training had started (${learn_lines} learner lines); not resubmitting"
  exit "$PY_EXIT"
fi

next_retry=$((RETRY_COUNT + 1))
echo "[seed3-retry] early failure (exit=${PY_EXIT}, learner lines=${learn_lines}); resubmitting attempt ${next_retry}"
sbatch \
  --job-name="nf_s3_${RETRY_TAG}" \
  --export=ALL,LOG_DIR="${LOG_DIR}",NF_GRAD_CLIP="${NF_GRAD_CLIP}",NF_REP_SIZE="${NF_REP_SIZE}",NF_NUM_BLOCKS="${NF_NUM_BLOCKS}",NF_COUPLING_WIDTH="${NF_COUPLING_WIDTH}",NF_NOISE_STD="${NF_NOISE_STD}",RETRY_COUNT="${next_retry}",RETRY_TAG="${RETRY_TAG}" \
  jobs/job_ppo_nf_sawyer_push_seed3_retry.slurm
exit "$PY_EXIT"
