#!/bin/bash
#SBATCH --job-name=ppo_nf_push
#SBATCH --output=slurm/ppo_nf_sawyer_push_%A_%a.out
#SBATCH --error=slurm/ppo_nf_sawyer_push_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --account=seas
#SBATCH --array=0-1

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
NUM_STEPS=${NUM_STEPS:-15000000}
RETURN_NORM_WINDOW=50000
SEED_OFFSET=${SEED_OFFSET:-0}
SEED=$(( SEED_OFFSET + ${SLURM_ARRAY_TASK_ID:-0} ))

LOG_DIR="${LOG_DIR:-logs/ppo_nf_push_v6/}"
PPO_NUM_ENVS=${PPO_NUM_ENVS:-4}
CRL_STEPS=${CRL_STEPS:-128}
ENT_COEF=0.01
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
NF_REP_SIZE=${NF_REP_SIZE:-256}
NF_NUM_BLOCKS=${NF_NUM_BLOCKS:-12}
NF_COUPLING_WIDTH=${NF_COUPLING_WIDTH:-512}
NF_GRAD_CLIP=${NF_GRAD_CLIP:-1.0}
NF_NOISE_STD=${NF_NOISE_STD:-0.05}
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
CKPT_KEEP=${CKPT_KEEP:-3}

mkdir -p slurm

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=nf"
echo "log_dir=${LOG_DIR} num_envs=${PPO_NUM_ENVS} density_steps=${CRL_STEPS}"
echo "nf_rep_size=${NF_REP_SIZE} nf_num_blocks=${NF_NUM_BLOCKS} nf_coupling_width=${NF_COUPLING_WIDTH} nf_grad_clip=${NF_GRAD_CLIP} nf_noise_std=${NF_NOISE_STD}"
echo "ckpt_interval=${CKPT_INTERVAL} ckpt_keep=${CKPT_KEEP}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

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
  --ppo_return_norm_window="${RETURN_NORM_WINDOW}" \
  --ppo_checkpoint_interval="${CKPT_INTERVAL}" \
  --ppo_checkpoint_keep_last="${CKPT_KEEP}"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
