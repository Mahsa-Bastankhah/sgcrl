#!/bin/bash
#SBATCH --job-name=ppo_imp_nft
#SBATCH --output=slurm/ppo_impossible_nf_tiny_e64_10m_%j.log
#SBATCH --error=slurm/ppo_impossible_nf_tiny_e64_10m_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=seas

# Impossible maze, NF tiny, 10M env steps, JAX-batched vec env.
# E=64 T=128 → 1220 iters (9,994,240 steps). Render 5 trajs + log p(g|s,a)
# heatmap every 50 iters (~24 PNGs).
set -euo pipefail

ROOT=/n/fs/mislresearch/sgcrl
PYTHON="${ROOT}/.venv/bin/python"
ENV=point_Impossible
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-10000000}"
LOG_DIR="${LOG_DIR:-logs/ppo_impossible_nf_tiny_e64_10m/}"
RUN_DIR="${LOG_DIR}/ppo_${ENV}_${SEED}"

module purge
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

CUDA_LOADED=0
for CUDA_MODULE in cudatoolkit/12.6 cudatoolkit/12.5 cudatoolkit/12.4; do
  if module load "${CUDA_MODULE}" 2>/dev/null; then
    CUDA_LOADED=1
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    echo "Loaded CUDA module: ${CUDA_MODULE}"
    break
  fi
done
if [[ "${CUDA_LOADED}" != "1" ]]; then
  for CANDIDATE in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [[ -d "${CANDIDATE}/lib64" ]]; then
      CUDA_LOADED=1
      export CUDA_HOME="${CANDIDATE}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      echo "Using CUDA fallback: ${CUDA_HOME}"
      break
    fi
  done
fi
if [[ "${CUDA_LOADED}" != "1" || -z "${CUDA_HOME:-}" ]]; then
  echo "ERROR: no CUDA toolkit found"
  exit 1
fi

cd "${ROOT}"
source .venv/bin/activate
unset -f python python3 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export TF_CPP_MIN_LOG_LEVEL=2

mkdir -p slurm "${LOG_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} log_dir=${LOG_DIR}"
echo "E=64 T=128 crl_steps=25 eval_interval=10 video_interval=50"
echo "nf_tiny: sa2x128 r32 b4 w128  r(s,a)  extrew=false  obs_norm=false"
echo "run_dir=${RUN_DIR}"
echo "ppo_eval_interval=10"
nvidia-smi || true
"${PYTHON}" -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

"${PYTHON}" -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_repr_mode=nf \
  --ppo_num_envs=64 \
  --ppo_rollout_length=128 \
  --ppo_crl_steps_per_iter=25 \
  --ppo_eval_interval=10 \
  --ppo_eval_episodes=5 \
  --ppo_video_interval=50 \
  --ppo_checkpoint_interval=200 \
  --ppo_checkpoint_keep_last=5 \
  --noppo_use_external_reward \
  --noppo_norm_obs \
  --hidden_layer_sizes=256,256,256,256,256,256 \
  --nf_sa_num_layers=2 \
  --nf_sa_hidden=128 \
  --nf_rep_size=32 \
  --nf_num_blocks=4 \
  --nf_coupling_width=128
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
