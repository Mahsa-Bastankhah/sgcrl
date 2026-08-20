#!/bin/bash
#SBATCH --job-name=nft_peg_grdual
#SBATCH --output=slurm/ppo_peg_nf_tiny_mixtaskg_grdual_%A_%a.log
#SBATCH --error=slurm/ppo_peg_nf_tiny_mixtaskg_grdual_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-0

# Same as mixtaskg (3714667) plus NF ∇_s log p dual grad-reg:
#   L = NLL + λ·E[‖∇_s log p‖],  λ ← clip(λ + lr(mean_gnorm − c))
#   c=100, λ_init=5e-4, lam_lr=auto 100*nf_critic_lr=0.01. No (s,a) mask.
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

SEED=${SEED:-0}

ENV=sawyer_peg
NUM_STEPS=${NUM_STEPS:-40000000}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-4}
CRL_STEPS=${CRL_STEPS:-10}
TAU=0.85
ENT_COEF=0.005
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
EXT_REW_SCALE=1
CKPT_INTERVAL=800
GRAD_REG_C=100.0
GRAD_REG_COEF=5e-4

NF_REP_SIZE=${NF_REP_SIZE:-32}
NF_NUM_BLOCKS=${NF_NUM_BLOCKS:-4}
NF_COUPLING_WIDTH=${NF_COUPLING_WIDTH:-128}
NF_SA_HIDDEN=${NF_SA_HIDDEN:-128}
NF_SA_NUM_LAYERS=${NF_SA_NUM_LAYERS:-2}

MIN_STD=0.00001
LOG_DIR="${LOG_DIR:-logs/ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_rand_minstd1e5_ent0005_mixtaskg_grdual/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} randomize_init min_std=${MIN_STD} ent_coef=${ENT_COEF}"
echo "repr_mode=nf nf_reward_tau=${TAU} num_steps=${NUM_STEPS} log_dir=${LOG_DIR}"
echo "density_updates=${CRL_STEPS} extrew=${EXT_REW_SCALE} nf_normalize_goals=true nf_mix_task_goal_stats=true ckpt=${CKPT_INTERVAL}"
echo "nf_grad_reg: dual c=${GRAD_REG_C} λ_init=${GRAD_REG_COEF} lam_lr=auto 100*nf_critic_lr=0.01"
echo "nf_tiny: sa${NF_SA_NUM_LAYERS}x${NF_SA_HIDDEN} r${NF_REP_SIZE} b${NF_NUM_BLOCKS} w${NF_COUPLING_WIDTH}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau="${TAU}" \
  --nf_rep_size="${NF_REP_SIZE}" \
  --nf_num_blocks="${NF_NUM_BLOCKS}" \
  --nf_coupling_width="${NF_COUPLING_WIDTH}" \
  --nf_sa_hidden="${NF_SA_HIDDEN}" \
  --nf_sa_num_layers="${NF_SA_NUM_LAYERS}" \
  --nf_grad_clip=1.0 \
  --nf_noise_std=0.05 \
  --nf_goal_std_min=0.02 \
  --nf_normalize_goals \
  --nf_mix_task_goal_stats \
  --nf_goal_enc_size=0 \
  --ppo_nf_grad_reg \
  --ppo_crl_grad_reg_c="${GRAD_REG_C}" \
  --ppo_crl_grad_reg_coef="${GRAD_REG_COEF}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --noppo_anneal_ent_coef \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_norm_obs=false \
  --ppo_use_external_reward=true \
  --ppo_external_reward_scale="${EXT_REW_SCALE}" \
  --ppo_checkpoint_interval="${CKPT_INTERVAL}" \
  --sawyer_randomize_init \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
