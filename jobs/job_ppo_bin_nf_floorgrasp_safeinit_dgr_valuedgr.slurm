#!/bin/bash
#SBATCH --job-name=nfbin_safe_dgr
#SBATCH --output=slurm/final_metaworld_runs/ppo_bin_nf_floorgrasp_safeinit_dgr_valuedgr_s%a_%A.log
#SBATCH --error=slurm/final_metaworld_runs/ppo_bin_nf_floorgrasp_safeinit_dgr_valuedgr_s%a_%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-2

# Sawyer bin PPO+NF floor-grasp recipe with opt-in bounded safe reset,
# NF dual gradient regularization, and value dual gradient regularization.
unset LD_PRELOAD
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

if [ "${CUDA_LOADED}" != 1 ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      CUDA_LOADED=1
      echo "Fallback CUDA_HOME=${CUDA_HOME}"
      break
    fi
  done
fi

if [ "${CUDA_LOADED}" != 1 ]; then
  echo "ERROR: no usable CUDA toolkit found"
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEED="${SLURM_ARRAY_TASK_ID}"
LOG_DIR="logs/final_metaworld_runs/ppo_bin_nf_floorgrasp_safeinit_mixtaskg_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6/"
mkdir -p slurm/final_metaworld_runs "${LOG_DIR}"

echo "env=sawyer_bin seed=${SEED} steps=40000000 ppo_eval_interval=10"
echo "peg recipe: nf tiny tau=.85 crl=10 ent=.005 extrew=1 normg mixtaskg"
echo "nf_grad_reg: dual primal-dual, coef=5e-4 c=100 rel=0.1 lam_lr=1e-6"
echo "value_grad_reg: dual, c=100 lam_lr=1e-6"
echo "init: SAFE bounded/rejection reset; random object XY; centered TCP +3cm; aperture=.4"
echo "goal: hand=[.12,.70,.05] grip=.4 object=[.12,.70,.02]"
echo "success: object-only, distance < .05"
echo "log_dir=${LOG_DIR}"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print(jax.devices())"

python -u ppo_contrastive.py \
  --env=sawyer_bin \
  --seed="${SEED}" \
  --num_steps=40000000 \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs=4 \
  --ppo_crl_steps_per_iter=10 \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau=0.85 \
  --nf_rep_size=32 \
  --nf_num_blocks=4 \
  --nf_coupling_width=128 \
  --nf_sa_hidden=128 \
  --nf_sa_num_layers=2 \
  --nf_grad_clip=1.0 \
  --nf_noise_std=0.05 \
  --nf_goal_std_min=0.02 \
  --nf_goal_enc_size=0 \
  --nf_normalize_goals \
  --nf_mix_task_goal_stats \
  --ppo_ent_coef=0.005 \
  --ppo_actor_min_std=0.00001 \
  --ppo_discount=0.99 \
  --ppo_clip_coef=0.2 \
  --ppo_use_external_reward=true \
  --ppo_external_reward_scale=1 \
  --ppo_eval_interval=10 \
  --sawyer_randomize_init=true \
  --sawyer_bin_safe_grasp_reset=true \
  --ppo_nf_grad_reg \
  --ppo_crl_grad_reg_coef=5e-4 \
  --ppo_crl_grad_reg_c=100.0 \
  --ppo_crl_grad_reg_rel=0.1 \
  --ppo_nf_grad_reg_lam_lr=1e-6 \
  --ppo_value_grad_reg \
  --ppo_value_grad_reg_c=100.0 \
  --ppo_value_grad_reg_lam_lr=1e-6 \
  --hidden_layer_sizes="256,256,256,256,256,256"

PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit "${PY_EXIT}"
