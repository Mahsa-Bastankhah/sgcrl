#!/bin/bash
#SBATCH --job-name=nfbin_g04
#SBATCH --output=slurm/final_metaworld_runs/ppo_bin_nf_pegrecipe_floorgrasp_g04_rand_s%a_%A.log
#SBATCH --error=slurm/final_metaworld_runs/ppo_bin_nf_pegrecipe_floorgrasp_g04_rand_s%a_%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=1-2

# Sawyer bin with the successful peg PPO+NF recipe.
# New geometry: init and task goal both use aperture=0.4 and TCP=obj+[0,0,0.03].
# Random init changes object XY; gripper follows it without independent noise.
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
LOG_DIR="logs/final_metaworld_runs/ppo_bin_nf_pegrecipe_floorgrasp_g04_rand_mixtaskg/"
mkdir -p slurm/final_metaworld_runs "${LOG_DIR}"

echo "env=sawyer_bin seed=${SEED} steps=40000000 eval_interval=10"
echo "peg recipe: nf tiny tau=.85 crl=10 ent=.005 extrew=1 normg mixtaskg"
echo "init: random object XY; gripper follows at +3cm; aperture=.4; no independent gripper noise"
echo "goal: hand=[.12,.70,.05] grip=.4 object=[.12,.70,.02]"
echo "success: object-only, distance < .05"
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
  --hidden_layer_sizes="256,256,256,256,256,256"

PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit "${PY_EXIT}"
