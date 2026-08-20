#!/bin/bash
#SBATCH --job-name=c5t2_nfcs_e5e4
#SBATCH --output=slurm/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_catselect_ent0005to0001_%A_%a.log
#SBATCH --error=slurm/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_catselect_ent0005to0001_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=14:59:00
#SBATCH --account=seas
#SBATCH --array=0-0

# Pyramid (creative-5-task2) NF compact-small, NO external reward.
# nopermute, catselect, no s/sf perturb. Compact-small NF: sa3x192 r64 b6 w192.
# Entropy anneal: 0.0005 -> 0.0001. In-job 10-video reward/grad probes.
# Seed 0, ep=50, 200M.
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
  module avail cudatoolkit 2>&1 || true
  ls -d /usr/local/cuda* 2>&1 || true
  exit 1
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench

source /n/fs/mislresearch/sgcrl/scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV=${ENV:-builderbench_creative_5_task2}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_STEPS=${NUM_STEPS:-200000000}
PPO_NUM_ENVS=${PPO_NUM_ENVS:-1024}
MAX_REPLAY_SIZE=${MAX_REPLAY_SIZE:-10000000}
ENT_COEF=0.0005
ENT_COEF_FINAL=0.0001
MIN_STD=0.01
DISCOUNT_FACTOR=0.99
PPO_CLIP_COEF=0.2
NF_REWARD_TAU=${NF_REWARD_TAU:-0.5}
EP_LEN=${EP_LEN:-50}
PD_DURATION=${PD_DURATION:-5}
MJ_EP_LEN=$((EP_LEN * PD_DURATION))
CRL_STEPS=${CRL_STEPS:-10}

NF_REP_SIZE=${NF_REP_SIZE:-64}
NF_NUM_BLOCKS=${NF_NUM_BLOCKS:-6}
NF_COUPLING_WIDTH=${NF_COUPLING_WIDTH:-192}
NF_SA_HIDDEN=${NF_SA_HIDDEN:-192}
NF_SA_NUM_LAYERS=${NF_SA_NUM_LAYERS:-3}
NF_GRAD_CLIP=${NF_GRAD_CLIP:-1.0}
NF_NOISE_STD=${NF_NOISE_STD:-0.05}
NF_GOAL_STD_MIN=${NF_GOAL_STD_MIN:-0.02}

LOG_DIR="${LOG_DIR:-logs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_catselect_ent0005to0001/}"

mkdir -p slurm "${LOG_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "env=${ENV} seed=${SEED} num_steps=${NUM_STEPS} repr_mode=nf nf_reward_tau=${NF_REWARD_TAU}"
echo "log_dir=${LOG_DIR} num_envs=${PPO_NUM_ENVS}"
echo "max_replay_size=${MAX_REPLAY_SIZE}"
echo "builderbench_use_pd=true pd_duration=${PD_DURATION} permute_start_boxes=false goals=MID+offsets(task2 pyramid)"
echo "episode_length=${EP_LEN} (mj_episode_length=${MJ_EP_LEN}, rollout_length=${EP_LEN})"
echo "ppo_norm_obs=false"
echo "ppo_use_external_reward=false"
echo "ppo_actor_min_std=${MIN_STD}"
echo "ppo_anneal_ent_coef=true ${ENT_COEF} -> ${ENT_COEF_FINAL}"
echo "ppo_categorical_select=true (hybrid Gaussian + categorical select)"
echo "ppo_crl_steps_per_iter=${CRL_STEPS}"
echo "nf_compact_small: sa${NF_SA_NUM_LAYERS}x${NF_SA_HIDDEN} r${NF_REP_SIZE} b${NF_NUM_BLOCKS} w${NF_COUPLING_WIDTH}"
echo "nf_grad_clip=${NF_GRAD_CLIP} nf_noise_std=${NF_NOISE_STD} nf_goal_std_min=${NF_GOAL_STD_MIN}"
echo "no s/sf perturb"
echo "ppo_eval_interval=30 ppo_video_interval=0 (10 in-job reward/grad probes only)"
echo "steps_per_iter=$((PPO_NUM_ENVS * EP_LEN))"
echo "num_iterations=$((NUM_STEPS / (PPO_NUM_ENVS * EP_LEN)))"

python -u -c "
from envs.builderbench_utils import default_fixed_target_goal, num_creative_tasks
import numpy as np
print('creative-5 n_tasks=', num_creative_tasks(5))
g = default_fixed_target_goal(5, 1)
print('hard_goal (flat)=', g.tolist())
print('hard_goal (cubes)=')
print(np.asarray(g).reshape(-1, 3))
"
nvidia-smi || true
python -u -c "import sgcrl_jax_acme_compat; import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

source /n/fs/mislresearch/sgcrl/scripts/bb_injob_reward_probe.sh
export PROBE_MODE=nf
export PROBE_ENV="${ENV}"
export PROBE_SEED="${SEED}"
export PROBE_CKPT_DIR="${LOG_DIR%/}/ppo_${ENV}_${SEED}/checkpoints"
export PROBE_OUT_DIR=figs/builderbench/nf_logp_reward_probe/c5t2_nfcs_catselect_ent0005to0001_s0_online_all
export PROBE_TAG_PREFIX=c5t2_nfcs_catselect_ent0005to0001_s0_online

python -u ppo_contrastive.py \
  --env="${ENV}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs="${PPO_NUM_ENVS}" \
  --ppo_ent_coef="${ENT_COEF}" \
  --ppo_anneal_ent_coef=true \
  --ppo_ent_coef_final="${ENT_COEF_FINAL}" \
  --ppo_actor_min_std="${MIN_STD}" \
  --ppo_discount="${DISCOUNT_FACTOR}" \
  --ppo_clip_coef="${PPO_CLIP_COEF}" \
  --ppo_checkpoint_interval=100 \
  --ppo_checkpoint_replay_max="${MAX_REPLAY_SIZE}" \
  --builderbench_use_pd=true \
  --builderbench_pd_duration="${PD_DURATION}" \
  --builderbench_permute_start_boxes=false \
  --builderbench_mj_episode_length="${MJ_EP_LEN}" \
  --ppo_rollout_length="${EP_LEN}" \
  --ppo_crl_steps_per_iter="${CRL_STEPS}" \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=30 \
  --ppo_eval_episodes=5 \
  --ppo_video_interval=0 \
  --max_replay_size="${MAX_REPLAY_SIZE}" \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau="${NF_REWARD_TAU}" \
  --nf_rep_size="${NF_REP_SIZE}" \
  --nf_num_blocks="${NF_NUM_BLOCKS}" \
  --nf_coupling_width="${NF_COUPLING_WIDTH}" \
  --nf_sa_hidden="${NF_SA_HIDDEN}" \
  --nf_sa_num_layers="${NF_SA_NUM_LAYERS}" \
  --nf_grad_clip="${NF_GRAD_CLIP}" \
  --nf_noise_std="${NF_NOISE_STD}" \
  --nf_goal_std_min="${NF_GOAL_STD_MIN}" \
  --nf_goal_enc_size=0 \
  --ppo_categorical_select \
  --hidden_layer_sizes="256,256,256,256,256,256" &
TRAIN_PID=$!
export PROBE_TRAIN_PID="${TRAIN_PID}"
bb_probe_sidecar &
PROBE_PID=$!
wait "${TRAIN_PID}"
PY_EXIT=$?
bb_probe_sweep
kill "${PROBE_PID}" 2>/dev/null || true
wait "${PROBE_PID}" 2>/dev/null || true
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
