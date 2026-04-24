#!/bin/bash
#SBATCH --job-name=kappa_g
#SBATCH --output=slurm/kappa_gamma_sweep_hard%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --account=seas
# Edit: DISCOUNT_KAPPA, SEEDS, and --array=0-(n-1) for n=len(SEEDS)
#SBATCH --array=0-4

DISCOUNT_KAPPA="${DISCOUNT_KAPPA:-0.5}"
GAMMA="${DISCOUNT_KAPPA}"
SEEDS=(30 31 32 33 34)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#SEEDS[@]}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} not in 0..$((${#SEEDS[@]} - 1))"
  exit 2
fi
SEED="${SEEDS[$TASK_ID]}"
SWEEP_TAG="gkappa_$(echo "${GAMMA}" | tr '.' 'p')"

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
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || true
[ "$CUDA_LOADED" = 1 ] || { echo "ERROR: could not load cudatoolkit"; exit 1; }

cd /n/fs/mislresearch/sgcrl
source .venv/bin/activate
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

ENV=sawyer_bin
ALG=kappa_sac
NUM_STEPS=15000000
LOG_DIR="logs/kappa_gamma_sweep_bin_hard/${SWEEP_TAG}/"

# --- lp_contrastive knobs (edit here; sweep_config.json uses the same values) ---
# Defaults: twin κ on, κ Adam LR 1e-4, κ grad clip 1.0, no hard-goal actor term, no repr_norm.
KAPPA_ACTOR_HARD_GOAL_COEF="${KAPPA_ACTOR_HARD_GOAL_COEF:-0.1}"
LEARNING_RATE_KAPPA="${LEARNING_RATE_KAPPA:-1e-4}"
KAPPA_MAX_GRAD_NORM="${KAPPA_MAX_GRAD_NORM:-1.0}"
Q_ACTOR_KL_TO_PREV_COEF="${Q_ACTOR_KL_TO_PREV_COEF:-0}"
TWIN_KAPPA="${TWIN_KAPPA:-1}"
REPR_NORM="${REPR_NORM:-0}"

LP_EXTRA=(
  --kappa_actor_hard_goal_coef="${KAPPA_ACTOR_HARD_GOAL_COEF}"
  --learning_rate_kappa="${LEARNING_RATE_KAPPA}"
  --kappa_max_grad_norm="${KAPPA_MAX_GRAD_NORM}"
  --q_actor_kl_to_prev_coef="${Q_ACTOR_KL_TO_PREV_COEF}"
)
if [ "${TWIN_KAPPA}" = "1" ]; then
  LP_EXTRA+=(--twin_kappa)
else
  LP_EXTRA+=(--notwin_kappa)
fi
if [ "${REPR_NORM}" = "1" ]; then
  LP_EXTRA+=(--repr_norm)
else
  LP_EXTRA+=(--norepr_norm)
fi

RUN_DIR="${LOG_DIR}${ALG}_${ENV}_${SEED}"
mkdir -p "${RUN_DIR}"

echo "task=${TASK_ID}  seed=${SEED}  discount_kappa=${GAMMA}  run_dir=${RUN_DIR}"
echo "kappa_actor_hard_goal_coef=${KAPPA_ACTOR_HARD_GOAL_COEF}  learning_rate_kappa=${LEARNING_RATE_KAPPA}  kappa_max_grad_norm=${KAPPA_MAX_GRAD_NORM}  q_actor_kl_to_prev_coef=${Q_ACTOR_KL_TO_PREV_COEF}  twin_kappa=${TWIN_KAPPA}  repr_norm=${REPR_NORM}"

export SWEEP_SEED="${SEED}" SWEEP_ENV="${ENV}" SWEEP_ALG="${ALG}" \
  SWEEP_NUM_STEPS="${NUM_STEPS}" SWEEP_LOG_DIR="${LOG_DIR}" \
  SWEEP_GAMMA="${GAMMA}" SWEEP_RUN_DIR="${RUN_DIR}" SWEEP_TAG="${SWEEP_TAG}" \
  SWEEP_KAPPA_ACTOR_HARD_GOAL_COEF="${KAPPA_ACTOR_HARD_GOAL_COEF}" \
  SWEEP_LEARNING_RATE_KAPPA="${LEARNING_RATE_KAPPA}" \
  SWEEP_KAPPA_MAX_GRAD_NORM="${KAPPA_MAX_GRAD_NORM}" \
  SWEEP_Q_ACTOR_KL_TO_PREV_COEF="${Q_ACTOR_KL_TO_PREV_COEF}" \
  SWEEP_TWIN_KAPPA="${TWIN_KAPPA}" SWEEP_REPR_NORM="${REPR_NORM}"

cat > "${RUN_DIR}/sweep_config.txt" <<EOF
discount_kappa=${GAMMA}
seed=${SEED}
slurm_array_task_id=${TASK_ID}
sweep_tag=${SWEEP_TAG}
kappa_actor_hard_goal_coef=${KAPPA_ACTOR_HARD_GOAL_COEF}
learning_rate_kappa=${LEARNING_RATE_KAPPA}
kappa_max_grad_norm=${KAPPA_MAX_GRAD_NORM}
q_actor_kl_to_prev_coef=${Q_ACTOR_KL_TO_PREV_COEF}
twin_kappa=${TWIN_KAPPA}
repr_norm=${REPR_NORM}
env=${ENV}  alg=${ALG}  num_steps=${NUM_STEPS}
run_dir=${RUN_DIR}
EOF

python -u -c "
import json, os, sys
sys.path.insert(0, '.')
import sgcrl_jax_acme_compat  # noqa: F401
import lp_contrastive as lp
from contrastive.config import ContrastiveConfig

def _b(s, d='0'):
    v = os.environ.get(s, d).strip().lower()
    return v in ('1', 'true', 'yes', 'on')

p = dict(
    seed=int(os.environ['SWEEP_SEED']),
    env_name=os.environ['SWEEP_ENV'],
    alg_name=os.environ['SWEEP_ALG'],
    max_number_of_steps=int(os.environ['SWEEP_NUM_STEPS']),
    log_dir=os.environ['SWEEP_LOG_DIR'],
    use_cpc=True,
    use_kappa=True,
    twin_kappa=_b('SWEEP_TWIN_KAPPA', '1'),
    reward_shaping_mode='kappa',
    discount_kappa=float(os.environ['SWEEP_GAMMA']),
    repr_norm=_b('SWEEP_REPR_NORM', '0'),
    kappa_actor_hard_goal_coef=float(os.environ['SWEEP_KAPPA_ACTOR_HARD_GOAL_COEF']),
    learning_rate_kappa=float(os.environ['SWEEP_LEARNING_RATE_KAPPA']),
    kappa_max_grad_norm=float(os.environ['SWEEP_KAPPA_MAX_GRAD_NORM']),
    q_actor_kl_to_prev_coef=float(os.environ['SWEEP_Q_ACTOR_KL_TO_PREV_COEF']),
)
hg = lp._extract_hard_goal(os.environ['SWEEP_ENV'])
if hg is not None:
    p['hard_goal'] = hg
cfg = ContrastiveConfig(**p)
d = {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else str(v))
     for k, v in cfg.__dict__.items()}
d['sweep_tag'] = os.environ['SWEEP_TAG']
d['slurm_job_id'] = os.environ.get('SLURM_JOB_ID', '')
d['slurm_array_task_id'] = os.environ.get('SLURM_ARRAY_TASK_ID', '')
d['launch_lp_extras'] = [
    '--kappa_actor_hard_goal_coef=' + os.environ['SWEEP_KAPPA_ACTOR_HARD_GOAL_COEF'],
    '--learning_rate_kappa=' + os.environ['SWEEP_LEARNING_RATE_KAPPA'],
    '--kappa_max_grad_norm=' + os.environ['SWEEP_KAPPA_MAX_GRAD_NORM'],
    '--q_actor_kl_to_prev_coef=' + os.environ['SWEEP_Q_ACTOR_KL_TO_PREV_COEF'],
    '--twin_kappa' if _b('SWEEP_TWIN_KAPPA', '1') else '--notwin_kappa',
    '--repr_norm' if _b('SWEEP_REPR_NORM', '0') else '--norepr_norm',
]
path = os.path.join(os.environ['SWEEP_RUN_DIR'], 'sweep_config.json')
with open(path, 'w') as f:
    json.dump(d, f, indent=2, default=str)
print('Wrote', path)
"

exec python -u lp_contrastive.py \
  --lp_launch_type=local_mp \
  --env="${ENV}" \
  --alg="${ALG}" \
  --seed="${SEED}" \
  --num_steps="${NUM_STEPS}" \
  --log_dir_path="${LOG_DIR}" \
  --discount_kappa="${GAMMA}" \
  "${LP_EXTRA[@]}" \
  # --hidden_layer_sizes=256,256,256,256,256,256
