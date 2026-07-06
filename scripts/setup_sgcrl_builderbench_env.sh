#!/usr/bin/env bash
# Create conda env for PPO+CRL on BuilderBench (no metaworld / mujoco-py).
#
# Usage (from repo root):
#   bash scripts/setup_sgcrl_builderbench_env.sh
#
# Then:
#   conda activate sgcrl_builderbench
#   export MUJOCO_GL=egl BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
#   python ppo_contrastive.py --env=builderbench_creative_1_task1 --seed=0 ...

set -euo pipefail

ENV_NAME="${ENV_NAME:-sgcrl_builderbench}"
BUILDERBENCH_ROOT="${BUILDERBENCH_ROOT:-/n/fs/mislresearch/builderbench}"
SGCRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="${TMP_ROOT:-/n/fs/mislresearch/mb6458-tmp}"

source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists — activating and upgrading packages."
else
  echo "Creating conda env '$ENV_NAME' (Python 3.11)..."
  conda create -n "$ENV_NAME" python=3.11 pip -y
fi

conda activate "$ENV_NAME"

export TMPDIR="${TMPDIR:-$TMP_ROOT}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TMP_ROOT/pip-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TMP_ROOT/mpl-cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

echo "Installing BuilderBench (mujoco 3.7 + mjx + jax 0.10)..."
pip install -U pip setuptools wheel
pip install -e "${BUILDERBENCH_ROOT}[train]"

echo "Installing sgcrl PPO deps (no metaworld / mujoco-py / dm-reverb)..."
pip install --no-cache-dir \
  'gym==0.26.2' \
  'dm-env>=1.6' \
  'dm-tree>=0.1.8' \
  'dm-haiku>=0.0.13' \
  'absl-py' \
  'matplotlib' \
  'pillow' \
  'chex' \
  'rlax' \
  'tensorflow-cpu>=2.15' \
  'tf-keras>=2.15' \
  'tensorflow-probability>=0.24' \
  'tensorboard'
pip install --no-cache-dir 'dm-acme==0.4.0' --no-deps

echo "Verifying imports..."
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export BUILDERBENCH_ROOT="$BUILDERBENCH_ROOT"
export BUILDERBENCH_MJX_IMPL="${BUILDERBENCH_MJX_IMPL:-jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

cd "$SGCRL_ROOT"
python - <<'PY'
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ['BUILDERBENCH_ROOT'] = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')

import jax
import mujoco
import mujoco.mjx  # noqa: F401
print('jax', jax.__version__, 'devices', jax.devices())
print('mujoco', mujoco.__version__)

import builderbench  # noqa: F401
from envs.builderbench_env import make_builderbench_creative_env
import numpy as np

env = make_builderbench_creative_env(seed=0, fixed_target_goal=np.array([0.27, 0.0, 0.02]))
obs = env.reset()
obs2, r, d, info = env.step(np.zeros(5, dtype=np.float32))
print('builderbench env OK:', obs.shape, 'reward', r)

import sgcrl_jax_acme_compat  # noqa: F401
import contrastive
from contrastive import ppo_learner
print('sgcrl contrastive + ppo_learner OK')
PY

cat <<EOF

================================================================================
Done. Activate with:

  conda activate ${ENV_NAME}
  export MUJOCO_GL=egl          # use egl on GPU nodes; osmesa for CPU-only
  export BUILDERBENCH_ROOT=${BUILDERBENCH_ROOT}
  export BUILDERBENCH_MJX_IMPL=jax   # or warp if warp-lang is installed
  export MPLCONFIGDIR=${MPLCONFIGDIR}  # avoids ~/.cache disk quota issues
  cd ${SGCRL_ROOT}

Example:

  python ppo_contrastive.py \\
    --env=builderbench_creative_1_task1 \\
    --seed=0 \\
    --num_steps=4000000 \\
    --log_dir_path=logs/ppo_builderbench/ \\
    --ppo_num_envs=4 \\
    --ppo_skip_first_eval=true
================================================================================
EOF
