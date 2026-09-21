#!/bin/bash
#SBATCH --job-name=bb_c3t1_offnf_warp_fx01
#SBATCH --output=slurm/bb_c3t1_offline_nf_random_8m_compactsmall_sweep_seed0_warp_nopermute_fixedx01_2h_%j.log
#SBATCH --error=slurm/bb_c3t1_offline_nf_random_8m_compactsmall_sweep_seed0_warp_nopermute_fixedx01_2h_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=seas

set -euo pipefail
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=warp
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

SMOKE=${SMOKE:-0}
if [[ "${SMOKE}" == "1" ]]; then
  DATA_DIR="results/builderbench/c3t1_offline_nf_random/smoke_dataset_random_seed0"
  OUT_DIR="figs/builderbench/c3t1_offline_nf_random/smoke_random_seed0_compactsmall"
  TARGET_TRANSITIONS=1000
  NUM_ENVS=16
  DATASET_SIZES=500
  TRAIN_EXTRA=(--smoke_test)
else
  DATA_DIR="results/builderbench/c3t1_offline_nf_random/dataset_random_seed0_8m"
  OUT_DIR="figs/builderbench/c3t1_offline_nf_random/random_seed0_compactsmall_sweep"
  TARGET_TRANSITIONS=8000000
  NUM_ENVS=1024
  DATASET_SIZES=500000,2000000,8000000
  TRAIN_EXTRA=()
fi
mkdir -p slurm "${DATA_DIR}" "${OUT_DIR}"

echo "experiment=BuilderBench c3t1 fixed-offline random-policy NF diagnostic"
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "env=builderbench_creative_3_task1 builderbench_id=creative-3-task1 seed=0"
echo "init=nopermute fixedx01; remaining reset randomness=standard"
echo "PD=true pd_duration=5 T=50 dims=state10 action5 goal9"
echo "collection=iid_Uniform[-1,1]^5 target_transitions=${TARGET_TRANSITIONS}"
echo "nf=p(g|s,a) compactsmall sa3x192 rank64 blocks6 width192 goal_enc0"
echo "gamma=.99 batch=256 updates=5000 eval_every=250"
echo "goal_noise=.05 goal_std_min=.02 normalization=replay-only_train_goals"
echo "split=global_90/10_by_complete_episode scales=${DATASET_SIZES}"
echo "DATA_DIR=${DATA_DIR} OUT_DIR=${OUT_DIR} SMOKE=${SMOKE}"
nvidia-smi || true
python -u -c "import os, warp, jax; print('warp', warp.__version__); print('BUILDERBENCH_MJX_IMPL', os.environ['BUILDERBENCH_MJX_IMPL']); print('JAX backend', jax.default_backend()); print('JAX devices', jax.devices())"

dataset_files=(
  "${DATA_DIR}/metadata.json"
  "${DATA_DIR}/observations.npy"
  "${DATA_DIR}/actions.npy"
)
present=0
for path in "${dataset_files[@]}"; do
  [[ -f "${path}" ]] && ((present += 1))
done
if (( present == 0 )); then
  python -u scripts/collect_builderbench_random_transitions.py \
    --env=builderbench_creative_3_task1 \
    --seed=0 \
    --num_envs="${NUM_ENVS}" \
    --logical_episode_length=50 \
    --pd_duration=5 \
    --target_transitions="${TARGET_TRANSITIONS}" \
    --builderbench_permute_start_boxes=false \
    --builderbench_fixed_start_x=0.1 \
    --output_dir="${DATA_DIR}"
elif (( present != ${#dataset_files[@]} )); then
  echo "ERROR: partial dataset in ${DATA_DIR}; refusing to overwrite" >&2
  exit 1
else
  echo "dataset reuse: all files present"
fi

python -u - "${DATA_DIR}" "${TARGET_TRANSITIONS}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
target = int(sys.argv[2])
with (root / "metadata.json").open() as handle:
    meta = json.load(handle)
obs = np.load(root / "observations.npy", mmap_mode="r")
act = np.load(root / "actions.npy", mmap_mode="r")
assert obs.shape == (target // 50, 51, 10), obs.shape
assert act.shape == (target // 50, 50, 5), act.shape
assert meta["actual_transitions"] == (target // 50) * 50
assert meta["achieved_goal_state_indices"] == list(range(9))
env = meta["environment"]
assert env["mjx_impl"] == "warp"
assert env["permute_start_boxes"] is False
assert env["fixed_start_x"] == 0.1
assert "no done in macro steps 1..49" in meta["episode_boundary_check"]
assert meta["reset_randomness_check"]["distinct_target_goal_rows"] > 1
assert np.isfinite(obs).all() and np.isfinite(act).all()
print("dataset validation PASS", obs.shape, act.shape)
PY

# The BuilderBench env is NumPy 2 + Warp. The fixed-dataset NF trainer has no
# BuilderBench dependency and uses the repository's validated Torch/cuDNN-8
# environment so Torch claims the process ABI before JAX imports.
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "trainer_conda_env=${CONDA_DEFAULT_ENV} (Torch-first JAX/cuDNN-8)"

python -u scripts/train_builderbench_nf_offline.py \
  --dataset_dir="${DATA_DIR}" \
  --output_dir="${OUT_DIR}" \
  --seed=0 \
  --dataset_sizes="${DATASET_SIZES}" \
  --updates=5000 \
  --eval_every=250 \
  --batch_size=256 \
  --binary_samples=2048 \
  --categorical_samples=512 \
  --categorical_k=32 \
  --discount=0.99 \
  "${TRAIN_EXTRA[@]}"
