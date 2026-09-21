#!/bin/bash
#SBATCH --job-name=its_offnf_rand8m_s0
#SBATCH --output=slurm/allegro_its_offline_nf_random_8m_compactsmall_sweep_seed0_2h_%j.log
#SBATCH --error=slurm/allegro_its_offline_nf_random_8m_compactsmall_sweep_seed0_2h_%j.log
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
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

DATA_DIR="results/allegro_kuka_throw/its_offline_nf_random/dataset_random_seed0_8m"
OUT_DIR="figs/allegro_kuka_throw/its_offline_nf_random/random_seed0_compactsmall_sweep"
mkdir -p slurm "${DATA_DIR}" "${OUT_DIR}"

echo "experiment=Allegro ITS fixed-offline random-policy NF learnability"
echo "env=index_thumb_straight trim_sa=true init=full_range coordinate=fully_scaled"
echo "logical_ep=150 effective_sim_horizon=151 bootstrap_ep=153 E=1024 seed=0 collection=iid_Uniform[-1,1]^8"
echo "dims=state16[q_norm8,qd_scaled8] action8 goal8[future_q_norm]"
echo "dataset=requested_8m complete_episodes; shared_global_episode_split=90/10"
echo "scales=500k,2m,8m updates_per_model=5000 eval_every=250 batch=256"
echo "nf=p(g|s,a) compactsmall sa3x192 rank64 blocks6 width192 goal_enc0"
echo "gamma=.99 sa_lr=3e-4 flow_lr=1e-4 wd=1e-6 clip=1 goal_noise=.05"
echo "goal_std_min=.02 mix_task_goal_stats=true frac=.5"
echo "binary_eval=heldout categorical_eval=heldout_K32 chance=0.03125 fixed_fixture=true"
echo "DATA_DIR=${DATA_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "python=$(python --version 2>&1)"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "CUDA library versions and JAX readiness are printed by the Torch-first trainer"
nvidia-smi || true

validate_dataset() {
  python -u - "${DATA_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

data_dir = Path(sys.argv[1])
metadata_path = data_dir / "metadata.json"
observations_path = data_dir / "observations.npy"
actions_path = data_dir / "actions.npy"

with metadata_path.open() as f:
    metadata = json.load(f)

expected = {
    "format_version": 1,
    "dtype": "float32",
    "logical_episode_length": 150,
    "requested_transitions": 8_000_000,
    "actual_transitions": 7_999_950,
    "num_episodes": 53_333,
    "observation_shape": [53_333, 151, 16],
    "action_shape": [53_333, 150, 8],
    "observations_file": "observations.npy",
    "actions_file": "actions.npy",
}
for key, expected_value in expected.items():
    actual_value = metadata.get(key)
    if actual_value != expected_value:
        raise ValueError(
            f"{metadata_path}: {key}={actual_value!r}, expected {expected_value!r}"
        )

observations = np.load(observations_path, mmap_mode="r")
actions = np.load(actions_path, mmap_mode="r")
if list(observations.shape) != metadata["observation_shape"]:
    raise ValueError(
        f"{observations_path}: shape {observations.shape}, "
        f"metadata says {metadata['observation_shape']}"
    )
if list(actions.shape) != metadata["action_shape"]:
    raise ValueError(
        f"{actions_path}: shape {actions.shape}, metadata says {metadata['action_shape']}"
    )
if observations.dtype != np.dtype(metadata["dtype"]):
    raise ValueError(
        f"{observations_path}: dtype {observations.dtype}, "
        f"metadata says {metadata['dtype']}"
    )
if actions.dtype != np.dtype(metadata["dtype"]):
    raise ValueError(
        f"{actions_path}: dtype {actions.dtype}, metadata says {metadata['dtype']}"
    )
if actions.shape[0] * actions.shape[1] != metadata["actual_transitions"]:
    raise ValueError("action shape does not match metadata actual_transitions")

chunk_size = 1024
for name, array in (("observations", observations), ("actions", actions)):
    for start in range(0, array.shape[0], chunk_size):
        if not np.isfinite(array[start : start + chunk_size]).all():
            raise ValueError(f"{name} contains non-finite values near episode {start}")

print(
    "dataset validation passed: "
    f"observations={observations.shape} actions={actions.shape} "
    f"transitions={metadata['actual_transitions']}"
)
PY
}

dataset_files=(
  "${DATA_DIR}/metadata.json"
  "${DATA_DIR}/observations.npy"
  "${DATA_DIR}/actions.npy"
)
present_files=0
for dataset_file in "${dataset_files[@]}"; do
  if [[ -f "${dataset_file}" ]]; then
    ((present_files += 1))
  fi
done

if (( present_files == ${#dataset_files[@]} )); then
  echo "dataset reuse: complete dataset found; validating before skipping collection"
  validate_dataset
elif (( present_files == 0 )); then
  echo "dataset collection: no dataset files found"
  python -u scripts/collect_allegro_random_transitions.py \
    --env=allegro_kuka_throw \
    --seed=0 \
    --ppo_num_envs=1024 \
    --isaacgym_episode_length=153 \
    --isaacgym_pipeline=gpu \
    --isaacgym_randomize_init=true \
    --isaacgym_randomize_object_shape=false \
    --isaacgym_control_sanity_mode=index_thumb_straight \
    --isaacgym_control_sanity_trim_sa=true \
    --isaacgym_control_sanity_trim_init_mode=full_range \
    --isaacgym_coordinate_mode=fully_scaled \
    --isaacgym_large_table=true \
    --logical_episode_length=150 \
    --target_transitions=8000000 \
    --output_dir="${DATA_DIR}"
  echo "dataset collection finished; validating output"
  validate_dataset
else
  echo "ERROR: partial dataset in ${DATA_DIR}; refusing collection or training" >&2
  for dataset_file in "${dataset_files[@]}"; do
    if [[ -f "${dataset_file}" ]]; then
      echo "  present: ${dataset_file}" >&2
    else
      echo "  missing: ${dataset_file}" >&2
    fi
  done
  exit 1
fi

python -u scripts/train_allegro_nf_offline.py \
  --dataset_dir="${DATA_DIR}" \
  --output_dir="${OUT_DIR}" \
  --seed=0 \
  --dataset_sizes=500000,2000000,8000000 \
  --updates=5000 \
  --eval_every=250 \
  --batch_size=256 \
  --binary_samples=2048 \
  --categorical_samples=512 \
  --categorical_k=32 \
  --discount=0.99
