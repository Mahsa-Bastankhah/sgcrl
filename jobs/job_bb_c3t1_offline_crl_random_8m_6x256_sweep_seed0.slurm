#!/bin/bash
#SBATCH --job-name=bb_c3t1_offcrl_rand8m_s0
#SBATCH --output=slurm/bb_c3t1_offline_crl_random_8m_6x256_sweep_seed0_%j.log
#SBATCH --error=slurm/bb_c3t1_offline_crl_random_8m_6x256_sweep_seed0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=seas

set -euo pipefail
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
# Same trainer env as the successful BB NF offline job: Torch-first cuDNN-8.
# No Warp / no env step; training is npy-only.
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

DATA_DIR="results/builderbench/c3t1_offline_nf_random/dataset_random_seed0_8m"
NF_DIR="figs/builderbench/c3t1_offline_nf_random/random_seed0_compactsmall_sweep"
OUT_DIR="figs/builderbench/c3t1_offline_crl_random/random_seed0_6x256_sweep"
OVERLAY_DIR="figs/builderbench/offline_learnability_nf_vs_crl"
FIXTURE="${NF_DIR}/fixed_heldout_eval_fixture.npz"
mkdir -p slurm "${OUT_DIR}" "${OVERLAY_DIR}"

echo "experiment=BuilderBench c3t1 fixed-offline random-policy CRL vs NF"
echo "env=builderbench_creative_3_task1 dims=state10 action5 goal9"
echo "dataset=reuse ${DATA_DIR} (npy only; no warp)"
echo "nf_fixture=${FIXTURE} (immutable; verify seed+91003)"
echo "scales=500k,2m,8m updates=5000 eval_every=250 batch=256"
echo "crl=forward InfoNCE + 0.01 logsumexp^2 6x256 relu repr64 lr=3e-4"
echo "raw achieved goals; no NF goal-stat mix / goal noise"
echo "DATA_DIR=${DATA_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "python=$(python --version 2>&1)"
nvidia-smi || true

for required in \
    "${DATA_DIR}/metadata.json" \
    "${DATA_DIR}/observations.npy" \
    "${DATA_DIR}/actions.npy" \
    "${FIXTURE}" \
    "${NF_DIR}/all_metrics.csv"
do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required existing file missing: ${required}" >&2
    exit 1
  fi
done
echo "dataset reuse: existing random-policy 8M dataset present"
echo "nf fixture reuse: ${FIXTURE}"

python -u - "${DATA_DIR}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
with (root / "metadata.json").open() as handle:
    meta = json.load(handle)
obs = np.load(root / "observations.npy", mmap_mode="r")
act = np.load(root / "actions.npy", mmap_mode="r")
assert list(obs.shape) == [160000, 51, 10], obs.shape
assert list(act.shape) == [160000, 50, 5], act.shape
assert meta["actual_transitions"] == 8_000_000
assert meta["achieved_goal_state_indices"] == list(range(9))
print("dataset validation PASS", obs.shape, act.shape)
PY

python -u scripts/train_builderbench_crl_offline.py \
  --dataset_dir="${DATA_DIR}" \
  --output_dir="${OUT_DIR}" \
  --nf_fixture_path="${FIXTURE}" \
  --nf_metrics_dir="${NF_DIR}" \
  --overlay_dir="${OVERLAY_DIR}" \
  --seed=0 \
  --dataset_sizes=500000,2000000,8000000 \
  --updates=5000 \
  --eval_every=250 \
  --batch_size=256 \
  --binary_samples=2048 \
  --categorical_samples=512 \
  --categorical_k=32 \
  --discount=0.99
