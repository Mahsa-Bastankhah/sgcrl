#!/bin/bash
#SBATCH --job-name=regen_spawn_data_adjacent_room
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/gliu2/dist_matching/sgcrl/slurm_scripts/maniskill_close_subtask_train/%x-%j.out
#SBATCH --error=/home/gliu2/dist_matching/sgcrl/slurm_scripts/maniskill_close_subtask_train/%x-%j.err

# One-off: regenerate CloseSubtaskTrain-v0's spawn_data.pt cache using the
# new geodesic-aware "adjacent room" spawn selection in
# gen_close_spawn_data (mshab/utils/gen/gen_spawn_positions.py), replacing
# the previous ~3m-Euclidean-only version. Mirrors the earlier regeneration
# job's resource shape (32 CPUs, 1 GPU, 64GB, ~4hr) and, like that job,
# writes --output/--error to shared/NFS storage under this repo -- NOT
# /tmp, which is local to whichever compute node runs the job and is not
# readable from the login node afterward.

source /home/gliu2/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/gliu2/conda_envs/mshab_rl

export MS_ASSET_DIR=/data/user_data/gliu2/maniskill_data
export MSHAB_TASK=set_table
export MSHAB_SPLIT=train

REARRANGE_DIR="${MS_ASSET_DIR}/data/scene_datasets/replica_cad_dataset/rearrange"
SPAWN_DATA_DIR="${REARRANGE_DIR}/spawn_data/${MSHAB_TASK}/close/${MSHAB_SPLIT}"
SPAWN_DATA_FP="${SPAWN_DATA_DIR}/spawn_data.pt"
BACKUP_FP="${SPAWN_DATA_DIR}/spawn_data.pt.orig_3m_band_backup"

if [ ! -f "${BACKUP_FP}" ]; then
  echo "Backing up current (~3m band) spawn_data.pt to ${BACKUP_FP}"
  cp "${SPAWN_DATA_FP}" "${BACKUP_FP}"
else
  echo "Backup already exists at ${BACKUP_FP}, not overwriting it"
fi

python -m mshab.utils.gen.gen_spawn_positions \
    --task "${MSHAB_TASK}" \
    --subtask close \
    --split "${MSHAB_SPLIT}" \
    --num-workers 32
