#!/bin/bash
#SBATCH --job-name=build_open_spawn_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gliu2/dist_matching/sgcrl/slurm_scripts/maniskill_open_subtask_train/%x-%j.out
#SBATCH --error=/home/gliu2/dist_matching/sgcrl/slurm_scripts/maniskill_open_subtask_train/%x-%j.err

# One-off: build OpenSubtaskTrain-v0's open/train/spawn_data.pt by reusing
# CloseSubtaskTrain-v0's already-validated (adjacent-room) robot spawn
# poses (keyed by the close<->open uid mapping) and sampling a fresh
# "mostly closed" articulation_qpos per uid -- see
# scripts/build_open_spawn_data.py's module docstring for the full
# rationale. Unlike regen_spawn_data_adjacent_room.slurm (the close-side
# ~4hr geodesic collision-search regen this deliberately does NOT repeat),
# this is just 63 cheap CPU-only scene reconfigures + tensor bookkeeping --
# no GPU, no collision search -- so modest resources/time suffice.

# Same conda activation pattern as close_subtask_train.slurm -- see that
# script's comment for why `source activate base; conda activate <env>`
# (open_cabinet_drawer.slurm's original pattern) was tried and empirically
# failed on a real compute node.
source /home/gliu2/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/gliu2/conda_envs/mshab_rl
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/data/user_data/gliu2/conda_envs/mshab_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gliu2/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# Same torch-bundled-libcusolver-vs-system-CUDA-11 fix as
# close_subtask_train.slurm -- see that script's comment for the full
# explanation.
TORCH_CUSOLVER_LIB=$(python -c "import os, nvidia.cusolver; print(os.path.join(os.path.dirname(nvidia.cusolver.__file__), 'lib'))" 2>/dev/null)
if [ -n "${TORCH_CUSOLVER_LIB}" ] && [ -d "${TORCH_CUSOLVER_LIB}" ]; then
  export LD_LIBRARY_PATH="${TORCH_CUSOLVER_LIB}:${LD_LIBRARY_PATH}"
fi
export CPATH=/data/user_data/gliu2/conda_envs/mshab_rl/include

export MS_ASSET_DIR=/data/user_data/gliu2/maniskill_data
export MSHAB_TASK=set_table
export MSHAB_SPLIT=train
export MSHAB_OBJ=kitchen_counter

REARRANGE_DIR="${MS_ASSET_DIR}/data/scene_datasets/replica_cad_dataset/rearrange"
CLOSE_SPAWN_DATA_FP="${REARRANGE_DIR}/spawn_data/${MSHAB_TASK}/close/${MSHAB_SPLIT}/spawn_data.pt"
OPEN_TASK_PLAN_FP="${REARRANGE_DIR}/task_plans/${MSHAB_TASK}/open/${MSHAB_SPLIT}/${MSHAB_OBJ}.json"

if [ ! -f "${CLOSE_SPAWN_DATA_FP}" ]; then
  echo "ERROR: close spawn data not found (needed as the source of robot"
  echo "spawn poses): ${CLOSE_SPAWN_DATA_FP}"
  echo "Run close_subtask_train.slurm's asset-download step first, and"
  echo "regen_spawn_data_adjacent_room.slurm if the adjacent-room band is"
  echo "not yet built."
  exit 1
fi
if [ ! -f "${OPEN_TASK_PLAN_FP}" ]; then
  echo "ERROR: open task plan not found: ${OPEN_TASK_PLAN_FP}"
  echo "Run: python -m mani_skill.utils.download_asset ReplicaCADRearrange -y"
  exit 1
fi

python /home/gliu2/dist_matching/sgcrl/scripts/build_open_spawn_data.py
