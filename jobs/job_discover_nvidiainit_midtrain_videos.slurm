#!/bin/bash
#SBATCH --job-name=akt_disc_mid_vid
#SBATCH --output=slurm/discover_nvidiainit_midtrain_videos_%j.log
#SBATCH --error=slurm/discover_nvidiainit_midtrain_videos_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# Mid-training DISCOVER *collect* policy (explore noise + typical UCB goal).
#   sbatch jobs/job_discover_nvidiainit_midtrain_videos.slurm
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
CONDA_PREFIX="/n/fs/mislresearch/miniconda3/envs/sgcrl_isaacgym"
ISAAC_PY="${CONDA_PREFIX}/bin/python"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/n/fs/mislresearch/isaacgym/python:/n/fs/mislresearch/IsaacGymEnvs:/n/fs/mislresearch/mb6458-tmp/ppo-rnd-flax-deps:/n/fs/mislresearch/sgcrl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

RUN=logs/new_allegrobaselines/discover_nvidiainit_z055_ep150_300m_seed0_14h/discover_allegro_kuka_throw_0
CKPT_DIR="${RUN}/checkpoints"
OUT_DIR="${RUN}/videos"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
nvidia-smi || true

for EPOCH in 0000200 0000900 0001800; do
  CKPT="${CKPT_DIR}/ckpt_epoch_${EPOCH}.pkl"
  if [ ! -f "${CKPT}" ]; then
    echo "SKIP missing ${CKPT}"
    continue
  fi
  echo "=== train collect policy + typical UCB (0,0,0.55)  epoch ${EPOCH}"
  "${ISAAC_PY}" -u scripts/discover_allegro_rollout_video.py \
    --checkpoint="${CKPT}" \
    --output="${OUT_DIR}/ckpt_epoch_${EPOCH}_train_ucb_0_00_0_00_0_55.mp4" \
    --goal-xyz=0.00,0.00,0.55 \
    --explore \
    --num-steps=150 \
    --episodes=2 \
    --fps=30 \
    --seed=0 \
    --pipeline=gpu
  ls -lh "${OUT_DIR}/ckpt_epoch_${EPOCH}_train_ucb"*.mp4
done
echo "done"
