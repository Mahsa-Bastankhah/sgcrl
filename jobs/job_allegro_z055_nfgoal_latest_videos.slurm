#!/bin/bash
#SBATCH --job-name=akt_z055_nfgoal_vid
#SBATCH --output=slurm/allegro_z055_nfgoal_latest_videos_%j.log
#SBATCH --error=slurm/allegro_z055_nfgoal_latest_videos_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --account=seas

# Re-render latest z=0.55 successful-candidate ckpts with orange ring at NF goal.
#   sbatch jobs/job_allegro_z055_nfgoal_latest_videos.slurm
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
CONDA_PREFIX="/n/fs/mislresearch/miniconda3/envs/sgcrl_isaacgym"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/n/fs/mislresearch/isaacgym/python:/n/fs/mislresearch/IsaacGymEnvs:/n/fs/mislresearch/sgcrl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
nvidia-smi || true

for RUN in \
  logs/successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h/ppo_allegro_kuka_throw_0 \
  logs/successful_candidate/mid_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h/ppo_allegro_kuka_throw_0 \
  logs/successful_candidate/hard_bucket_arm_init_z055_ep80_q40_300m_seed0_4h/ppo_allegro_kuka_throw_0
do
  CKPT=$(ls -1 "${RUN}/checkpoints"/ckpt_iter_*.pkl 2>/dev/null | grep -v '_in_train_video' | sort | tail -1)
  if [ -z "${CKPT}" ]; then
    echo "SKIP no ckpt in ${RUN}"
    continue
  fi
  ITER=$(basename "${CKPT}" .pkl | sed 's/ckpt_iter_//')
  OUT="${RUN}/videos/iter_${ITER}_stoch_nfgoal.mp4"
  echo "=== ${RUN}"
  echo "ckpt=${CKPT}"
  echo "out=${OUT}"
  python -u scripts/allegro_kuka_throw_ckpt_video.py \
    --checkpoint="${CKPT}" \
    --output="${OUT}" \
    --num-steps=80 \
    --episodes=1 \
    --fps=30 \
    --seed=0 \
    --pipeline=gpu \
    --mark-horizon=0
  ls -lh "${OUT}"
done
echo "done"
