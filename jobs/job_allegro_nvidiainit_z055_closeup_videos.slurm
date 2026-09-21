#!/bin/bash
#SBATCH --job-name=akt_nvidiainit_closeup_vid
#SBATCH --output=slurm/allegro_nvidiainit_z055_closeup_videos_%j.log
#SBATCH --error=slurm/allegro_nvidiainit_z055_closeup_videos_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# Re-render NVIDIA-init z=0.55 videos: closer cam, no pink cube marker, no T=50 chip.
#   sbatch jobs/job_allegro_nvidiainit_z055_closeup_videos.slurm
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

RUN=logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed0_8h/ppo_allegro_kuka_throw_0
CKPT_DIR="${RUN}/checkpoints"
OUT_DIR="${RUN}/videos"
mkdir -p "${OUT_DIR}"

for CKPT in \
  "${CKPT_DIR}/ckpt_iter_0002500.pkl" \
  "${CKPT_DIR}/ckpt_iter_0002600.pkl"
do
  if [ ! -f "${CKPT}" ]; then
    echo "SKIP missing ${CKPT}"
    continue
  fi
  ITER=$(basename "${CKPT}" .pkl | sed 's/ckpt_iter_//')
  OUT="${OUT_DIR}/iter_${ITER}_stoch_closeup.mp4"
  echo "=== ${CKPT} -> ${OUT}"
  python -u scripts/allegro_kuka_throw_ckpt_video.py \
    --checkpoint="${CKPT}" \
    --output="${OUT}" \
    --num-steps=150 \
    --episodes=1 \
    --fps=30 \
    --seed=0 \
    --pipeline=gpu \
    --mark-horizon=0
  ls -lh "${OUT}"
done
echo "done"
