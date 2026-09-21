#!/bin/bash
#SBATCH --job-name=akt_near10_045m060_ckpt700_vid
#SBATCH --output=slurm/allegro_near10_045m060_ckpt700_wide_video_%j.log
#SBATCH --error=slurm/allegro_near10_045m060_ckpt700_wide_video_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=seas

# Re-render hover (0.45,-0.60) ckpt 700 with auto-framed camera (bucket in view).
#   sbatch jobs/job_allegro_near10_045m060_ckpt700_wide_video.slurm
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

RUN_DIR="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_tablespawn_near10_above10_corr03_nvbucket_xy045m060_ep70_1b_q40shift_noreplace_inbucket_seed0_14h/ppo_allegro_kuka_throw_0"
CKPT="${RUN_DIR}/checkpoints/ckpt_iter_0000700.pkl"
OUT="${RUN_DIR}/videos/iter_0000700_stoch_wide.mp4"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "ckpt=${CKPT}"
echo "out=${OUT}"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_ckpt_video.py \
  --checkpoint="${CKPT}" \
  --output="${OUT}" \
  --num-steps=70 \
  --episodes=2 \
  --fps=30 \
  --seed=0 \
  --pipeline=gpu \
  --mark-horizon=70

ls -lh "${OUT}"
echo "done ${OUT}"
