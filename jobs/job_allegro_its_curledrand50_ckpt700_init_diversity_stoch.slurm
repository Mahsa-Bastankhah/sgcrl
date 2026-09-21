#!/bin/bash
#SBATCH --job-name=its_ckpt700_initdiv
#SBATCH --output=slurm/allegro_its_curledrand50_ckpt700_init_diversity_stoch_%j.log
#SBATCH --error=slurm/allegro_its_curledrand50_ckpt700_init_diversity_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --account=seas

# Mid-ckpt init-diversity: ONE Isaac Gym process, many reset seeds, stochastic.
# Full mp4s for a few representative seeds; first-frame montage for all.
#
#   sbatch jobs/job_allegro_its_curledrand50_ckpt700_init_diversity_stoch.slurm

set -euo pipefail
unset LD_PRELOAD
module purge

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

RUN_DIR="logs/ppo_allegro_index_thumb_straight_curledrand50_normgoal_e1024_nf_tiny_sa2x128_r32_b4_w128_200m_ep150_minstd1e4_ent01fixed_mix50_extrew1_seed0_4h/ppo_allegro_kuka_throw_0"
CKPT="${RUN_DIR}/checkpoints/ckpt_iter_0000700.pkl"
OUT_DIR="videos/allegro_init_diversity_diag/its_curledrand50_mix50_seed0_ckpt700_stoch"
SEEDS="0,1,2,7,13,42,100,400,600,700,1100,9999"
# Full 150-step videos for a few seeds; all seeds contribute to montage.
FULL_SEEDS="0,42,400,700,1100,9999"

mkdir -p "${OUT_DIR}" slurm

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CKPT=${CKPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "SEEDS=${SEEDS}"
echo "FULL_SEEDS=${FULL_SEEDS}"
echo "policy=stochastic (no --deterministic)"
nvidia-smi || true

python -u scripts/allegro_ckpt_init_diversity_video.py \
  --checkpoint="${CKPT}" \
  --outdir="${OUT_DIR}" \
  --seeds="${SEEDS}" \
  --full-video-seeds="${FULL_SEEDS}" \
  --montage-steps=1 \
  --episodes=1 \
  --pipeline=gpu

echo "=== outputs ==="
ls -lh "${OUT_DIR}"
echo "DONE"
