#!/bin/bash
#SBATCH --job-name=akt_ht_q05_s1_vid_t150
#SBATCH --output=slurm/allegro_hidetable_q05_seed1_ckpt_videos_t150_stoch_%j.log
#SBATCH --error=slurm/allegro_hidetable_q05_seed1_ckpt_videos_t150_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# Re-render hidetable q05 seed1 T=150 videos with RESET/SUCCESS banners.
#   sbatch jobs/job_allegro_hidetable_q05_seed1_ckpt_videos_t150_stoch.slurm
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
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

CKPT_DIR="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed1_14h/ppo_allegro_kuka_throw_1/checkpoints"
OUT_DIR="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed1_14h/ppo_allegro_kuka_throw_1/videos_eval_t150_stoch"
ITERS="8800 9000 9200 9400 9600 9764"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "hidetable q05 seed1  T=150 stochastic  episodes=2  mark-horizon=50"
echo "ITERS=${ITERS}"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi || true

FAILED=0
for iter in ${ITERS}; do
  ckpt=$(printf '%s/ckpt_iter_%07d.pkl' "${CKPT_DIR}" "${iter}")
  out=$(printf '%s/iter_%07d_t150_stoch.mp4' "${OUT_DIR}" "${iter}")
  if [ ! -f "${ckpt}" ]; then
    echo "MISSING ${ckpt}"
    FAILED=1
    continue
  fi
  echo "=== ${ckpt} -> ${out} ==="
  if ! python -u scripts/allegro_kuka_throw_ckpt_video.py \
    --checkpoint="${ckpt}" \
    --output="${out}" \
    --num-steps=150 \
    --episodes=2 \
    --fps=30 \
    --seed=0 \
    --pipeline=gpu \
    --mark-horizon=50; then
    echo "FAIL ${ckpt}"
    FAILED=1
  fi
done

ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null || true
if [ "${FAILED}" -ne 0 ]; then
  echo "one or more videos failed"
  exit 1
fi
echo "done"
