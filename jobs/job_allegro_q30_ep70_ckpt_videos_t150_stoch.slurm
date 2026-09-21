#!/bin/bash
#SBATCH --job-name=akt_q30_ep70_ckpt_vid_t150
#SBATCH --output=slurm/allegro_q30_ep70_ckpt_videos_t150_stoch_%j.log
#SBATCH --error=slurm/allegro_q30_ep70_ckpt_videos_t150_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# T=150 stochastic videos for the newest 10 q30 ep70 ckpts (skip existing).
#   sbatch jobs/job_allegro_q30_ep70_ckpt_videos_t150_stoch.slurm
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

CKPT_DIR="logs/ppo_allegro_kuka_nf_keeparm_bucket_ep70_500m_q30shift_seed0/ppo_allegro_kuka_throw_0/checkpoints"
OUT_DIR="logs/ppo_allegro_kuka_nf_keeparm_bucket_ep70_500m_q30shift_seed0/ppo_allegro_kuka_throw_0/videos_eval_t150_stoch"
LAST_N=10
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "T=150 stochastic  episodes=2  last-n=${LAST_N}  mark-horizon=50"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi || true

mapfile -t CKPTS < <(ls -1 "${CKPT_DIR}"/ckpt_iter_*.pkl 2>/dev/null | grep -v latest | sort | tail -n "${LAST_N}")
echo "n_ckpts=${#CKPTS[@]}"
FAILED=0
for ckpt in "${CKPTS[@]}"; do
  base="$(basename "${ckpt}" .pkl)"
  iter="${base#ckpt_iter_}"
  out="${OUT_DIR}/iter_${iter}_t150_stoch.mp4"
  if [ -f "${out}" ]; then
    echo "overwrite ${out}"
    rm -f "${out}"
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
