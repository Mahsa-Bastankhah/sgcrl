#!/bin/bash
#SBATCH --job-name=akt_keeparm_follow_vids
#SBATCH --output=slurm/allegro_kuka_throw_keeparm_follow_videos_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_keeparm_follow_videos_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --account=seas

# Follow-cube side videos for every keep-arm checkpoint. Not a train job.
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
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

RUN="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_largetable15_objgoal_norand_ep50_200m_seed0_4h/ppo_allegro_kuka_throw_0"
CKPT_DIR="${RUN}/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/inhand_keep_arm/follow_videos"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  follow-cube videos  ${CKPT_DIR} -> ${OUT_DIR}"
echo "NOT training"
nvidia-smi || true

shopt -s nullglob
CKPTS=("${CKPT_DIR}"/ckpt_iter_*.pkl)
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "no checkpoints in ${CKPT_DIR}"
  exit 1
fi
echo "n_ckpts=${#CKPTS[@]}"

RC=0
for CKPT in "${CKPTS[@]}"; do
  BASE="$(basename "${CKPT}" .pkl)"
  echo "=== ${BASE} ==="
  python -u scripts/allegro_kuka_throw_ckpt_video.py \
    --checkpoint="${CKPT}" \
    --output="${OUT_DIR}/${BASE}_follow.mp4" \
    --episodes=1 \
    --fps=20 \
    --seed=0 \
    --pipeline=gpu \
    --follow-cube \
    --write-still \
    || RC=$?
done
ls -lh "${OUT_DIR}" || true
exit $RC
