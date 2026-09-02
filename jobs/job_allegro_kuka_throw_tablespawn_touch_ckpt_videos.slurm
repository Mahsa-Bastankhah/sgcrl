#!/bin/bash
#SBATCH --job-name=akt_touch_ckptvid
#SBATCH --output=slurm/allegro_kuka_throw_tablespawn_touch_ckpt_videos_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_tablespawn_touch_ckpt_videos_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --account=seas

# Stochastic 1-ep videos + raw NF log-p / success overlay for table-spawn touch.
#   sbatch jobs/job_allegro_kuka_throw_tablespawn_touch_ckpt_videos.slurm
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

CKPT_DIR="${CKPT_DIR:-logs/ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_touch_noshape_mixtaskg_mix50_actorreset_palmgoal_extrew1_4h/ppo_allegro_kuka_throw_0/checkpoints}"
OUT_DIR="${OUT_DIR:-videos/allegro_kuka_throw/tablespawn_touch}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  ckpt_dir=${CKPT_DIR}  out=${OUT_DIR}"
nvidia-smi || true

FAILED=0
shopt -s nullglob
for ckpt in "${CKPT_DIR}"/ckpt_iter_*.pkl; do
  base="$(basename "${ckpt}" .pkl)"
  out="${OUT_DIR}/${base}_stoch_rawlogp.mp4"
  if [ -f "${out}" ]; then
    echo "SKIP ${base}: already have ${out}"
    continue
  fi
  echo "=== ${base} -> ${out} ==="
  if ! python -u scripts/allegro_kuka_throw_ckpt_video.py \
      --checkpoint="${ckpt}" \
      --output="${out}" \
      --episodes=1 \
      --fps=30 \
      --seed=0 \
      --pipeline=gpu \
      --reward-overlay \
      --dump-grads; then
    echo "FAIL ${base}"
    FAILED=1
  fi
done

ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null || true
if [ "${FAILED}" -ne 0 ]; then
  echo "one or more videos failed"
  exit 1
fi
