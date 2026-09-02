#!/bin/bash
#SBATCH --job-name=akt_ckpt_vids
#SBATCH --output=slurm/allegro_kuka_throw_ckpt_videos_latest_%j.log
#SBATCH --error=slurm/allegro_kuka_throw_ckpt_videos_latest_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --account=seas

# Latest-ckpt policy videos for the Allegro throw / slide runs.
#   sbatch jobs/job_allegro_kuka_throw_ckpt_videos_latest.slurm
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

OUT_DIR="${OUT_DIR:-videos/allegro_kuka_throw}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  writing ${OUT_DIR}"
nvidia-smi || true
FAILED=0

latest_ckpt() {
  local d="$1"
  ls -1 "${d}"/ckpt_iter_*.pkl 2>/dev/null | grep -v latest | sort | tail -n 1
}

render_one() {
  local tag="$1"
  local ckpt_dir="$2"
  local ckpt
  ckpt="$(latest_ckpt "${ckpt_dir}")"
  if [ -z "${ckpt}" ]; then
    echo "FAIL ${tag}: no ckpt in ${ckpt_dir}"
    FAILED=1
    return 0
  fi
  local base
  base="$(basename "${ckpt}" .pkl)"
  local out="${OUT_DIR}/${tag}_${base}.mp4"
  if [ -f "${out}" ]; then
    echo "SKIP ${tag}: already have ${out}"
    return 0
  fi
  echo "=== ${tag}  ${ckpt} -> ${out} ==="
  if ! python -u scripts/allegro_kuka_throw_ckpt_video.py \
    --checkpoint="${ckpt}" \
    --output="${out}" \
    --episodes=2 \
    --fps=30 \
    --seed=0 \
    --pipeline=gpu \
    --deterministic; then
    echo "FAIL ${tag}"
    FAILED=1
  fi
}

render_one nfcsm_300m \
  logs/ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001/ppo_allegro_kuka_throw_0/checkpoints

render_one nfcsm_ep150 \
  logs/ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192_tau05_minstd1e5_entanneal_ep150_T150_300m_crl10_ent05to001_2h/ppo_allegro_kuka_throw_0/checkpoints

render_one nfc_palmgoal \
  logs/ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_palmgoal_stateonly_2h/ppo_allegro_kuka_throw_0/checkpoints

render_one nfcsm_norand_mtg \
  logs/ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_norand_mixtaskg_2h/ppo_allegro_kuka_throw_0/checkpoints

ls -lh "${OUT_DIR}"/*.mp4 || true
if [ "${FAILED}" -ne 0 ]; then
  echo "one or more videos failed"
  exit 1
fi
