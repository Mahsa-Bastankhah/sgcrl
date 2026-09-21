#!/bin/bash
#SBATCH --job-name=akt_000m080_reframe
#SBATCH --output=slurm/allegro_nvbucket_xy000m080_ckpt_videos_reframe_%j.log
#SBATCH --error=slurm/allegro_nvbucket_xy000m080_ckpt_videos_reframe_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --account=seas

# Re-render (0.00,-0.80) keep-arm videos with the auto-framed camera.
# Overwrites the existing iter_*_stoch.mp4 so the same paths stay valid.
#   sbatch jobs/job_allegro_nvbucket_xy000m080_ckpt_videos_reframe.slurm
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

ITERS="100 200 300 400"
mkdir -p slurm
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "reframe (0.00,-0.80)  episodes=1 stoch  no-follow-cube  ITERS=${ITERS}"
nvidia-smi || true

FAILED=0
for seed in 0 1; do
  run="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_nvbucket_xy000m080_norand_ep50_300m_q05shift_noreplace_inbucket_seed${seed}_4h/ppo_allegro_kuka_throw_${seed}"
  ckpt_dir="${run}/checkpoints"
  out_dir="${run}/videos"
  mkdir -p "${out_dir}"
  for iter in ${ITERS}; do
    ckpt=$(printf '%s/ckpt_iter_%07d.pkl' "${ckpt_dir}" "${iter}")
    out=$(printf '%s/iter_%07d_stoch.mp4' "${out_dir}" "${iter}")
    if [ ! -f "${ckpt}" ]; then
      echo "MISSING ${ckpt}"
      FAILED=1
      continue
    fi
    echo "=== seed=${seed} ${ckpt} -> ${out} ==="
    if ! python -u scripts/allegro_kuka_throw_ckpt_video.py \
      --checkpoint="${ckpt}" \
      --output="${out}" \
      --episodes=1 \
      --fps=30 \
      --seed=0 \
      --pipeline=gpu \
      --no-follow-cube; then
      echo "FAIL ${ckpt}"
      FAILED=1
    fi
  done
done

ls -lh \
  logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_nvbucket_xy000m080_norand_ep50_300m_q05shift_noreplace_inbucket_seed0_4h/ppo_allegro_kuka_throw_0/videos/*.mp4 \
  logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_nvbucket_xy000m080_norand_ep50_300m_q05shift_noreplace_inbucket_seed1_4h/ppo_allegro_kuka_throw_1/videos/*.mp4 \
  2>/dev/null || true
if [ "${FAILED}" -ne 0 ]; then
  echo "one or more videos failed"
  exit 1
fi
echo "done"
