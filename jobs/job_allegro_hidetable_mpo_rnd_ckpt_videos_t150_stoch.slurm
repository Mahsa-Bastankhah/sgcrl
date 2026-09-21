#!/bin/bash
#SBATCH --job-name=akt_ht_mpo_rnd_vid_t150
#SBATCH --output=slurm/allegro_hidetable_mpo_rnd_ckpt_videos_t150_stoch_%j.log
#SBATCH --error=slurm/allegro_hidetable_mpo_rnd_ckpt_videos_t150_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# A handful of T=150 stoch clips (2 eps), not every checkpoint.
# MPO 500M seed1 last 5 + RND 500M seed1 four earlier iters (880..976 already exist).
#   sbatch jobs/job_allegro_hidetable_mpo_rnd_ckpt_videos_t150_stoch.slurm
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
CONDA_PREFIX="/n/fs/mislresearch/miniconda3/envs/sgcrl_isaacgym"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/n/fs/mislresearch/isaacgym/python:/n/fs/mislresearch/IsaacGymEnvs:/n/fs/mislresearch/mb6458-tmp/ppo-rnd-flax-deps:/n/fs/mislresearch/sgcrl:/n/fs/mislresearch/builderbench${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

FLAGS_JSON="logs/ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h/videos_eval_t150_stoch/hidetable_rnd_flags.json"
MPO_CKPT_DIR="logs/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h/mpo_crl_allegro_kuka_throw_1/checkpoints"
MPO_OUT="logs/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h/mpo_crl_allegro_kuka_throw_1/videos_eval_t150_stoch"
RND_CKPT_DIR="logs/ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h/checkpoints/allegro_kuka_throw__1__ppo-rnd__1789391632"
RND_OUT="logs/ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h/videos_eval_t150_stoch"
mkdir -p slurm "${MPO_OUT}" "${RND_OUT}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "T=150 stochastic  episodes=2  mark-horizon=50  (handful of ckpts only)"
nvidia-smi || true

FAILED=0
render() {
  local ckpt="$1" out="$2"
  if [ ! -f "${ckpt}" ]; then
    echo "MISSING ${ckpt}"
    FAILED=1
    return
  fi
  if [ -f "${out}" ]; then
    echo "SKIP exists ${out}"
    return
  fi
  echo "=== ${ckpt} -> ${out} ==="
  if ! python -u scripts/allegro_kuka_throw_ckpt_video.py \
    --checkpoint="${ckpt}" \
    --flags-json="${FLAGS_JSON}" \
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
}

for iter in 3300 3400 3500 3600 3700; do
  render \
    "$(printf '%s/ckpt_iter_%07d.pkl' "${MPO_CKPT_DIR}" "${iter}")" \
    "$(printf '%s/iter_%07d_t150_stoch.mp4' "${MPO_OUT}" "${iter}")"
done

for iter in 800 820 840 860; do
  render \
    "$(printf '%s/params_%d.pkl' "${RND_CKPT_DIR}" "${iter}")" \
    "$(printf '%s/params_%07d_t150_stoch.mp4' "${RND_OUT}" "${iter}")"
done

echo "--- MPO ---"
ls -lh "${MPO_OUT}"/*.mp4 2>/dev/null || true
echo "--- RND ---"
ls -lh "${RND_OUT}"/*.mp4 2>/dev/null || true
if [ "${FAILED}" -ne 0 ]; then
  echo "one or more videos failed"
  exit 1
fi
echo "done"
