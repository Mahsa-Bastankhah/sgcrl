#!/bin/bash
#SBATCH --job-name=akt_ht_rnd_s1_vid_t150
#SBATCH --output=slurm/allegro_hidetable_rnd_seed1_ckpt_videos_t150_stoch_%j.log
#SBATCH --error=slurm/allegro_hidetable_rnd_seed1_ckpt_videos_t150_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# Re-render hidetable PPO+RND seed1 last ckpts, T=150, RESET/SUCCESS banners.
#   sbatch jobs/job_allegro_hidetable_rnd_seed1_ckpt_videos_t150_stoch.slurm
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

CKPT_DIR="logs/ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h/checkpoints/allegro_kuka_throw__1__ppo-rnd__1789391632"
OUT_DIR="logs/ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h/videos_eval_t150_stoch"
# Eval steps ≈ train_iter/10. Match the NF remake cadence (8800..9764).
ITERS="880 900 920 940 960 976"
FLAGS_JSON="${OUT_DIR}/hidetable_rnd_flags.json"
mkdir -p slurm "${OUT_DIR}"
cat > "${FLAGS_JSON}" <<'EOF'
{
  "isaacgym_episode_length": 50,
  "isaacgym_pipeline": "gpu",
  "isaacgym_fixed_target_xyz": "0.00,-0.22,0.40",
  "isaacgym_randomize_init": false,
  "isaacgym_randomize_object_xyz": false,
  "isaacgym_randomize_object_shape": false,
  "isaacgym_palm_goal": false,
  "isaacgym_table_push": false,
  "isaacgym_table_spawn": true,
  "isaacgym_table_spawn_object_xy": "0.0,0.0",
  "isaacgym_table_spawn_in_hand": true,
  "isaacgym_table_spawn_in_hand_offset": 0.042,
  "isaacgym_table_spawn_in_hand_obj_noise": 0.0,
  "isaacgym_table_spawn_in_hand_keep_arm": true,
  "isaacgym_table_spawn_in_hand_wrist_offset": -3.141592653589793,
  "isaacgym_table_spawn_in_hand_wrist_noise": 0.0,
  "isaacgym_table_spawn_finger_curl_scale": 1.0,
  "isaacgym_table_spawn_finger_noise": 0.0,
  "isaacgym_table_spawn_arm_noise": 0.0,
  "isaacgym_hide_table": true,
  "isaacgym_large_table": false
}
EOF

echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "hidetable PPO+RND seed1  T=150 stochastic  episodes=2  mark-horizon=50"
echo "ITERS=${ITERS}"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi || true

FAILED=0
for iter in ${ITERS}; do
  ckpt=$(printf '%s/params_%d.pkl' "${CKPT_DIR}" "${iter}")
  out=$(printf '%s/params_%07d_t150_stoch.mp4' "${OUT_DIR}" "${iter}")
  if [ ! -f "${ckpt}" ]; then
    echo "MISSING ${ckpt}"
    FAILED=1
    continue
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
done

ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null || true
if [ "${FAILED}" -ne 0 ]; then
  echo "one or more videos failed"
  exit 1
fi
echo "done"
