#!/bin/bash
#SBATCH --job-name=akt_rnd_050045_latest_vid
#SBATCH --output=slurm/allegro_rnd_near10_050045_latest_ckpt_video_%j.log
#SBATCH --error=slurm/allegro_rnd_near10_050045_latest_ckpt_video_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=seas

# Stochastic T=70 clip from the latest hover RND ckpt (3884294).
#   sbatch jobs/job_allegro_rnd_near10_050045_latest_ckpt_video.slurm
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

RUN_DIR="logs/ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03_nvbucket_xy050045_noshape_ep70_500m_inbucket_seed0_7h"
CKPT_DIR="${RUN_DIR}/checkpoints/allegro_kuka_throw__0__ppo-rnd__1789481151"
OUT_DIR="${RUN_DIR}/videos_latest_t70_stoch"
FLAGS_JSON="${OUT_DIR}/rnd_near10_050045_flags.json"
mkdir -p slurm "${OUT_DIR}"
cat > "${FLAGS_JSON}" <<'EOF'
{
  "isaacgym_episode_length": 70,
  "isaacgym_pipeline": "gpu",
  "isaacgym_fixed_target_xyz": "0.50,-0.45,0.40",
  "isaacgym_throw_success": "in_bucket",
  "isaacgym_randomize_init": false,
  "isaacgym_randomize_object_xyz": false,
  "isaacgym_randomize_object_shape": false,
  "isaacgym_palm_goal": false,
  "isaacgym_table_push": false,
  "isaacgym_table_spawn": true,
  "isaacgym_table_spawn_object_xy": "0.20,0.08",
  "isaacgym_table_spawn_behind": true,
  "isaacgym_table_spawn_behind_dy": 0.10,
  "isaacgym_table_spawn_behind_above": 0.10,
  "isaacgym_table_spawn_correlated_xy": 0.03,
  "isaacgym_table_spawn_finger_curl_scale": 0.50,
  "isaacgym_table_spawn_finger_noise": 0.0,
  "isaacgym_table_spawn_arm_noise": 0.0,
  "isaacgym_hide_table": false,
  "isaacgym_large_table": false
}
EOF

CKPT=$(ls -1 "${CKPT_DIR}"/params_*.pkl | sort -t_ -k2 -n | tail -1)
ITER=$(basename "${CKPT}" .pkl | sed 's/params_//')
OUT=$(printf '%s/params_%07d_t70_stoch.mp4' "${OUT_DIR}" "${ITER}")

echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "ckpt=${CKPT}"
echo "out=${OUT}"
nvidia-smi || true

python -u scripts/allegro_kuka_throw_ckpt_video.py \
  --checkpoint="${CKPT}" \
  --flags-json="${FLAGS_JSON}" \
  --output="${OUT}" \
  --num-steps=70 \
  --episodes=2 \
  --fps=30 \
  --seed=0 \
  --pipeline=gpu \
  --mark-horizon=70

ls -lh "${OUT}"
echo "done ${OUT}"
