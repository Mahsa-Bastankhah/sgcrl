#!/bin/bash
#SBATCH --job-name=vid_sgcrl_ppo
#SBATCH --output=/n/fs/mislresearch/sgcrl/slurm/vid_oldsgcrl_vs_ppo_%j.log
#SBATCH --error=/n/fs/mislresearch/sgcrl/slurm/vid_oldsgcrl_vs_ppo_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=seas

# Latest-ckpt rollouts: original SGCRL (old-sgcrl) vs PPO+NF (this repo).
unset LD_PRELOAD
module purge

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(
    printf '%s\n' "${LD_LIBRARY_PATH//:/$'\n'}" |
      grep -vF '/usr/local/cuda-11.3' |
      grep -vF '/usr/local/cuda-12' |
      paste -sd: -
  )"
fi

try_cuda() {
  module purge 2>/dev/null || true
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -z "${CUDA_HOME:-}" ] && return 1
    echo "Loaded CUDA modules: $*"
    return 0
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || true
if [ -z "${CUDA_HOME:-}" ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      echo "Fallback CUDA_HOME=${CUDA_HOME}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

OUT=/n/fs/mislresearch/sgcrl/videos/old_sgcrl_vs_ppo
OLD=/n/fs/mislresearch/old-sgcrl/logs
PPO_BIN=/n/fs/mislresearch/sgcrl/logs/final_metaworld_runs/ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_minstd1e5_extrew1_rand_mixtaskg
PPO_PEG=/n/fs/mislresearch/sgcrl/logs/final_metaworld_runs/ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_rand_minstd1e5_ent0005_mixtaskg
mkdir -p slurm "${OUT}"

echo "Rendering latest-ckpt videos → ${OUT}"
nvidia-smi || true

python -u scripts/q_sac_rollout_video.py \
  --run_dir="${OLD}/sawyer_bin_s1/contrastive_cpc_sawyer_bin_1" \
  --env=sawyer_bin --only_latest \
  --output="${OUT}/oldsgcrl_bin_s1_latest.mp4"

python -u scripts/q_sac_rollout_video.py \
  --run_dir="${OLD}/sawyer_bin_s0/contrastive_cpc_sawyer_bin_0" \
  --env=sawyer_bin --only_latest \
  --output="${OUT}/oldsgcrl_bin_s0_latest.mp4"

python -u scripts/q_sac_rollout_video.py \
  --run_dir="${OLD}/sawyer_peg_s1/contrastive_cpc_sawyer_peg_1" \
  --env=sawyer_peg --only_latest \
  --output="${OUT}/oldsgcrl_peg_s1_latest.mp4"

python -u scripts/ppo_rollout_video.py \
  --checkpoint="${PPO_BIN}/ppo_sawyer_bin_0/checkpoints/latest.pkl" \
  --env=sawyer_bin \
  --output="${OUT}/ppo_nf_bin_s0_latest.mp4"

python -u scripts/ppo_rollout_video.py \
  --checkpoint="${PPO_PEG}/ppo_sawyer_peg_2/checkpoints/latest.pkl" \
  --env=sawyer_peg \
  --output="${OUT}/ppo_nf_peg_s2_latest.mp4"

ls -lh "${OUT}"
