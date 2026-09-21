#!/bin/bash
#SBATCH --job-name=akt_zigzag_ckpt_ep200
#SBATCH --output=slurm/allegro_hand16fig_trimsa_fullrange_ckpt_eval_ep200_%j.log
#SBATCH --error=slurm/allegro_hand16fig_trimsa_fullrange_ckpt_eval_ep200_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas

# Offline success eval of zigzag trim_sa fullrange ckpts at T=200
# (train was T=50). Meant to run after 3861594 ends.
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

CKPT_DIR="logs/ppo_allegro_hand16fig_hardzigzag_trimsa_fullrange_e1024_nf_compactsmall_fully_scaled_ep50_200m_seed0_4h/ppo_allegro_kuka_throw_0/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/hand16fig_trimsa_fullrange_ckpt_eval_ep200"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "T=200 E=256 deterministic  all ckpt_iter_*.pkl"
echo "CKPT_DIR=${CKPT_DIR}"
ls -1 "${CKPT_DIR}"/ckpt_iter_*.pkl | wc -l
nvidia-smi || true

python -u scripts/eval_allegro_ckpt_success.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --output-csv="${OUT_DIR}/ckpt_eval_ep200.csv" \
  --output-plot="${OUT_DIR}/ckpt_eval_ep200.png" \
  --episode-length=200 \
  --num-envs=256 \
  --seed=0 \
  --pipeline=gpu \
  --deterministic \
  --steps-per-iter=$((1024 * 50))

echo "done"
ls -lh "${OUT_DIR}"
