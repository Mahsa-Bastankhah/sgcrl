#!/bin/bash
#SBATCH --job-name=akt_nf_rank_rand
#SBATCH --output=slurm/allegro_hand16fig_trimsa_nf_rank_pi_vs_random_%j.log
#SBATCH --error=slurm/allegro_hand16fig_trimsa_nf_rank_pi_vs_random_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=seas

# NF rank acc on trim_sa zigzag ckpts: learned π vs Uniform[-1,1] actions.
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

CKPT_DIR="logs/ppo_allegro_hand16fig_hardzigzag_trimsa_randinit_e1024_nf_compactsmall_fully_scaled_ep50_200m_seed0_4h/ppo_allegro_kuka_throw_0/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/hand16fig_trimsa_nf_rank"
ITERS="0,1000,2000,3000,3900"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} ckpts=${ITERS} steps=300"
echo "CKPT_DIR=${CKPT_DIR}"
nvidia-smi || true

python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --output-csv="${OUT_DIR}/rank_learned_pi.csv" \
  --output-plot="${OUT_DIR}/rank_learned_pi.png" \
  --iterations="${ITERS}" \
  --num-steps=300 \
  --seed=0 \
  --min-samples=256 \
  --pipeline=gpu

python -u scripts/eval_allegro_nf_rank_accuracy.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --output-csv="${OUT_DIR}/rank_random_actions.csv" \
  --output-plot="${OUT_DIR}/rank_random_actions.png" \
  --iterations="${ITERS}" \
  --num-steps=300 \
  --seed=0 \
  --min-samples=256 \
  --pipeline=gpu \
  --random-actions

echo "done"
ls -lh "${OUT_DIR}"
