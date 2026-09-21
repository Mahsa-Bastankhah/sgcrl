#!/bin/bash
#SBATCH --job-name=akt_q30_ep70_ckpt_eval_t150
#SBATCH --output=slurm/allegro_q30_ep70_ckpt_eval_t150_stoch_%j.log
#SBATCH --error=slurm/allegro_q30_ep70_ckpt_eval_t150_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas

# Offline T=150 stochastic eval of the newest 10 (then any new) q30 ep70 ckpts.
# Re-runnable: skips iters already in the CSV.
#   sbatch jobs/job_allegro_q30_ep70_ckpt_eval_t150_stoch.slurm
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

CKPT_DIR="logs/ppo_allegro_kuka_nf_keeparm_bucket_ep70_500m_q30shift_seed0/ppo_allegro_kuka_throw_0/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/q30_ep70_ckpt_eval_t150_stoch"
CSV="${OUT_DIR}/ckpt_eval_t150_stoch.csv"
PLOT="${OUT_DIR}/ckpt_eval_t150_stoch.png"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "T=150 E=256 stochastic  last-n=10  steps_per_iter=$((1024 * 70))"
echo "CKPT_DIR=${CKPT_DIR} n=$(ls -1 ${CKPT_DIR}/ckpt_iter_*.pkl | wc -l)"
nvidia-smi || true

python -u scripts/eval_allegro_ckpt_success.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --output-csv="${CSV}" \
  --output-plot="${PLOT}" \
  --plot-title="q30 keep-arm NVIDIA-bucket  train T=70  eval T=150 stoch  (roll mean w=5)" \
  --episode-length=150 \
  --num-envs=256 \
  --seed=0 \
  --pipeline=gpu \
  --stochastic \
  --last-n=10 \
  --steps-per-iter=$((1024 * 70))

echo "done"
ls -lh "${OUT_DIR}"
