#!/bin/bash
#SBATCH --job-name=akt_ht_q05_s1_eval_t150
#SBATCH --output=slurm/allegro_hidetable_q05_seed1_ckpt_eval_t150_stoch_%j.log
#SBATCH --error=slurm/allegro_hidetable_q05_seed1_ckpt_eval_t150_stoch_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --account=seas

# Every hidetable q05 seed1 ckpt, T=150 stochastic, E=256.
# Reports 7.5cm success and ever-inside-bucket (12cm x 19.8cm cylinder).
#   sbatch jobs/job_allegro_hidetable_q05_seed1_ckpt_eval_t150_stoch.slurm
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

CKPT_DIR="logs/ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed1_14h/ppo_allegro_kuka_throw_1/checkpoints"
OUT_DIR="figs/allegro_kuka_throw/seed1_q05_ckpt_eval_ep150_stoch"
CSV="${OUT_DIR}/ckpt_eval_t150_stoch.csv"
PLOT="${OUT_DIR}/ckpt_eval_t150_stoch.png"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "ppo_eval_interval=n/a (offline ckpt eval)"
echo "hidetable q05 seed1  T=150 E=256 stochastic  all ckpts"
echo "metrics: ever ≤7.5cm  +  ever in bucket cylinder"
echo "CKPT_DIR=${CKPT_DIR} n=$(ls -1 ${CKPT_DIR}/ckpt_iter_*.pkl | wc -l)"
nvidia-smi || true

python -u scripts/eval_allegro_ckpt_success.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --output-csv="${CSV}" \
  --output-plot="${PLOT}" \
  --plot-title="hidetable q05 seed1  train T=50  eval T=150 stoch  (roll mean w=5)" \
  --episode-length=150 \
  --num-envs=256 \
  --seed=0 \
  --pipeline=gpu \
  --stochastic \
  --steps-per-iter=$((1024 * 50))

echo "done"
ls -lh "${OUT_DIR}"
