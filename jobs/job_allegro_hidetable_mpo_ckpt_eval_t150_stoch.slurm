#!/bin/bash
#SBATCH --job-name=akt_ht_mpo_eval_t150
#SBATCH --output=slurm/allegro_hidetable_mpo_ckpt_eval_t150_stoch_%A_%a.log
#SBATCH --error=slurm/allegro_hidetable_mpo_ckpt_eval_t150_stoch_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=seas
#SBATCH --array=0-3

# MPO-CRL hidetable seeds 0–3. Same T=150 metrics as NF seed1 eval.
# Only the last ~5 ckpts are kept on disk.
#   sbatch jobs/job_allegro_hidetable_mpo_ckpt_eval_t150_stoch.slurm
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/n/fs/mislresearch/sgcrl:/n/fs/mislresearch/sgcrl/baseline-agents:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SEED="${SLURM_ARRAY_TASK_ID}"
case "${SEED}" in
  0) CKPT_DIR="logs/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed0_14h/mpo_crl_allegro_kuka_throw_0/checkpoints" ;;
  1) CKPT_DIR="logs/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h/mpo_crl_allegro_kuka_throw_1/checkpoints" ;;
  2) CKPT_DIR="logs/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed2_20h/mpo_crl_allegro_kuka_throw_2/checkpoints" ;;
  3) CKPT_DIR="logs/mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold_hidetable_bucket_xy0022_norand_ep50_500m_seed3_20h/mpo_crl_allegro_kuka_throw_3/checkpoints" ;;
  *) echo "bad seed ${SEED}"; exit 1 ;;
esac
OUT_DIR="figs/allegro_kuka_throw/mpo_hidetable_ckpt_eval_ep150_stoch/seed${SEED}"
mkdir -p slurm "${OUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} seed=${SEED}"
echo "hidetable MPO-CRL  T=150 E=256 stochastic  all kept ckpts"
echo "CKPT_DIR=${CKPT_DIR}"
nvidia-smi || true

python -u scripts/eval_allegro_ckpt_success.py \
  --checkpoint-dir="${CKPT_DIR}" \
  --output-csv="${OUT_DIR}/ckpt_eval_t150_stoch.csv" \
  --output-plot="${OUT_DIR}/ckpt_eval_t150_stoch.png" \
  --plot-title="hidetable MPO-CRL seed${SEED}  train T=50  eval T=150 stoch  (roll mean w=5)" \
  --episode-length=150 \
  --num-envs=256 \
  --seed=0 \
  --pipeline=gpu \
  --stochastic \
  --steps-per-iter=$((1024 * 50))

echo "done"
ls -lh "${OUT_DIR}"
