#!/bin/bash
#SBATCH --job-name=plot_16r_crl_nf
#SBATCH --output=slurm/plot_sixteenrooms_crl_nf_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --account=seas

# Batch CRL vs NF reward/rollout comparison for 2D SixteenRooms.
# Waits until NF seed-0 has MIN_CKPTS checkpoints, then plots all available pairs.
# Submit alongside training, e.g.:
#   sbatch --dependency=after:3310475_0 jobs/job_plot_sixteenrooms_crl_nf.slurm

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

ENV=point_SixteenRooms
CRL_SEED=120
NF_SEED=${NF_SEED:-0}
CRL_CKPT_DIR=logs/ppo_sixteenrooms/ppo_${ENV}_${CRL_SEED}/checkpoints
NF_CKPT_DIR=logs/ppo_nf_sixteenrooms/ppo_${ENV}_${NF_SEED}/checkpoints
OUT_DIR=figs/sixteenrooms_crl_nf_2d/all/

MIN_CKPTS=${MIN_CKPTS:-5}
if [ ! -d "${CRL_CKPT_DIR}" ] || [ -z "$(ls -A "${CRL_CKPT_DIR}"/ckpt_iter_*.pkl 2>/dev/null)" ]; then
  echo "ERROR: missing CRL checkpoints in ${CRL_CKPT_DIR}"
  exit 1
fi

echo "Waiting for >= ${MIN_CKPTS} NF checkpoints in ${NF_CKPT_DIR} ..."
while true; do
  n_nf=$(ls -1 "${NF_CKPT_DIR}"/ckpt_iter_*.pkl 2>/dev/null | wc -l)
  if [ "${n_nf}" -ge "${MIN_CKPTS}" ]; then
    echo "Found ${n_nf} NF checkpoints — starting plots."
    break
  fi
  echo "  only ${n_nf} NF ckpts so far; sleeping 120s ..."
  sleep 120
done

echo "CRL ckpts: ${CRL_CKPT_DIR}"
echo "NF  ckpts: ${NF_CKPT_DIR}"
echo "OUT: ${OUT_DIR}"

python -u scripts/plot_crl_nf_maze_compare.py \
  --env="${ENV}" \
  --crl_ckpt_dir="${CRL_CKPT_DIR}" \
  --nf_ckpt_dir="${NF_CKPT_DIR}" \
  --crl_seed="${CRL_SEED}" \
  --nf_seed="${NF_SEED}" \
  --output="${OUT_DIR}" \
  --heatmap_subcells 5
