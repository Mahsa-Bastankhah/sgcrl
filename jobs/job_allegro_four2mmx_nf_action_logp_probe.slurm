#!/bin/bash
#SBATCH --job-name=allegro_nf_sf_logp
#SBATCH --output=slurm/allegro_four2mmx_nf_sf_logp_probe_%j.log
#SBATCH --error=slurm/allegro_four2mmx_nf_sf_logp_probe_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas

# Usage:
#   sbatch jobs/job_allegro_four2mmx_nf_action_logp_probe.slurm \
#     /path/to/ckpt_iter_0000100.pkl
#
# Fixes a=g_norm and sweeps K=8 uniform future goals sf ~ U([-1,1]^d)
# to test whether log p(g|s,a) beats log p(sf|s,a).

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
export MPLBACKEND=Agg

CKPT="${1:?pass checkpoint path as first arg}"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

OUT_DIR="figs/allegro_kuka_throw/nf_sf_logp_probe"
mkdir -p slurm "${OUT_DIR}" "${MPLCONFIGDIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CKPT=${CKPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "sweep: log p(sf|s,a=g_norm) for true g + K=8 uniform sf"
nvidia-smi || true

python -u scripts/probe_allegro_nf_action_logp.py \
  --checkpoint="${CKPT}" \
  --out-dir="${OUT_DIR}" \
  --seed=0 \
  --pipeline=gpu \
  --num-sf=8 \
  --stride=5

ls -lh "${OUT_DIR}"/nf_sf_logp_*.{csv,png} | tail -n 20
