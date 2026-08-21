#!/bin/bash
#SBATCH --job-name=c5t2_dgr_c100_gae
#SBATCH --output=slurm/bb_probe_c5t2_nfcs_dualgradreg_c100_lamlr1e6_ent0005to0001_gae_%j.log
#SBATCH --error=slurm/bb_probe_c5t2_nfcs_dualgradreg_c100_lamlr1e6_ent0005to0001_gae_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=05:59:00
#SBATCH --account=seas

# Same c=100 λ_lr=1e-6 (3718571) reward+video probes, plus GAE A_t / V(s_t).
# Renders every iter that already has an mp4 in the original online_all dir.
# Writes to a sibling _gae folder so the no-V videos stay.
unset LD_PRELOAD
module purge

try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -n "${CUDA_HOME:-}" ] && return 0
    for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
      [ -d "${_cand}/lib64" ] && export CUDA_HOME="${_cand}" && return 0
    done
    return 1
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || try_cuda cudatoolkit/12.4 || true
if [ -z "${CUDA_HOME:-}" ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh

export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

ENV_NAME=builderbench_creative_5_task2
CKPT_DIR=logs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect_dualgradreg_c100_lamlr1e6_ent0005to0001/ppo_builderbench_creative_5_task2_0/checkpoints
SRC_VID_DIR=figs/builderbench/nf_logp_reward_probe/c5t2_nfcs_catselect_dualgradreg_c100_lamlr1e6_ent0005to0001_s0_online_all
OUT_DIR=figs/builderbench/nf_logp_reward_probe/c5t2_nfcs_catselect_dualgradreg_c100_lamlr1e6_ent0005to0001_s0_online_gae
TAG_PREFIX=c5t2_nfcs_catselect_dualgradreg_c100_lamlr1e6_ent0005to0001_s0_online_gae

mkdir -p slurm "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"
export CKPT_DIR SRC_VID_DIR

echo "=== c5t2 dualgradreg c=100 λ_lr=1e-6 reward+video+V/GAE ==="
echo "ckpt_dir=${CKPT_DIR}"
echo "out_dir=${OUT_DIR}"
nvidia-smi || true

mapfile -t CKPTS < <(python - <<'PY'
import glob, os, re
ckpt_dir = os.environ["CKPT_DIR"]
src = os.environ["SRC_VID_DIR"]
iters = []
for p in glob.glob(os.path.join(src, "*_iter_*.mp4")):
    m = re.search(r"iter_(\d+)\.mp4$", p)
    if m:
        iters.append(int(m.group(1)))
iters = sorted(set(iters))
for it in iters:
    path = os.path.join(ckpt_dir, f"ckpt_iter_{it:07d}.pkl")
    if os.path.isfile(path):
        print(path)
PY
)

echo "selected ${#CKPTS[@]} checkpoint(s):"
printf '  %s\n' "${CKPTS[@]}"

PY_EXIT=0
for CKPT in "${CKPTS[@]}"; do
  echo ""
  echo "[probe] rendering ${CKPT}"
  python -u scripts/render_nf_traj_reward_video.py \
    --checkpoint="${CKPT}" \
    --env="${ENV_NAME}" \
    --out_dir="${OUT_DIR}" \
    --tag_prefix="${TAG_PREFIX}" \
    --allow_no_success \
    --max_tries=1 \
    --seed=0 \
    --fps=8 \
    --normalize_reward \
    --show_logp_grad \
    --show_gae \
    --skip_existing || PY_EXIT=$?
done

echo ""
echo "=== probe finished (PY_EXIT=${PY_EXIT}) ==="
ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null | tail -30 || true
pkill -P $$ 2>/dev/null || true
exit ${PY_EXIT}
