#!/bin/bash
#SBATCH --job-name=vid_mpo_act
#SBATCH --output=slurm/final_metaworld_runs/vid_mpoinit_trackerr_action_%j.log
#SBATCH --error=slurm/final_metaworld_runs/vid_mpoinit_trackerr_action_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas

# Re-render the MPO-init seed0 videos with ‖mocap−TCP‖ + 4-D action overlay.
# Same recipe as the original stoch/varyinit videos: safe MPO init,
# stochastic policy, init_seed = 0..7 across the 8 checkpoints.
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
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_flow

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH=/n/fs/mislresearch/miniconda3/envs/sgcrl_flow/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${HOME}/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export JAX_PLATFORMS=cuda

CKPT_ROOT=logs/final_metaworld_runs/ppo_bin_nf_floorgrasp_safeinit_mpoinit_mixtaskg_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6/ppo_sawyer_bin_0/checkpoints
OUT_DIR=videos/final_metaworld_runs/bin_nf_mpoinit_s0_trackerr_action_overlay
ITERS=(0 4800 9600 14400 19600 24400 29200 34000)
mkdir -p slurm/final_metaworld_runs "${OUT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "overlaying ‖mocap-TCP‖ + action[-1,1] on MPO-init stoch/varyinit episodes"
nvidia-smi || true

SUMMARY="${OUT_DIR}/summary.jsonl"
: > "${SUMMARY}"

for i in "${!ITERS[@]}"; do
  ITER="${ITERS[$i]}"
  INIT_SEED="${i}"
  printf -v ITER_PAD "%07d" "${ITER}"
  CKPT="${CKPT_ROOT}/ckpt_iter_${ITER_PAD}.pkl"
  OUT="${OUT_DIR}/sawyer_bin_iter_${ITER_PAD}_init${INIT_SEED}_trackerr.mp4"
  echo "=== iter=${ITER} init_seed=${INIT_SEED} ==="
  if [ ! -f "${CKPT}" ]; then
    echo "MISSING ${CKPT}"
    continue
  fi
  python -u scripts/render_sawyer_mocap_tracking_error_video.py \
    --checkpoint="${CKPT}" \
    --env=sawyer_bin \
    --output="${OUT}" \
    --width=640 \
    --height=480 \
    --fps=30 \
    --rotate=180 \
    --camera=corner \
    --stochastic \
    --seed="${INIT_SEED}" \
    --ymax_cm=10 \
    --sawyer_bin_safe_grasp_reset \
    --bin_randomize_gripper_init \
    | tee -a "${SUMMARY}"
done

python -u - <<'PY'
import csv, glob, json, os, statistics
root = "videos/final_metaworld_runs/bin_nf_mpoinit_s0_trackerr_action_overlay"
rows = []
for path in sorted(glob.glob(os.path.join(root, "*_trackerr.csv"))):
  with open(path, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
      a_inf = r.get("a_inf", "")
      rows.append({
          "file": os.path.basename(path),
          "error_cm": float(r["error_cm"]),
          "slam": int(r["slam_jump"]),
          "a_inf": float(a_inf) if a_inf not in ("", None) else None,
      })
if not rows:
  raise SystemExit("no csv rows")
calm = [r["error_cm"] for r in rows if not r["slam"]]
slam = [r["error_cm"] for r in rows if r["slam"]]
all_e = [r["error_cm"] for r in rows]
ainf = [r["a_inf"] for r in rows if r["a_inf"] is not None]

def q(xs, p):
  xs = sorted(xs)
  if not xs:
    return None
  i = int(round((len(xs) - 1) * p))
  return xs[i]

report = {
    "n_frames": len(all_e),
    "n_slam_frames": len(slam),
    "all_median_cm": statistics.median(all_e),
    "all_p90_cm": q(all_e, 0.90),
    "all_p95_cm": q(all_e, 0.95),
    "all_max_cm": max(all_e),
    "calm_median_cm": statistics.median(calm) if calm else None,
    "calm_p90_cm": q(calm, 0.90),
    "calm_p95_cm": q(calm, 0.95),
    "calm_max_cm": max(calm) if calm else None,
    "slam_median_cm": statistics.median(slam) if slam else None,
    "slam_p90_cm": q(slam, 0.90),
    "slam_max_cm": max(slam) if slam else None,
    "action_absmax": max(ainf) if ainf else None,
    "frac_action_sat_0p95": (sum(v >= 0.95 for v in ainf) / len(ainf)) if ainf else None,
    "frac_action_tiny_0p10": (sum(v < 0.10 for v in ainf) / len(ainf)) if ainf else None,
    "suggested_thresholds_cm": [2.0, 3.0, 5.0],
}
out = os.path.join(root, "aggregate_stats.json")
with open(out, "w", encoding="utf-8") as fh:
  json.dump(report, fh, indent=2)
print("AGGREGATE", json.dumps(report, indent=2))
print("WROTE", out)
PY

echo "Done:"
ls -lh "${OUT_DIR}"
echo TRACKERR_ACTION_OVERLAY_JOB_DONE
