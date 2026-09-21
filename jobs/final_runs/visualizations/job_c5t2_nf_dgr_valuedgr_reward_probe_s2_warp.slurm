#!/bin/bash
#SBATCH --job-name=c5t2_dgr_vid
#SBATCH --output=slurm/final_runs/visualizations/c5t2_nf_dgr_valuedgr_reward_probe_s2_warp_%j.log
#SBATCH --error=slurm/final_runs/visualizations/c5t2_nf_dgr_valuedgr_reward_probe_s2_warp_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --account=seas
#SBATCH --exclude=neu303

# Reward-probe videos for the final c5t2 NF dual-gradreg + valuedgr run
# (seed 2, ckpts 400 and 500: shortly after first train success at ~334).
# Deterministic. Require a successful episode (up to 16 reset seeds).
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
export BUILDERBENCH_MJX_IMPL=warp
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export JAX_PLATFORMS=cuda

ENV_NAME=builderbench_creative_5_task2
CKPT_DIR=logs/final_runs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_warp_2h30/ppo_builderbench_creative_5_task2_2/checkpoints
OUT_DIR=figs/builderbench/nf_logp_reward_probe/c5t2_dgr_valuedgr_s2_online_all
TAG_PREFIX=c5t2_dgr_valuedgr_s2_online
TARGET_ITERS="400 500"

mkdir -p slurm/final_runs/visualizations "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== c5t2 final dual-gradreg seed-2 reward probes ==="
echo "BUILDERBENCH_MJX_IMPL=${BUILDERBENCH_MJX_IMPL}"
echo "ckpt_dir=${CKPT_DIR}"
echo "out_dir=${OUT_DIR}"
echo "targets=${TARGET_ITERS}"
nvidia-smi || true
python -u -c "import os, warp, jax; print('warp', warp.__version__); print('JAX', jax.default_backend(), jax.devices())"

export CKPT_DIR TARGET_ITERS ENV_NAME OUT_DIR TAG_PREFIX
python -u - <<'PY'
import argparse
import importlib.util
import os
import sys
import traceback

spec = importlib.util.spec_from_file_location(
    'render_nf_traj_reward_video',
    os.path.join(os.getcwd(), 'scripts', 'render_nf_traj_reward_video.py'))
vid = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vid)

a = argparse.Namespace(
    checkpoint='',
    checkpoint_dir='',
    env=os.environ['ENV_NAME'],
    out_dir=os.environ['OUT_DIR'],
    tag='',
    tag_prefix=os.environ['TAG_PREFIX'],
    title='',
    allow_no_success=False,
    max_tries=16,
    seed=0,
    seeds='',
    stochastic=False,
    fps=8,
    skip_existing=True,
    show_select=False,
    show_pos_delta=False,
    normalize_reward=True,
    show_logp_grad=True,
    show_policy_loc_scale=False,
    show_policy_std=False,
    show_select_entropy=False,
    show_gae=False,
    match_run_init=False,
    fixed_start_x=vid.VIDEO_FIXED_START_X,
)

ckpt_dir = os.environ['CKPT_DIR']
entries = []
for it in os.environ['TARGET_ITERS'].split():
    path = os.path.join(ckpt_dir, f'ckpt_iter_{int(it):07d}.pkl')
    if not os.path.isfile(path):
        print(f'[probe] missing {path}', flush=True)
        continue
    entries.append((f'iter_{int(it):07d}', path))
if not entries:
    raise SystemExit('no checkpoints found')

import jax
print(f'[vid] jax={jax.default_backend()} devices={jax.devices()}', flush=True)
env_name = a.env
env_id = vid.sgcrl_env_name_to_bb_env_id(env_name)
num_cubes, _ = vid.parse_bb_env_id(env_id)
ctx = vid._load_train_ctx(env_name, entries[0][1])
vid.force_video_nopermute_norand(ctx, fixed_start_x=float(a.fixed_start_x))
print(f'[vid] forcing nopermute + fixed_start_x={ctx.fixed_start_x}', flush=True)
run_dir = vid._run_dir_from_ckpt(entries[0][1])
arch = vid._load_nf_arch(run_dir)
networks, nf_nets, goal_dim = vid._build_networks(env_name, a.seed, ctx, arch)
env, _base, mocap_targets, ep_len = vid._make_bb_env(env_id, ctx)

n_fail = 0
for label, path in entries:
    print(f'[probe] render {label} {path}', flush=True)
    try:
        vid._render_one(
            a, label=label, ckpt_path=path, ctx=ctx, networks=networks,
            nf_nets=nf_nets, env=env, mocap_targets=mocap_targets,
            ep_len=ep_len, num_cubes=num_cubes, goal_dim=goal_dim,
            arch=arch, out_dir=a.out_dir)
    except Exception:
        n_fail += 1
        traceback.print_exc()
        print(f'[probe] FAILED {label}', flush=True)

print(f'[probe] done n={len(entries)} failed={n_fail}', flush=True)
sys.exit(1 if n_fail else 0)
PY
PY_EXIT=$?

echo ""
echo "=== probe finished (PY_EXIT=${PY_EXIT}) ==="
ls -lh "${OUT_DIR}"/*.mp4 2>/dev/null || true
pkill -P $$ 2>/dev/null || true
exit ${PY_EXIT}
