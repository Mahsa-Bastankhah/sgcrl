#!/usr/bin/env python3
"""Login-node watcher: spawn 200M RND when an NF throw eval stays above 0.2.

If NF eval success > 0.2 for 3 consecutive eval checkpoints, submit a matching
200M PPO+RND job (same env / success / T). Cancel that RND if it later
succeeds at high numbers; keep logs. Does not use a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import plot_allegro_hidetable_running_success as hidetable  # noqa: E402
import plot_allegro_live_queue_success as live  # noqa: E402
import plot_builderbench_train_success1000 as base  # noqa: E402

STATE_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'nf_eval_spawn_rnd_state.json')
AUTO_PATH = live.AUTO_RND_PATH
NF_THRESH = 0.20
NF_CONSEC = 3
RND_HIGH = 0.50
RND_HIGH_CONSEC = 2
RND_EASY = 0.80
RND_EASY_STEPS = 40_000_000
RND_STUCK_STEPS = 100_000_000
RND_STUCK_MAX = 0.20
NUM_STEPS = 200_000_000
NUM_ENVS = 1024

RND_BOOLS = (
    'isaacgym_table_push', 'isaacgym_randomize_init',
    'isaacgym_randomize_object_xyz', 'isaacgym_randomize_object_shape',
    'isaacgym_palm_goal', 'isaacgym_palm_and_object_success',
    'isaacgym_large_table', 'isaacgym_table_spawn',
    'isaacgym_table_spawn_behind', 'isaacgym_table_spawn_in_hand',
    'isaacgym_table_spawn_in_hand_keep_arm', 'isaacgym_hide_table',
)
RND_VALS = (
    'isaacgym_table_push_xyz', 'isaacgym_episode_length', 'isaacgym_pipeline',
    'isaacgym_palm_goal_xyz', 'isaacgym_throw_success',
    'isaacgym_table_spawn_object_xy', 'isaacgym_table_spawn_behind_dy',
    'isaacgym_table_spawn_behind_above', 'isaacgym_table_spawn_correlated_xy',
    'isaacgym_table_spawn_finger_curl_scale',
    'isaacgym_table_spawn_finger_noise', 'isaacgym_table_spawn_arm_noise',
    'isaacgym_table_spawn_in_hand_offset',
    'isaacgym_table_spawn_in_hand_obj_noise',
    'isaacgym_table_spawn_in_hand_wrist_offset',
    'isaacgym_table_spawn_in_hand_wrist_noise', 'isaacgym_fixed_target_xyz',
)


def _load_json(path: str, default):
  if not os.path.isfile(path):
    return default
  with open(path, encoding='utf-8') as fh:
    return json.load(fh)


def _dump_json(path: str, data) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + '.tmp'
  with open(tmp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, sort_keys=True)
    fh.write('\n')
  os.replace(tmp, path)


def _as_bool(val) -> bool:
  if isinstance(val, bool):
    return val
  return str(val).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _run_config_flags(log_dir: str) -> dict:
  root = os.path.join(REPO, 'logs', log_dir)
  candidates = [
      os.path.join(root, 'ppo_allegro_kuka_throw_0', 'run_config.json'),
      os.path.join(root, 'run_config.json'),
  ]
  if os.path.isdir(root):
    for name in sorted(os.listdir(root)):
      cand = os.path.join(root, name, 'run_config.json')
      candidates.append(cand)
  for path in candidates:
    if not os.path.isfile(path):
      continue
    with open(path, encoding='utf-8') as fh:
      data = json.load(fh)
    if isinstance(data, dict) and isinstance(data.get('flags'), dict):
      return data['flags']
    if isinstance(data, dict):
      return data
  return {}


def _flags_from_slurm(script: str) -> dict:
  if not script or not os.path.isfile(script):
    return {}
  text = open(script, encoding='utf-8', errors='replace').read()
  ep_m = re.search(r'^EP_LEN=(\d+)', text, re.M)
  ep_s = ep_m.group(1) if ep_m else ''
  if ep_s:
    text = text.replace('${EP_LEN}', ep_s).replace('"${EP_LEN}"', ep_s)
  flags = {}
  for match in re.finditer(
      r'--(no-?)?(isaacgym[_\-][A-Za-z0-9_\-]+)(?:=(\S+))?', text):
    no_pfx, raw, val = match.group(1), match.group(2), match.group(3)
    key = raw.replace('-', '_')
    if no_pfx:
      flags[key] = False
    elif val is None:
      flags[key] = True
    else:
      val = val.strip().strip('"').strip("'")
      if '${' in val:
        continue
      flags[key] = val
  if ep_s:
    flags['isaacgym_episode_length'] = int(ep_s)
  return flags


def _xyz_str(val, default='0.5,-0.3,0.4') -> str:
  if val is None:
    return default
  if isinstance(val, (list, tuple)):
    return ','.join(str(float(v)) for v in val)
  return str(val)


def _fingerprint(flags: dict) -> str:
  tgt = _xyz_str(flags.get('isaacgym_fixed_target_xyz'))
  succ = str(flags.get('isaacgym_throw_success', 'in_bucket'))
  ep = int(float(flags.get('isaacgym_episode_length', 50)))
  spawn = _as_bool(flags.get('isaacgym_table_spawn', False))
  behind = _as_bool(flags.get('isaacgym_table_spawn_behind', False))
  keep = _as_bool(flags.get('isaacgym_table_spawn_in_hand_keep_arm', False))
  hide = _as_bool(flags.get('isaacgym_hide_table', False))
  rinit = _as_bool(flags.get('isaacgym_randomize_init', False))
  obj_raw = flags.get('isaacgym_table_spawn_object_xy', '')
  obj_xy = _xyz_str(obj_raw, '') if obj_raw not in ('', None) else ''
  dy = str(flags.get('isaacgym_table_spawn_behind_dy', ''))
  dz = str(flags.get('isaacgym_table_spawn_behind_above', ''))
  corr = str(flags.get('isaacgym_table_spawn_correlated_xy', ''))
  return '|'.join([
      f'tgt={tgt}', f'succ={succ}', f'ep={ep}', f'spawn={int(spawn)}',
      f'behind={int(behind)}', f'keep={int(keep)}', f'hide={int(hide)}',
      f'rinit={int(rinit)}', f'obj={obj_xy}', f'dy={dy}', f'dz={dz}',
      f'corr={corr}',
  ])


def _xy_token(xyz: str) -> str:
  parts = [float(v) for v in str(xyz).split(',')]
  x, y = parts[0], parts[1]

  def _cm(v: float) -> str:
    sign = 'm' if v < 0 else ''
    return f'{sign}{abs(v) * 100:03.0f}'

  return f'{_cm(x)}{_cm(y)}'


def _succ_token(succ: str) -> str:
  return {
      'in_bucket': 'inbkt',
      'goal_ball': 'ball75',
      'nvidia_goal': 'nvg',
  }.get(str(succ), str(succ)[:6])


def _init_token(flags: dict) -> str:
  if _as_bool(flags.get('isaacgym_table_spawn_in_hand_keep_arm', False)):
    return 'keeparm'
  if _as_bool(flags.get('isaacgym_table_spawn_behind', False)):
    return 'near10'
  if _as_bool(flags.get('isaacgym_table_spawn', False)):
    return 'spawn'
  return 'nvidiainit'


def _pretty_label(flags: dict) -> str:
  xyz = _xyz_str(flags.get('isaacgym_fixed_target_xyz'))
  parts = [float(v) for v in xyz.split(',')]
  xy = f'({parts[0]:.2f},{parts[1]:.2f})'
  succ = str(flags.get('isaacgym_throw_success', 'in_bucket'))
  succ_l = {
      'in_bucket': 'in-bucket',
      'goal_ball': 'goal-ball 7.5cm',
      'nvidia_goal': 'NV-goal-ball 11.25cm',
  }.get(succ, succ)
  ep = int(float(flags.get('isaacgym_episode_length', 50)))
  hide = ' hidetable' if _as_bool(flags.get('isaacgym_hide_table', False)) else ''
  return f'RND {_init_token(flags)}{hide} T={ep} 200M  {succ_l}  {xy}'


def _tyro_flags(flags: dict, ep: int) -> list[str]:
  out = []
  merged = dict(flags)
  merged.setdefault('isaacgym_episode_length', ep)
  merged.setdefault('isaacgym_pipeline', 'gpu')
  merged.setdefault('isaacgym_throw_success', 'in_bucket')
  merged.setdefault('isaacgym_table_push', False)
  for key in RND_BOOLS:
    if key not in merged:
      continue
    name = key.replace('_', '-')
    out.append(f'--{name}' if _as_bool(merged[key]) else f'--no-{name}')
  for key in RND_VALS:
    if key not in merged:
      continue
    name = key.replace('_', '-')
    out.append(f'--{name}={merged[key]}')
  return out


def _consec_above(ys: list[float], thresh: float, n: int) -> bool:
  run = 0
  for y in ys:
    if y > thresh:
      run += 1
      if run >= n:
        return True
    else:
      run = 0
  return False


def _nf_eval_ys(log_dir: str) -> list[float]:
  series = base._read_eval_seed_series(base.LOG_ROOT, log_dir)
  best = []
  for pts in series or []:
    cand = [float(p[1]) for p in pts]
    if len(cand) > len(best):
      best = cand
  return best


def _rnd_series(log_dir: str) -> list[tuple[int, float]]:
  return hidetable._read_rnd_eval(log_dir) or []


def _existing_rnd_fingerprints() -> dict[str, dict]:
  found = {}
  for path in sorted(os.listdir(os.path.join(REPO, 'jobs'))):
    if not path.startswith('job_ppo_rnd_allegro_kuka_throw'):
      continue
    if '200m' not in path:
      continue
    flags = _flags_from_slurm(os.path.join(REPO, 'jobs', path))
    if not flags:
      continue
    try:
      fp = _fingerprint(flags)
    except (TypeError, ValueError):
      continue
    log_m = re.search(r'LOG_DIR="([^"]+)"',
                      open(os.path.join(REPO, 'jobs', path),
                           encoding='utf-8', errors='replace').read())
    log_dir = ''
    if log_m:
      log_dir = log_m.group(1).replace('${SEED}', '0').split('logs/', 1)[-1]
      log_dir = log_dir.strip().strip('/')
    found[fp] = {'job_file': path, 'log_dir': log_dir, 'source': 'job'}
  for job_id, job_name, state in live._squeue_jobs():
    if not job_name.startswith('akt_rnd'):
      continue
    script = live._scontrol_command(job_id)
    flags = _flags_from_slurm(script)
    if not flags:
      continue
    try:
      fp = _fingerprint(flags)
    except (TypeError, ValueError):
      continue
    log_dir = live._log_dir_from_script(script, '0')
    found[fp] = {
        'job_id': job_id, 'job_name': job_name, 'log_dir': log_dir,
        'state': state, 'source': 'squeue',
    }
  return found


def _write_slurm(flags: dict) -> tuple[str, str, str]:
  ep = int(float(flags.get('isaacgym_episode_length', 50)))
  iters = max(1, int(NUM_STEPS / (NUM_ENVS * ep)))
  n_eval = max(20, iters // 10)
  flags = dict(flags)
  flags['isaacgym_fixed_target_xyz'] = _xyz_str(
      flags.get('isaacgym_fixed_target_xyz'))
  flags['isaacgym_table_spawn_object_xy'] = _xyz_str(
      flags.get('isaacgym_table_spawn_object_xy'), '0.0,0.0')
  xyz = flags['isaacgym_fixed_target_xyz']
  succ = str(flags.get('isaacgym_throw_success', 'in_bucket'))
  init = _init_token(flags)
  xy = _xy_token(xyz)
  stok = _succ_token(succ)
  hide = 'hide_' if _as_bool(flags.get('isaacgym_hide_table', False)) else ''
  job_name = f'akt_rnd_{init}_{hide}{xy}_{stok}_e{ep}_200m'
  slug = (
      f'ppo_rnd_allegro_kuka_throw_e1024_{init}_{hide}'
      f'{xy}_{stok}_ep{ep}_200m_seed0_4h')
  slug = re.sub(r'[^A-Za-z0-9_]+', '_', slug)
  log_dir = f'{slug}'
  script = os.path.join(REPO, 'jobs', f'job_{slug}.slurm')
  tyro = _tyro_flags(flags, ep)
  tyro_txt = ' \\\n  '.join(tyro)
  body = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm/{slug}_%j.log
#SBATCH --error=slurm/{slug}_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas

# Auto-spawned 200M RND twin of an NF run whose eval stayed >0.2.
#   sbatch jobs/job_{slug}.slurm
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym

CONDA_PREFIX="/n/fs/mislresearch/miniconda3/envs/sgcrl_isaacgym"
ISAAC_PY="${{CONDA_PREFIX}}/bin/python"
export PATH="${{CONDA_PREFIX}}/bin:${{PATH}}"
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="/n/fs/mislresearch/isaacgym/python:/n/fs/mislresearch/IsaacGymEnvs:/n/fs/mislresearch/mb6458-tmp/ppo-rnd-flax-deps:/n/fs/mislresearch/sgcrl:/n/fs/mislresearch/builderbench${{PYTHONPATH:+:${{PYTHONPATH}}}}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export PIP_CACHE_DIR=/n/fs/mislresearch/mb6458-tmp/pip-cache
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SEED=0
ENV_ID=allegro_kuka_throw
NUM_STEPS={NUM_STEPS}
NUM_ENVS={NUM_ENVS}
EP_LEN={ep}
NUM_EVAL_STEPS={n_eval}
ISAACGYM_PIPELINE=gpu

LOG_DIR="logs/{log_dir}/"
mkdir -p slurm "${{LOG_DIR}}"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID}}  host=$(hostname)"
echo "env_id=${{ENV_ID}} seed=${{SEED}} num_timesteps=${{NUM_STEPS}}"
echo "auto RND twin  T={ep}  target={xyz}  success={succ}"
echo "ppo_eval_interval~10 via num_eval_steps=${{NUM_EVAL_STEPS}}"
echo "LOG_DIR=${{LOG_DIR}}"
"${{ISAAC_PY}}" -c "import sys; print('sys.executable', sys.executable)"
nvidia-smi || true
"${{ISAAC_PY}}" -u -c "
import isaacgym
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import flax, jax
print('flax', flax.__version__)
print('JAX backend:', jax.default_backend(), 'devices:', jax.devices())
" || {{ echo 'ERROR: isaacgym/flax import failed'; exit 1; }}

"${{ISAAC_PY}}" -u baseline-agents/ppo-rnd.py \\
  --env-id="${{ENV_ID}}" \\
  --seed="${{SEED}}" \\
  --num-timesteps="${{NUM_STEPS}}" \\
  --num-envs="${{NUM_ENVS}}" \\
  --rollout-length="${{EP_LEN}}" \\
  --num-eval-steps="${{NUM_EVAL_STEPS}}" \\
  --isaacgym-episode-length="${{EP_LEN}}" \\
  --isaacgym-pipeline="${{ISAACGYM_PIPELINE}}" \\
  --wandb-dir="${{LOG_DIR}}" \\
  {tyro_txt}
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
'''
  with open(script, 'w', encoding='utf-8') as fh:
    fh.write(body)
  os.chmod(script, 0o755)
  return script, job_name, log_dir


def _sbatch(script: str) -> str:
  out = subprocess.check_output(['sbatch', script], cwd=REPO, text=True)
  match = re.search(r'Submitted batch job (\d+)', out)
  if not match:
    raise RuntimeError(f'sbatch failed: {out}')
  return match.group(1)


def _refresh_auto(state: dict) -> None:
  labels = []
  extras = []
  colors = list(live.RND_COLORS)
  for i, rec in enumerate(state.get('submitted', [])):
    token = rec.get('job_name', '')
    label = rec.get('label', token)
    color = rec.get('color') or colors[i % len(colors)]
    log_dir = rec.get('log_dir', '')
    if token:
      labels.append([token, label])
    if log_dir:
      extras.append([
          f'{label}  s0', log_dir, 'rnd', color,
          rec.get('job_id', f'auto-{i}'),
      ])
  _dump_json(AUTO_PATH, {'labels': labels, 'extras': extras})


def _cancel_if_high(state: dict) -> None:
  by_log = {rec.get('log_dir'): rec for rec in state.get('submitted', [])
            if rec.get('log_dir')}
  for job_id, job_name, state_name in live._squeue_jobs():
    if not (job_name.startswith('akt_rnd') or '_rnd_' in job_name):
      continue
    if 'allegro' not in job_name and not job_name.startswith('akt_rnd'):
      continue
    script = live._scontrol_command(job_id)
    log_dir = live._log_dir_from_script(script, live._seed_from_job_id(job_id))
    if not log_dir:
      continue
    pts = _rnd_series(log_dir)
    if not pts:
      continue
    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]
    high = _consec_above(ys, RND_HIGH, RND_HIGH_CONSEC)
    easy = ys[-1] >= RND_EASY and xs[-1] <= RND_EASY_STEPS
    stuck = xs[-1] >= RND_STUCK_STEPS and max(ys) < RND_STUCK_MAX
    if not (high or easy or stuck):
      continue
    subprocess.check_call(['scancel', str(job_id)])
    reason = (
        f'eval last={ys[-1]:.3f} max={max(ys):.3f} steps={xs[-1]} '
        f'high={high} easy={easy} stuck={stuck}')
    rec = by_log.get(log_dir)
    if rec is None:
      rec = {
          'job_id': job_id, 'job_name': job_name, 'log_dir': log_dir,
          'label': job_name,
      }
      state.setdefault('submitted', []).append(rec)
    rec['cancelled'] = True
    rec['cancel_reason'] = reason
    rec['job_id'] = job_id
    print(
        f'[nf→rnd] CANCEL {job_name} job={job_id} {reason} (logs kept)',
        flush=True)


def _maybe_spawn(state: dict) -> None:
  existing = _existing_rnd_fingerprints()
  known = {rec.get('fingerprint') for rec in state.get('submitted', [])}
  methods = live.discover_methods()
  for label, log_dir, kind, _color, job_id in methods:
    if kind != 'nf':
      continue
    if not log_dir.startswith('ppo_allegro_kuka'):
      continue
    flags = _run_config_flags(log_dir)
    if not flags:
      script = live._scontrol_command(str(job_id).split('_')[0])
      flags = _flags_from_slurm(script)
    if not flags:
      print(f'[nf→rnd] skip {job_id}: no flags', flush=True)
      continue
    ys = _nf_eval_ys(log_dir)
    last = ys[-3:] if ys else []
    triggered = _consec_above(ys, NF_THRESH, NF_CONSEC)
    print(
        f'[nf→rnd] {job_id} n_eval={len(ys)} last={ [round(y, 3) for y in last] } '
        f'trigger={triggered}',
        flush=True)
    if not triggered:
      continue
    fp = _fingerprint(flags)
    if fp in known:
      print(f'[nf→rnd] already spawned for {fp}', flush=True)
      continue
    if fp in existing:
      rec = existing[fp]
      print(f'[nf→rnd] matching 200M RND already exists: {rec}', flush=True)
      state.setdefault('submitted', []).append({
          'fingerprint': fp,
          'job_id': rec.get('job_id', ''),
          'job_name': rec.get('job_name', ''),
          'log_dir': rec.get('log_dir', ''),
          'label': _pretty_label(flags),
          'color': live.RND_COLORS[len(state.get('submitted', []))
                                   % len(live.RND_COLORS)],
          'from_nf': log_dir,
          'existing': True,
      })
      known.add(fp)
      continue
    script, job_name, rnd_log = _write_slurm(flags)
    new_id = _sbatch(script)
    rec = {
        'fingerprint': fp,
        'job_id': new_id,
        'job_name': job_name,
        'log_dir': rnd_log,
        'script': os.path.relpath(script, REPO),
        'label': _pretty_label(flags),
        'color': live.RND_COLORS[len(state.get('submitted', []))
                                 % len(live.RND_COLORS)],
        'from_nf': log_dir,
        'from_job': job_id,
    }
    state.setdefault('submitted', []).append(rec)
    known.add(fp)
    print(
        f'[nf→rnd] SUBMIT {new_id} {job_name} ← {job_id} {label}',
        flush=True)
    print(f'  script={script}', flush=True)
    print(f'  log_dir={rnd_log}', flush=True)


def tick(state: dict) -> None:
  _maybe_spawn(state)
  _cancel_if_high(state)
  _refresh_auto(state)
  _dump_json(STATE_PATH, state)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--interval', type=int, default=120)
  parser.add_argument('--watch-max-sec', type=int, default=0)
  parser.add_argument('--once', action='store_true')
  args = parser.parse_args()
  state = _load_json(STATE_PATH, {'submitted': []})
  print(
      f'[nf→rnd] every {args.interval}s '
      f'(max {args.watch_max_sec or "∞"}s) '
      f'NF eval>{NF_THRESH} x{NF_CONSEC} → RND 200M; '
      f'cancel RND eval>{RND_HIGH} x{RND_HIGH_CONSEC} or '
      f'>{RND_EASY} by {RND_EASY_STEPS/1e6:.0f}M or '
      f'stuck <{RND_STUCK_MAX} after {RND_STUCK_STEPS/1e6:.0f}M',
      flush=True)
  t0 = time.time()
  while True:
    try:
      tick(state)
    except Exception as exc:  # noqa: BLE001
      print(f'[nf→rnd] error: {exc}', flush=True)
    if args.once:
      return
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[nf→rnd] watch-max-sec reached; exiting', flush=True)
      return
    time.sleep(args.interval)


if __name__ == '__main__':
  main()
