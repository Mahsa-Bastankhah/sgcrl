#!/usr/bin/env python3
"""Incrementally eval BuilderBench checkpoints and refresh the success plot.

Polls ``logs/<log_root>/ppo_<env>_<seed>/checkpoints/ckpt_iter_*.pkl``.
For each new checkpoint, runs deterministic eval episodes, appends the row
to a per-seed CSV, then rewrites an aggregated mean ± SE plot across seeds
(shaded band).

State: ``figs/builderbench/checkpoint_eval/.watch_eval_state.json``

Examples::

  python scripts/watch_builderbench_checkpoint_eval.py \\
      --log_dir logs/ppo_builderbench_creative3_task1_e1024_pd_nf_tau05 \\
      --env builderbench_creative_3_task1 \\
      --plot_tag creative3_task1_pd_nf_tau05 \\
      --watch_interval 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('BUILDERBENCH_MJX_IMPL', 'jax')
os.environ.setdefault(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')

import importlib.util
import sgcrl_jax_acme_compat  # noqa: F401

_eval_path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_checkpoint_eval.py')
_eval_spec = importlib.util.spec_from_file_location('bb_ckpt_eval', _eval_path)
_bb_eval = importlib.util.module_from_spec(_eval_spec)
sys.modules['bb_ckpt_eval'] = _bb_eval
_eval_spec.loader.exec_module(_bb_eval)
discover_run_dirs = _bb_eval.discover_run_dirs
eval_run = _bb_eval.eval_run
list_checkpoint_files = _bb_eval.list_checkpoint_files
plot_multi_seed_results = _bb_eval.plot_multi_seed_results
read_csv_results = _bb_eval.read_csv_results
seed_csv_path = _bb_eval.seed_csv_path


def _state_key(run_dir: str, label: str) -> str:
  return f'{run_dir}|{label}'


def _load_state(path: str) -> dict:
  if not os.path.isfile(path):
    return {}
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def _save_state(path: str, state: dict) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)


@dataclass
class _WatchSpec:
  log_dir: str
  env: str
  plot_tag: str
  seeds: Optional[set[int]]
  plot_path: str


def _parse_seeds(s: str | None) -> set[int] | None:
  if not s:
    return None
  return {int(x.strip()) for x in s.split(',') if x.strip()}


def _resolve_spec(args, repo: str) -> _WatchSpec:
  log_dir = args.log_dir
  if not os.path.isabs(log_dir):
    log_dir = os.path.join(repo, log_dir)
  seeds = _parse_seeds(args.seeds)
  plot_tag = args.plot_tag or os.path.basename(os.path.normpath(log_dir))
  if not args.env:
    raise SystemExit('--env is required')
  if args.output:
    plot_path = (args.output if os.path.isabs(args.output)
                 else os.path.join(repo, args.output))
  else:
    plot_path = os.path.join(
        repo, 'figs', 'builderbench', f'{plot_tag}_checkpoint_success.png')
  return _WatchSpec(
      log_dir=log_dir, env=args.env, plot_tag=plot_tag, seeds=seeds,
      plot_path=plot_path)


def _csv_for_seed(spec: _WatchSpec, seed: int, args) -> str:
  """Per-seed CSV. Shared --csv_output is only used when watching one seed."""
  if args.csv_output and spec.seeds is not None and len(spec.seeds) == 1:
    return (args.csv_output if os.path.isabs(args.csv_output)
            else os.path.join(_REPO, args.csv_output))
  return seed_csv_path(spec.plot_tag, seed)


def _write_aggregate_plot(spec: _WatchSpec, args) -> None:
  results_by_seed: Dict[int, List] = {}
  for seed, run_dir, _ckpt_dir in discover_run_dirs(
      spec.log_dir, spec.env, spec.seeds):
    csv_path = _csv_for_seed(spec, seed, args)
    rows = read_csv_results(csv_path)
    if rows:
      results_by_seed[seed] = rows
  if not results_by_seed:
    return
  seed_list = sorted(results_by_seed)
  title = (
      f'BuilderBench eval success — {spec.env} '
      f'({spec.plot_tag}, seeds {seed_list})')
  n = plot_multi_seed_results(
      results_by_seed,
      title=title,
      output_path=spec.plot_path,
      x_axis=args.x_axis,
  )
  if n:
    print(f'[watch_bb_eval] wrote aggregate plot (n={n}): {spec.plot_path}',
          flush=True)


def _scan_and_eval(spec: _WatchSpec, args, state: dict, state_path: str) -> int:
  n_evaluated = 0
  if not os.path.isdir(spec.log_dir):
    print(f'[watch_bb_eval] waiting for log dir: {spec.log_dir}', flush=True)
    return 0

  for seed, run_dir, ckpt_dir in discover_run_dirs(
      spec.log_dir, spec.env, spec.seeds):
    if not os.path.isdir(ckpt_dir):
      print(f'[watch_bb_eval] seed={seed}: no checkpoints/ yet ({run_dir})',
            flush=True)
      continue

    csv_path = _csv_for_seed(spec, seed, args)
    # Per-seed plot is skipped; aggregate plot is written after the loop.
    per_seed_plot = os.path.join(
        _REPO, 'figs', 'builderbench',
        f'{spec.plot_tag}_seed{seed}_checkpoint_success.png')

    pending = []
    for label, path, mtime in list_checkpoint_files(ckpt_dir):
      key = _state_key(run_dir, label)
      prev = state.get(key)
      if not args.force and prev is not None and float(prev) >= mtime:
        continue
      pending.append((label, path, mtime))

    if not pending and os.path.isfile(csv_path):
      continue

    if not pending:
      print(f'[watch_bb_eval] seed={seed}: no new checkpoints', flush=True)
      continue

    print(
        f'[watch_bb_eval] seed={seed}: {len(pending)} new checkpoint(s) '
        f'({run_dir})',
        flush=True,
    )
    eval_run(
        run_dir,
        spec.env,
        num_eval_episodes=args.num_eval_episodes,
        eval_seed=args.eval_seed,
        network_seed=args.seed,
        incremental=True,
        csv_path=csv_path,
        plot_path=per_seed_plot,
        plot_tag=spec.plot_tag,
        x_axis=args.x_axis,
        only_labels=[x[0] for x in pending],
    )

    for label, path, mtime in pending:
      state[_state_key(run_dir, label)] = mtime
      n_evaluated += 1

    for row in read_csv_results(csv_path):
      if os.path.isfile(row.path):
        state[_state_key(run_dir, row.label)] = os.path.getmtime(row.path)

  _write_aggregate_plot(spec, args)
  _save_state(state_path, state)
  return n_evaluated


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--repo_root', default='.')
  ap.add_argument('--log_dir', required=True,
                  help='Training log root (e.g. logs/ppo_builderbench_.../).')
  ap.add_argument('--env', required=True, help='sgcrl env name.')
  ap.add_argument('--plot_tag', default=None,
                  help='Plot filename stem (default: log_dir basename).')
  ap.add_argument('--output', default=None,
                  help='Aggregated mean±SE plot PNG path.')
  ap.add_argument('--csv_output', default=None,
                  help='Only used when watching a single --seeds value; '
                       'otherwise per-seed CSVs are used.')
  ap.add_argument('--seeds', default=None,
                  help='Comma-separated seeds (default: all found).')
  ap.add_argument('--num_eval_episodes', type=int, default=20)
  ap.add_argument('--eval_seed', type=int, default=0)
  ap.add_argument('--seed', type=int, default=0,
                  help='Network-init probe seed.')
  ap.add_argument('--x_axis', choices=('global_step', 'iteration'),
                  default='global_step')
  ap.add_argument(
      '--state_file',
      default='figs/builderbench/checkpoint_eval/.watch_eval_state.json')
  ap.add_argument('--once', action='store_true')
  ap.add_argument('--force', action='store_true',
                  help='Re-eval checkpoints even if state says done.')
  ap.add_argument('--watch_interval', type=float, default=300.0)
  args = ap.parse_args()

  repo = os.path.abspath(os.path.expanduser(args.repo_root))
  spec = _resolve_spec(args, repo)
  state_path = (
      args.state_file if os.path.isabs(args.state_file)
      else os.path.join(repo, args.state_file))

  print(f'[watch_bb_eval] log_dir={spec.log_dir}  env={spec.env}  '
        f'plot_tag={spec.plot_tag}  episodes={args.num_eval_episodes}',
        flush=True)
  print(f'[watch_bb_eval] aggregate plot → {spec.plot_path}', flush=True)

  try:
    while True:
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[watch_bb_eval] === scan @ {ts} ===', flush=True)
      state = _load_state(state_path)
      n = _scan_and_eval(spec, args, state, state_path)
      print(f'[watch_bb_eval] evaluated {n} checkpoint(s)', flush=True)
      if args.once:
        break
      delay = max(60.0, float(args.watch_interval))
      print(f'[watch_bb_eval] next scan in {delay:.0f}s', flush=True)
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[watch_bb_eval] stopped.', flush=True)


if __name__ == '__main__':
  main()
