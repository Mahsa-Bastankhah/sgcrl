#!/usr/bin/env python3
"""Plot mean eval success_1000 vs iteration, one line per config (averaged over seeds).

Looks for Acme-style eval logs::

    <log_root>/<config_name>/<run_name>/logs/eval/logs.csv

where ``run_name`` is typically ``ppo_point_FourRooms_<seed>``.  By default only
paths containing ``point_FourRooms`` are used (so e.g. ``ppo_point_Impossible_*``
is excluded).  Override with ``--run_name_substring ''`` to plot every env.

With ``--all_logs``, all runs under the same log **folder name** (config
directory) are averaged as seeds of one line.

With the default **Slurm whitelist**, each line is one **unique** ``PPO knobs:``
fingerprint parsed from the Slurm header (clip / min std / discounts / LR
anneal / …); seeds that share identical printed knobs are averaged together.

By default (**Slurm whitelist**), only runs that still have a matching file under
``--slurm_whitelist_glob`` (default ``slurm/*.out``) *and* a present
``logs/.../eval/logs.csv`` are plotted—deleted log dirs are dropped automatically.
Use ``--all_logs`` to scan every eval CSV under ``--log_root`` again.

Optionally (with ``--all_logs``), parses Slurm ``[Eval]`` lines for stdout-only
curves via ``--slurm_glob``.

Use ``--watch`` to refresh the PNG on an interval; each cycle re-glob Slurm files
and re-read eval CSVs, so **new jobs and new PPO-knob fingerprints** show up
without restarting the process (good under ``tmux``/``screen``).

Examples::

  python scripts/plot_success_1000_by_config.py \\
      --log_root logs/ppo_ablations \\
      --output plots/success_1000_mean.png

  python scripts/plot_success_1000_by_config.py \\
      --log_root logs/ppo_ablations \\
      --slurm_glob 'slurm/ppo_c64e05m01_*.out' \\
      --output plots/success_1000_mean.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import re
import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np

try:
  import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
  raise SystemExit('matplotlib is required: pip install matplotlib') from e


def _discover_eval_csvs(log_root: str, run_substring: str) -> List[str]:
  pattern = os.path.join(log_root, '*', '*', 'logs', 'eval', 'logs.csv')
  paths = sorted(glob.glob(pattern))
  if not run_substring:
    return paths
  return [p for p in paths if run_substring in p]


def _config_key_from_eval_csv(csv_path: str) -> str:
  """.../<config>/<run>/logs/eval/logs.csv -> config folder name."""
  p = os.path.abspath(csv_path)
  parts = p.split(os.sep)
  try:
    i_eval = parts.index('eval')
  except ValueError:
    return os.path.basename(os.path.dirname(os.path.dirname(p)))
  # .../<run>/logs/eval/logs.csv  -> run at i_eval-3, config at i_eval-4
  if i_eval < 4:
    return 'unknown'
  return parts[i_eval - 3]


def _read_csv_rows(path: str) -> csv.DictReader:
  """Read eval CSV; strip NULs that occasionally appear in partially-written files."""
  with open(path, 'rb') as fh:
    raw = fh.read().replace(b'\x00', b'')
  text = raw.decode('utf-8', errors='replace')
  return csv.DictReader(io.StringIO(text))


def _parse_csv_float(cell: Optional[str]) -> Optional[float]:
  """Parse a CSV cell to float; None / empty / non-numeric -> None."""
  if cell is None:
    return None
  s = str(cell).strip()
  if not s:
    return None
  try:
    return float(s)
  except ValueError:
    return None


def _load_eval_csv(path: str, x_axis: str) -> Tuple[Dict[int, float], str]:
  """Return mapping x -> success_1000 and run id (folder name).

  Skips malformed or in-flight rows (e.g. truncated lines where DictReader
  maps trailing columns to ``None``).
  """
  run_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
  x_key = 'iteration' if x_axis == 'iteration' else 'learner_steps'
  series: Dict[int, float] = {}
  reader = _read_csv_rows(path)
  for row in reader:
    raw_x = row.get(x_key)
    if raw_x is None:
      continue
    try:
      x = int(float(str(raw_x).strip()))
    except (TypeError, ValueError):
      continue
    y = _parse_csv_float(row.get('success_1000'))
    if y is None:
      continue
    series[x] = y
  return series, run_dir


def _aggregate_config(
    series_list: List[Dict[int, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Mean and std of success_1000 across seeds at each iteration (any seed with data)."""
  all_iters: set = set()
  for s in series_list:
    all_iters.update(s.keys())
  iters_sorted = sorted(all_iters)
  means = []
  stds = []
  for it in iters_sorted:
    vals = [float(s[it]) for s in series_list if it in s]
    if not vals:
      continue
    means.append(float(np.mean(vals)))
    stds.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0)
  return np.asarray(iters_sorted, dtype=np.int32), np.asarray(means), np.asarray(stds)


# --- Slurm: whitelist existing .out -> eval CSV on disk -----------------------------

_VARIANT_RE = re.compile(r'variant=(\S+)')
_SEED_RE = re.compile(r'\bseed=(\d+)\b')
_ENV_RE = re.compile(r'env=(\S+)')
_LOGDIR_RE = re.compile(r'log_dir=(\S+)')
_EVAL_LOG_RE = re.compile(r'Logging to\s+(\S+logs/eval/logs\.csv)')
_PPO_KNOBS_LINE_RE = re.compile(r'PPO knobs:\s*([^\n]+)')
# Canonical PPO hyperparameter keys (order defines fingerprint + legend).
_PPO_KNOB_KEYS: Tuple[str, ...] = (
    'rollout_length',
    'crl_steps_per_iter',
    'num_envs',
    'clip_coef',
    'actor_min_std',
    'ent_coef',
    'discount_crl',
    'discount_ppo',
    'norm_reward',
    'repr_norm',
    'ppo_anneal_lr',
)
_ITER_RE = re.compile(r'Iteration\s*=\s*(\d+)')
_LS_RE = re.compile(r'Learner Steps\s*=\s*(\d+)')
_S1000_RE = re.compile(r'Success 1000\s*=\s*([0-9.eE+-]+)')


def _parse_ppo_knobs_line(text: str) -> Dict[str, str]:
  """Parse ``PPO knobs: k=v, ...`` from Slurm / training header into a flat dict."""
  m = _PPO_KNOBS_LINE_RE.search(text)
  if not m:
    return {}
  blob = m.group(1).strip()
  out: Dict[str, str] = {}
  for chunk in blob.split(','):
    chunk = chunk.strip()
    if not chunk or '=' not in chunk:
      continue
    k, _, v = chunk.partition('=')
    k, v = k.strip(), v.strip()
    if k:
      out[k] = v
  return out


def _normalize_knobs_for_grouping(knobs: Dict[str, str]) -> Dict[str, str]:
  """Older Slurm lines omit trailing ``ppo_anneal_lr=``; align with current defaults."""
  k = dict(knobs)
  if 'ppo_anneal_lr' not in k:
    k['ppo_anneal_lr'] = 'True'
  return k


def _ppo_knobs_fingerprint(knobs: Dict[str, str]) -> str:
  """Stable string so runs with identical printed PPO knobs merge across seeds."""
  parts = [f'{k}={knobs.get(k, "__missing__")}' for k in _PPO_KNOB_KEYS]
  return ' | '.join(parts)


def _ppo_knobs_legend_label(knobs: Dict[str, str]) -> str:
  """One-line legend text."""
  if not knobs:
    return '(no PPO knobs line in Slurm)'

  def g(key: str) -> str:
    return knobs.get(key, '?')

  return (
      f'clip={g("clip_coef")} minstd={g("actor_min_std")} ent={g("ent_coef")} '
      f'γppo={g("discount_ppo")} γcrl={g("discount_crl")} '
      f'anneal={g("ppo_anneal_lr")} T={g("rollout_length")} crl={g("crl_steps_per_iter")}')


def _resolve_eval_csv_via_slurm_out(
    slurm_path: str,
    repo_root: str,
    run_filter: str,
) -> Optional[Tuple[str, Dict[str, str]]]:
  """Map one Slurm stdout file to (abs_eval_csv, ppo_knobs_dict) or None.

  Prefers ``log_dir=`` + ``env=`` + ``seed=``; falls back to ``Logging to
  .../logs/eval/logs.csv`` or ``wrote run config: .../run_config.json``.
  ``ppo_knobs_dict`` comes from the printed ``PPO knobs:`` line (may be empty).
  Returns None if the eval CSV is missing (deleted run).
  """
  buf: List[str] = []
  with open(slurm_path, encoding='utf-8', errors='replace') as fh:
    for _ in range(320):
      line = fh.readline()
      if not line:
        break
      buf.append(line)
  text = ''.join(buf)
  knobs = _parse_ppo_knobs_line(text)

  env_m = _ENV_RE.search(text)
  seed_m = _SEED_RE.search(text)
  log_m = _LOGDIR_RE.search(text)
  env_name = env_m.group(1) if env_m else None
  seed = int(seed_m.group(1)) if seed_m else None

  if run_filter:
    if not env_name or run_filter not in env_name:
      return None

  eval_csv: Optional[str] = None
  if log_m and env_name is not None and seed is not None:
    ld = log_m.group(1).strip().rstrip('/')
    ld_abs = ld if os.path.isabs(ld) else os.path.normpath(os.path.join(repo_root, ld))
    run_name = f'ppo_{env_name}_{seed}'
    cand = os.path.join(ld_abs, run_name, 'logs', 'eval', 'logs.csv')
    if os.path.isfile(cand):
      eval_csv = cand

  if eval_csv is None:
    el_m = _EVAL_LOG_RE.search(text)
    if el_m:
      rel = el_m.group(1).strip()
      eval_csv = (
          rel if os.path.isabs(rel) else os.path.normpath(os.path.join(repo_root, rel)))
  if eval_csv is None or not os.path.isfile(eval_csv):
    rc_m = re.search(r'wrote run config:\s*(\S+)', text)
    if rc_m:
      rc = rc_m.group(1).strip()
      if not os.path.isabs(rc):
        rc = os.path.normpath(os.path.join(repo_root, rc))
      run_dir = os.path.dirname(rc)
      cand = os.path.join(run_dir, 'logs', 'eval', 'logs.csv')
      if os.path.isfile(cand):
        eval_csv = cand

  if not eval_csv or not os.path.isfile(eval_csv):
    return None
  return eval_csv, knobs


def _parse_slurm_eval(
    path: str, x_axis: str, require_run_substring: str,
) -> Optional[Tuple[str, Dict[int, float]]]:
  """Return (variant, x->success_1000) or None if not parseable."""
  variant: Optional[str] = None
  env_name: Optional[str] = None
  pts: Dict[int, float] = {}
  with open(path, encoding='utf-8', errors='replace') as fh:
    for line in fh:
      if env_name is None:
        me = _ENV_RE.search(line)
        if me:
          env_name = me.group(1)
          if require_run_substring and require_run_substring not in env_name:
            return None
      if variant is None:
        m = _VARIANT_RE.search(line)
        if m:
          variant = m.group(1)
      if '[Eval]' not in line or 'Success 1000' not in line:
        continue
      ms = _S1000_RE.search(line)
      if not ms:
        continue
      if x_axis == 'learner_steps':
        mx = _LS_RE.search(line)
      else:
        mx = _ITER_RE.search(line)
      if not mx:
        continue
      pts[int(mx.group(1))] = float(ms.group(1))
  if variant is None or not pts:
    return None
  if require_run_substring:
    if env_name is None or require_run_substring not in env_name:
      return None
  return variant, pts


def _discover_slurm_series(
    glob_pat: str,
    x_axis: str,
    require_run_substring: str,
) -> DefaultDict[str, List[Dict[int, float]]]:
  out: DefaultDict[str, List[Dict[int, float]]] = defaultdict(list)
  for path in sorted(glob.glob(glob_pat)):
    parsed = _parse_slurm_eval(path, x_axis, require_run_substring)
    if parsed is None:
      continue
    variant, series = parsed
    out[variant].append(series)
  return out


def _plot(
    config_to_series: Dict[str, List[Dict[int, float]]],
    output: str,
    x_key: str,
    title: Optional[str],
    min_runs: int,
    show_std: bool,
    figsize: Tuple[float, float],
    legend_labels: Optional[Dict[str, str]] = None,
) -> None:
  fig, ax = plt.subplots(figsize=figsize)
  cmap = plt.get_cmap('tab10')
  labels = sorted(config_to_series.keys())
  for idx, cfg in enumerate(labels):
    series_list = config_to_series[cfg]
    if len(series_list) < min_runs:
      continue
    xs, mean_y, std_y = _aggregate_config(series_list)
    if xs.size == 0:
      continue
    color = cmap(idx % 10)
    base = (legend_labels or {}).get(cfg, cfg)
    lab = f'{base} (n={len(series_list)})'
    ax.plot(xs, mean_y, label=lab, color=color, linewidth=1.8)
    if show_std and np.any(std_y > 0):
      ax.fill_between(xs, mean_y - std_y, mean_y + std_y, color=color, alpha=0.18)
  ax.set_xlabel(x_key.replace('_', ' ').title())
  ax.set_ylabel('success_1000 (mean over seeds)')
  ax.set_ylim(0.0, 1.02)
  ax.grid(True, alpha=0.35)
  ax.legend(loc='best', fontsize=8)
  ax.set_title(title or 'Eval success_1000 (EMA) averaged over seeds')
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
  fig.savefig(output, dpi=160)
  plt.close(fig)
  print(f'[plot] wrote {output}')


def _run_one_cycle(args: argparse.Namespace) -> bool:
  """Build merged series, write PNG. Returns False if nothing to plot (watch: retry)."""
  log_roots = list(args.log_root)
  if not log_roots and not (args.slurm_glob or '').strip():
    log_roots = ['logs/ppo_ablations']

  run_filter = (args.run_name_substring or '').strip()
  merged: DefaultDict[str, List[Dict[int, float]]] = defaultdict(list)
  legend_labels: Dict[str, str] = {}
  repo_root = os.path.abspath(os.path.expanduser(args.repo_root))

  use_whitelist = (
      not bool(args.all_logs)
      and bool((args.slurm_whitelist_glob or '').strip()))

  if use_whitelist:
    wg = os.path.join(repo_root, args.slurm_whitelist_glob)
    slurm_files = sorted(glob.glob(wg))
    if not slurm_files:
      if getattr(args, 'watch', False):
        print(f'[plot] no Slurm files match {wg!r} yet; will retry.')
        return False
      raise SystemExit(
          f'No Slurm files match {wg!r}. Restore logs with --all_logs or fix '
          '--slurm_whitelist_glob / --repo_root.')
    seen_csv: set = set()
    skipped = 0
    for sf in slurm_files:
      hit = _resolve_eval_csv_via_slurm_out(sf, repo_root, run_filter)
      if hit is None:
        skipped += 1
        continue
      eval_csv, knobs = hit
      if eval_csv in seen_csv:
        continue
      seen_csv.add(eval_csv)
      series, _run = _load_eval_csv(eval_csv, args.x_axis)
      if not series:
        print(f'[plot] skip (no valid eval rows): {eval_csv}')
        continue
      if knobs:
        knobs_n = _normalize_knobs_for_grouping(knobs)
        fp = _ppo_knobs_fingerprint(knobs_n)
        leg = _ppo_knobs_legend_label(knobs_n)
      else:
        fp = f'__no_knobs__|{_config_key_from_eval_csv(eval_csv)}'
        leg = fp
      legend_labels.setdefault(fp, leg)
      merged[fp].append(series)
    print(f'[plot] slurm whitelist: {len(slurm_files)} .out file(s), '
          f'{len(seen_csv)} unique eval CSV(s), skipped {skipped} '
          f'(no CSV / wrong env / etc.)')
    print(f'[plot] {len(merged)} unique PPO-knob fingerprint(s) from Slurm headers')
  else:
    for root in log_roots:
      root = os.path.expanduser(root)
      if not os.path.isdir(root):
        print(f'[plot] skip missing log_root: {root!r}')
        continue
      for csv_path in _discover_eval_csvs(root, run_filter):
        cfg = _config_key_from_eval_csv(csv_path)
        series, _run = _load_eval_csv(csv_path, args.x_axis)
        if not series:
          print(f'[plot] skip (no valid eval rows): {csv_path}')
          continue
        merged[cfg].append(series)

    if args.slurm_glob:
      for cfg, lst in _discover_slurm_series(
          args.slurm_glob, args.x_axis, run_filter).items():
        merged[cfg].extend(lst)

  if not merged:
    if getattr(args, 'watch', False):
      print('[plot] no eval series this cycle; will retry.')
      return False
    raise SystemExit('No eval series found. Check --log_root and/or --slurm_glob.')

  x_label = args.x_axis
  _plot(
      dict(merged),
      args.output,
      x_label,
      args.title or None,
      min_runs=int(args.min_runs),
      show_std=bool(args.std_band),
      figsize=(float(args.fig_width), float(args.fig_height)),
      legend_labels=legend_labels if legend_labels else None,
  )

  for cfg, lst in sorted(merged.items()):
    print(f'  {cfg}: {len(lst)} run(s)')
  return True


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--log_root',
      action='append',
      default=[],
      help='Root such as logs/ppo_ablations (repeatable). '
           'If omitted entirely, defaults to logs/ppo_ablations when '
           '--slurm_glob is not the only data source.')
  ap.add_argument(
      '--slurm_glob',
      default='',
      help="Optional: merge eval curves parsed from Slurm stdout (not CSV). "
           "Only used with --all_logs.")
  ap.add_argument(
      '--repo_root',
      default='.',
      help='Repo root for resolving relative paths in Slurm (default: cwd).')
  ap.add_argument(
      '--slurm_whitelist_glob',
      default='slurm/*.out',
      help='Only plot runs that still have a matching Slurm .out and an eval '
           'CSV on disk. Set to empty and pass --all_logs to scan all logs again.')
  ap.add_argument(
      '--all_logs',
      action='store_true',
      help='Ignore --slurm_whitelist_glob; scan every eval CSV under --log_root '
           '(old behavior).')
  ap.add_argument('--output', required=True, help='Output PNG path.')
  ap.add_argument(
      '--x_axis',
      choices=('iteration', 'learner_steps'),
      default='iteration',
      help='X axis from eval CSV (both are usually aligned for PPO).')
  ap.add_argument('--title', default='', help='Figure title (optional).')
  ap.add_argument(
      '--min_runs',
      type=int,
      default=1,
      help='Minimum number of seeds (CSV runs or slurm files) per config to plot.')
  ap.add_argument('--std_band', action='store_true',
                  help='Shade mean ± std across seeds.')
  ap.add_argument('--fig_width', type=float, default=10.0)
  ap.add_argument('--fig_height', type=float, default=5.5)
  ap.add_argument(
      '--run_name_substring',
      default='point_FourRooms',
      help='Only include eval CSV paths whose run directory contains this '
           'substring (default: point_FourRooms).  Use empty string to include '
           'all runs.  Slurm logs are skipped unless their env= line matches too.')
  ap.add_argument(
      '--watch',
      action='store_true',
      help='Loop forever: re-glob Slurm / re-read eval CSVs each cycle so new '
           'configs and longer curves appear in the same --output PNG.')
  ap.add_argument(
      '--watch_interval',
      type=float,
      default=90.0,
      help='Seconds between refresh cycles when --watch is set (default: 90).')
  args = ap.parse_args()

  cycle = 0
  try:
    while True:
      cycle += 1
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[plot] === cycle {cycle} @ {ts} ===')
      _run_one_cycle(args)
      if not args.watch:
        break
      delay = max(5.0, float(args.watch_interval))
      print(f'[plot] next refresh in {delay:.0f}s  (Ctrl+C to stop)')
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[plot] watch stopped by user.')


if __name__ == '__main__':
  main()
