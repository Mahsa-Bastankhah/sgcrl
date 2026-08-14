#!/usr/bin/env python3
"""Creative7-task2 CRL / NF / TD3: train+eval success split by worked vs failed.

Discovers every historical ``logs/ppo_builderbench_creative7_task2_*`` run for
CRL, NF, and TD3 density estimators (including currently RUNNING squeue jobs),
groups directories that are the exact same config (seed / eval-interval suffixes
stripped), and classifies each config as:

  worked  — peak(train success_1000, eval success) >= --threshold (default 0.1)
  failed  — never reached that bar

Successful configs stay on the worked figures even when they look very
different from the failed set. Multiple seeds of one config share one
legend entry: mean ± stderr (shaded). Per-seed peaks are reported in the
legend; faint unlabeled seed traces stay on the axes so variance is visible.

Writes 6 dual-panel figures (train | eval), 2 per density estimator:

  figs/builderbench/active_train_eval/creative7_task2_worked_vs_failed/
    creative7_task2_{crl|nf|td3}_{worked|failed}_train_eval.png
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

TASK_PREFIX = 'ppo_builderbench_creative7_task2_'
GROUP = 'creative7_task2'
FAMILIES = ('crl', 'nf', 'td3')
OUT_DIR = os.path.join(base.FIGS_DIR, 'creative7_task2_worked_vs_failed')
DEFAULT_THRESHOLD = 0.1
# job scripts sometimes write seed1 to a sibling dir with these suffixes
CONFIG_STRIP_RE = re.compile(r'(?:_seed\d+|_eval\d+)+$')


def _discover_c7t2_runs() -> list[str]:
  names: set[str] = set()
  if os.path.isdir(base.LOG_ROOT):
    for n in os.listdir(base.LOG_ROOT):
      if n.startswith(TASK_PREFIX):
        names.add(n)
  names |= {
      n for n in base._log_dirs_from_slurm_files(base.SLURM_DIR)
      if n.startswith(TASK_PREFIX)
  }
  names |= {
      n for n in base._active_log_dirs(base.SLURM_DIR)
      if n.startswith(TASK_PREFIX)
  }
  out = []
  for name in sorted(names):
    fam = base._method_family(name)
    if fam not in FAMILIES:
      continue
    has_csv = base._has_csv(base.LOG_ROOT, name, 'learner', min_size=200)
    has_slurm = bool(base._read_train_from_slurm(base.SLURM_DIR, name))
    if not has_csv and not has_slurm:
      print(f'  skip (no train data yet): {name}')
      continue
    out.append(name)
  return out


def _config_key(log_dir_name: str) -> str:
  """Same training config; ignore seed / eval-interval dir suffixes."""
  return CONFIG_STRIP_RE.sub('', log_dir_name)


def _legend_label(log_dir_name: str) -> str:
  """Distinctive short legend from dir-name knobs (avoids CRL label collisions)."""
  low = log_dir_name.lower()
  fam = base._method_family(log_dir_name)
  bits: list[str] = []

  if fam == 'crl':
    bits.append('CRL')
  elif fam == 'td3':
    bits.append('TD3')
  elif '_nf_compact_small_' in low:
    bits.append('NF compact-small')
  elif '_nf_compact_' in low:
    bits.append('NF compact')
  else:
    bits.append('NF')

  mt = re.search(r'tau(\d+)', low)
  if mt:
    raw = mt.group(1)
    tau = float(raw[0] + '.' + raw[1:]) if len(raw) > 1 else float(raw)
    if abs(tau - 0.5) > 1e-9:
      bits.append(f'τ={tau:g}')

  me = re.search(r'task\d+_e(\d+)_', low)
  if me and me.group(1) != '1024':
    bits.append(f'e{me.group(1)}')

  if 'permute_rand' in low:
    bits.append('permute+rand')
  elif 'nopermute' in low and 'fixedx01' in low:
    bits.append('noperm+fixx')
  elif 'nopermute' in low:
    bits.append('noperm')
  elif 'norand' in low:
    bits.append('norand')

  if 'catwp' in low:
    bits.append('catwp')
  elif 'catselect' in low:
    bits.append('catselect')

  if 'extrew10' in low:
    bits.append('extrew10')
  elif 'extrew1' in low:
    bits.append('extrew1')

  if 'minstd1e5' in low:
    bits.append('minstd1e-5')
  elif 'minstd1e4' in low:
    bits.append('minstd1e-4')

  if 'entanneal' in low:
    bits.append('ent-anneal')
  if 'noanneallr' in low:
    bits.append('no-anneal-lr')

  if 'normobs' in low and 'nonormobs' not in low:
    bits.append('normobs')

  m = re.search(r'(?:^|_)crl(\d+)(?:_|$)', low)
  if m and fam != 'crl':
    bits.append(f'crl-steps={m.group(1)}')
  elif m and fam == 'crl' and int(m.group(1)) != 1:
    bits.append(f'steps/iter={m.group(1)}')

  if 'h4x256' in low:
    bits.append('h4×256')
  if 'sparseafter' in low:
    bits.append('sparse-after')
  if 'freezeafter' in low or 'freezerepr' in low:
    bits.append('freeze-repr')
  if 'sfpert' in low:
    meps = re.search(r'sfpert_eps(\d+)', low)
    if meps:
      digits = meps.group(1)
      bits.append(f'SF-pert ε={float(digits) / (10 ** (len(digits) - 1)):g}')
    else:
      bits.append('SF-pert')

  if 'ep50' in low:
    bits.append('ep50')
  elif 'ep60' in low:
    bits.append('ep60')
  elif 'ep70' in low:
    bits.append('ep70')

  if 'tol0013' in low:
    bits.append('tol=0.013')
  if 'actorreset' in low:
    bits.append('actor-reset')

  return ' · '.join(bits)


def _peak_from_mean(mean: list[float]) -> float:
  return max(mean) if mean else 0.0


def _interp_at(pts: list[tuple[int, float]], xq: list[int]) -> list[float | None]:
  """Linear interp onto ``xq``; None outside this seed's x range (no extrapolate)."""
  xs = [p[0] for p in pts]
  ys = [p[1] for p in pts]
  out: list[float | None] = []
  j = 0
  n = len(xs)
  for x in xq:
    if x < xs[0] or x > xs[-1]:
      out.append(None)
      continue
    while j + 1 < n and xs[j + 1] < x:
      j += 1
    if xs[j] == x or j + 1 >= n:
      out.append(ys[j])
      continue
    x0, x1 = xs[j], xs[j + 1]
    y0, y1 = ys[j], ys[j + 1]
    t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    out.append(y0 + t * (y1 - y0))
  return out


def _aggregate_aligned(
    seed_series: list[list[tuple[int, float]]],
    max_pts: int = 800,
):
  """Mean ± stderr on a shared x-grid so shade is visible across seeds."""
  nonempty = [sorted(pts) for pts in seed_series if pts]
  n_seeds = len(nonempty)
  if n_seeds == 0:
    return [], [], [], 0
  if n_seeds == 1:
    xs = [p[0] for p in nonempty[0]]
    mean = [p[1] for p in nonempty[0]]
    se = [0.0] * len(xs)
    return xs, mean, se, 1

  all_x = sorted({int(x) for pts in nonempty for x, _ in pts})
  if len(all_x) > max_pts:
    step = max(1, len(all_x) // max_pts)
    grid = all_x[::step]
    if grid[-1] != all_x[-1]:
      grid.append(all_x[-1])
  else:
    grid = all_x

  xs, mean, se = [], [], []
  for x in grid:
    ys = []
    for pts in nonempty:
      val = _interp_at(pts, [x])[0]
      if val is not None:
        ys.append(val)
    if not ys:
      continue
    m = sum(ys) / len(ys)
    if len(ys) > 1:
      var = sum((y - m) ** 2 for y in ys) / (len(ys) - 1)
      s = (var / len(ys)) ** 0.5
    else:
      s = 0.0
    xs.append(x)
    mean.append(m)
    se.append(s)
  return xs, mean, se, n_seeds


def _mean_se(vals: list[float]) -> tuple[float, float]:
  n = len(vals)
  if n == 0:
    return 0.0, 0.0
  m = sum(vals) / n
  if n == 1:
    return m, 0.0
  var = sum((v - m) ** 2 for v in vals) / (n - 1)
  return m, math.sqrt(var / n)


def _parse_csv_xy(path: str, x_col: str, y_col: str) -> list[tuple[int, float]]:
  pts: list[tuple[int, float]] = []
  try:
    with open(path, newline='') as f:
      for row in csv.DictReader(f):
        x = base._coerce(row.get(x_col, ''))
        y = base._coerce(row.get(y_col, ''))
        if x is not None and y is not None:
          pts.append((int(x), y))
  except (OSError, csv.Error):
    return []
  return pts


def _seed_id_from_run(run_name: str) -> str:
  m = re.search(r'_(\d+)$', run_name)
  return m.group(1) if m else run_name


def _named_runs_from_csv(log_dir_name: str) -> list[dict]:
  root = os.path.join(base.LOG_ROOT, log_dir_name)
  try:
    run_names = sorted(os.listdir(root))
  except OSError:
    return []
  out: list[dict] = []
  for run_name in run_names:
    tpath = os.path.join(root, run_name, 'logs', 'learner', 'logs.csv')
    epath = os.path.join(root, run_name, 'logs', 'eval', 'logs.csv')
    train = (
        _parse_csv_xy(tpath, base.TRAIN_X_COL, base.TRAIN_METRIC)
        if os.path.isfile(tpath) else [])
    eval_iter = (
        _parse_csv_xy(epath, base.EVAL_X_COL, base.EVAL_METRIC)
        if os.path.isfile(epath) else [])
    ev = (
        base._iters_to_env_steps(eval_iter, base.LOG_ROOT, log_dir_name)
        if eval_iter else [])
    if not train and not ev:
      continue
    out.append({
        'id': _seed_id_from_run(run_name),
        'run': run_name,
        'train': train,
        'eval': ev,
    })
  return out


def _named_runs_from_slurm(log_dir_name: str) -> list[dict]:
  trains = base._read_train_seed_series_from_slurm(base.SLURM_DIR, log_dir_name)
  evals_iter = base._read_eval_seed_series_from_slurm(
      base.SLURM_DIR, log_dir_name)
  evals = [
      base._iters_to_env_steps(pts, base.LOG_ROOT, log_dir_name)
      for pts in evals_iter
  ]
  n = max(len(trains), len(evals))
  out: list[dict] = []
  for i in range(n):
    out.append({
        'id': str(i),
        'run': f'slurm_{i}',
        'train': trains[i] if i < len(trains) else [],
        'eval': evals[i] if i < len(evals) else [],
    })
  return out


def _load_config(members: list[str], active: set[str]) -> dict:
  """Load every seed of dirs that share one config (not averaged away)."""
  seeds: list[dict] = []
  running = False
  multi_dir = len(members) > 1
  for name in members:
    csv_runs = _named_runs_from_csv(name)
    slurm_runs = _named_runs_from_slurm(name)
    chosen = csv_runs if len(csv_runs) >= len(slurm_runs) else slurm_runs
    if multi_dir:
      tag = name.replace(TASK_PREFIX, '')[-24:]
      for r in chosen:
        r['id'] = f'{tag}/{r["id"]}'
    seeds.extend(chosen)
    running = running or name in active

  train_series = [s['train'] for s in seeds if s['train']]
  eval_series = [s['eval'] for s in seeds if s['eval']]
  txs, tmean, tse, n_train = _aggregate_aligned(train_series)
  exs, emean, ese, n_eval = _aggregate_aligned(eval_series)

  for s in seeds:
    s['peak_train'] = _peak_from_mean([y for _, y in s['train']])
    s['peak_eval'] = _peak_from_mean([y for _, y in s['eval']])
    s['end_train'] = s['train'][-1][1] if s['train'] else 0.0
    s['end_eval'] = s['eval'][-1][1] if s['eval'] else 0.0
    s['peak'] = max(s['peak_train'], s['peak_eval'])

  peaks_t = [s['peak_train'] for s in seeds]
  peaks_e = [s['peak_eval'] for s in seeds]
  mt, set_ = _mean_se(peaks_t)
  me, see = _mean_se(peaks_e)
  key = _config_key(members[0])
  n = len(seeds)
  return {
      'name': key,
      'members': members,
      'fam': base._method_family(key),
      'label': _legend_label(key),
      'running': running,
      'seeds': seeds,
      'train': (txs, tmean, tse, n_train),
      'eval': (exs, emean, ese, n_eval),
      'peak_train': mt,
      'peak_eval': me,
      'peak_train_se': set_,
      'peak_eval_se': see,
      # classify worked/failed by the best seed, not the mean curve
      'peak': max((s['peak'] for s in seeds), default=0.0),
      'n': n,
  }


def _xy_sub(pts: list[tuple[int, float]]):
  if not pts:
    return [], []
  xs = [p[0] for p in pts]
  ys = [p[1] for p in pts]
  xs, ys, _ = base._subsample_curve(xs, ys, [0.0] * len(xs))
  return xs, ys


def _plot_config(axes, run: dict, i: int) -> bool:
  """One legend entry per config; n≥2 is mean ± stderr (plus faint seed traces)."""
  color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
  seeds = run['seeds']
  n = len(seeds)
  if n == 0:
    return False
  status = ' · RUNNING' if run['running'] else ''
  lw = 2.6 if run['running'] else 2.0
  plotted = False

  if n == 1:
    seed = seeds[0]
    legend = (
        f"{run['label']} (n=1){status}\n"
        f"    peak train={seed['peak_train']:.2f} · peak eval={seed['peak_eval']:.2f}"
    )
    for ax, key in ((axes[0], 'train'), (axes[1], 'eval')):
      xs, ys = _xy_sub(seed[key])
      if not xs:
        continue
      kw = dict(color=color, linestyle='-', linewidth=lw, alpha=0.95, zorder=3 + i)
      if legend is not None and (ax is axes[0] or not seed['train']):
        kw['label'] = legend
        legend = None
      ax.plot(xs, ys, **kw)
      plotted = True
    return plotted

  legend = (
      f"{run['label']} (n={n}){status}\n"
      f"    mean±SE peakT={run['peak_train']:.2f}±{run['peak_train_se']:.2f}"
      f" · peakE={run['peak_eval']:.2f}±{run['peak_eval_se']:.2f}"
  )
  for ax, series_key in ((axes[0], 'train'), (axes[1], 'eval')):
    xs, mean, se, _n = run[series_key]
    if not xs:
      continue
    xs, mean, se = base._subsample_curve(xs, mean, se)
    kw = dict(color=color, linestyle='-', linewidth=lw, alpha=0.95, zorder=3 + i)
    if legend is not None and (ax is axes[0] or not run['train'][0]):
      kw['label'] = legend
      legend = None
    ax.plot(xs, mean, **kw)
    if any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(
          xs, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2 + i)
    plotted = True

  for seed in seeds:
    for ax, key in ((axes[0], 'train'), (axes[1], 'eval')):
      xs, ys = _xy_sub(seed[key])
      if not xs:
        continue
      ax.plot(
          xs, ys, color=color, linestyle='-', linewidth=1.0, alpha=0.28,
          zorder=2 + i)
  return plotted


def _format_axes(ax, *, title: str, ylabel: str):
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.set_ylabel(ylabel, fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def _plot_bucket(runs: list[dict], *, fam: str, bucket: str,
                 threshold: float, out_dir: str) -> str | None:
  title_fam = base.FAMILY_TITLE.get(fam, fam.upper())
  bucket_title = (
      f'worked (peak ≥ {threshold:g})' if bucket == 'worked'
      else f'failed (peak < {threshold:g})'
  )

  n_lines = max(1, len(runs))
  fig_h = 5.2 + 0.22 * max(0, n_lines - 6)
  fig, axes = plt.subplots(1, 2, figsize=(14.5, fig_h), sharey=True)
  plotted = 0
  for i, run in enumerate(runs):
    if _plot_config(axes, run, i):
      plotted += 1
      seed_bits = ', '.join(
          f"s{s['id']} peakT={s['peak_train']:.2f} endT={s['end_train']:.2f}"
          for s in run['seeds']
      )
      print(
          f"  [{bucket}] {run['label']}: n={run['n']} "
          f"mean peakT={run['peak_train']:.3f}±{run['peak_train_se']:.3f} "
          f"peakE={run['peak_eval']:.3f}±{run['peak_eval_se']:.3f}"
          f"{' RUNNING' if run['running'] else ''}"
      )
      if run['n'] > 1:
        print(f"           {seed_bits}")

  _format_axes(
      axes[0],
      title=f'{GROUP} · {title_fam} · {bucket_title}\ntrain success (last 1000)',
      ylabel='Train Success (last 1000)',
  )
  _format_axes(
      axes[1],
      title=f'{GROUP} · {title_fam} · {bucket_title}\neval success',
      ylabel='Eval Success',
  )

  if plotted == 0:
    for ax in axes:
      ax.text(
          0.5, 0.5,
          f'No {title_fam} runs in this bucket',
          transform=ax.transAxes, ha='center', va='center',
          fontsize=12, color='#666666', fontstyle='italic',
      )
    handles, labels = [], []
  else:
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
      handles, labels = axes[1].get_legend_handles_labels()

  fig.suptitle(
      f'BuilderBench Creative 7 Task 2 — {title_fam} density estimator\n'
      f'{bucket_title} · n≥2: one series per config (mean ± SE)',
      fontsize=13, fontweight='bold', y=1.02,
  )

  if handles:
    leg = fig.legend(
        handles, labels,
        loc='upper left', bbox_to_anchor=(1.01, 0.98),
        fontsize=8.0, framealpha=1.0, ncol=1,
        edgecolor='#333333', fancybox=False, borderpad=0.6,
        handlelength=2.2, labelspacing=0.85, borderaxespad=0.0,
    )
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(1.1)
    for text in leg.get_texts():
      text.set_fontweight('bold')
    extra = (leg,)
  else:
    extra = ()

  fname = f'{GROUP}_{fam}_{bucket}_train_eval.png'
  out_path = os.path.join(out_dir, fname)
  fig.tight_layout(rect=[0, 0, 0.72 if handles else 1.0, 0.92])
  fig.savefig(
      out_path, dpi=150, bbox_inches='tight',
      bbox_extra_artists=extra,
  )
  plt.close(fig)
  return out_path


def run(threshold: float = DEFAULT_THRESHOLD, out_dir: str = OUT_DIR) -> list[str]:
  os.makedirs(out_dir, exist_ok=True)
  active = base._active_log_dirs(base.SLURM_DIR)
  names = _discover_c7t2_runs()
  print(f'Discovered {len(names)} creative7_task2 CRL/NF/TD3 log dirs '
        f'(threshold={threshold:g})')

  grouped: dict[str, list[str]] = defaultdict(list)
  for name in names:
    grouped[_config_key(name)].append(name)
  print(f'Grouped into {len(grouped)} unique configs')
  for key, members in sorted(grouped.items()):
    if len(members) > 1:
      print(f'  merge {len(members)} dirs → {key}')

  loaded = [_load_config(members, active) for members in grouped.values()]
  outs: list[str] = []

  for fam in FAMILIES:
    fam_runs = [r for r in loaded if r['fam'] == fam]
    fam_runs.sort(key=lambda r: (-r['peak'], r['label'], r['name']))
    worked = [r for r in fam_runs if r['peak'] >= threshold]
    failed = [r for r in fam_runs if r['peak'] < threshold]
    print(f'\n=== {fam.upper()} ({len(fam_runs)} configs: '
          f'{len(worked)} worked, {len(failed)} failed) ===')
    for bucket, bucket_runs in (('worked', worked), ('failed', failed)):
      path = _plot_bucket(
          bucket_runs, fam=fam, bucket=bucket,
          threshold=threshold, out_dir=out_dir,
      )
      if path:
        print(f'  → {path}')
        outs.append(path)
  return outs


def main():
  p = argparse.ArgumentParser()
  p.add_argument(
      '--threshold', type=float, default=DEFAULT_THRESHOLD,
      help='Peak train-or-eval success to count as worked (default 0.1).')
  p.add_argument('--out_dir', default=OUT_DIR)
  args = p.parse_args()
  outs = run(threshold=args.threshold, out_dir=args.out_dir)
  print(f'\nWrote {len(outs)} figure(s).')


if __name__ == '__main__':
  main()
