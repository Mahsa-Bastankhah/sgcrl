#!/usr/bin/env python3
"""Presentation-ready success plots (large fonts, smoothed mean ± SE).

Writes PNGs under ``ready_plots/``:
  - builderbench_cube3_task1.png
  - builderbench_cube3_task2.png
  - sawyer_peg.png
  - point_impossible.png

Legend (colors match figs/ready_plots/cube3_task1_density_estimators):
  - Goal preimage  (algorithm)  — orange #E67E22
  - Dirac target   (baseline)   — blue   #2E86C1  (flat zero if no logs)
  - PPO+RND        (baseline)   — green  #27AE60  (flat zero if no logs)
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Same palette as cube3_task1_density_estimators (CRL / NF / TD3).
COLORS = {
    'Goal preimage': '#E67E22',
    'Dirac target': '#2E86C1',
    'PPO+RND': '#27AE60',
}

TITLE_FS = 28
LABEL_FS = 24
TICK_FS = 22
LEGEND_FS = 20

# (label, xs_list, ys_list, se_list|None)
# se_list: optional per-seed uncertainty (e.g. episode std); used when n_seeds==1.
LabelSeries = Tuple[
    str, List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]]]


def _read_csv(path: str) -> List[Dict[str, str]]:
  with open(path, 'rb') as fh:
    raw = fh.read().replace(b'\x00', b'')
  text = raw.decode('utf-8', errors='replace')
  lines = [ln for ln in text.splitlines() if ln.strip()]
  if not lines:
    return []
  return list(csv.DictReader(io.StringIO('\n'.join(lines))))


def _to_env_steps(
    rows: Sequence[Dict[str, str]],
    total_steps: Optional[float] = None,
) -> np.ndarray:
  if not rows:
    return np.asarray([], dtype=np.float64)
  keys = rows[0].keys()
  if 'global_step' in keys:
    return np.asarray([float(r['global_step']) for r in rows], dtype=np.float64)
  if 'learner_steps' in keys:
    xs = np.asarray([float(r['learner_steps']) for r in rows], dtype=np.float64)
    if xs.max() >= 1e5:
      return xs
    if total_steps is not None and xs.max() > 0:
      return xs / xs.max() * float(total_steps)
    return xs
  if 'iteration' in keys:
    xs = np.asarray([float(r['iteration']) for r in rows], dtype=np.float64)
    if total_steps is not None and xs.max() > 0:
      return xs / xs.max() * float(total_steps)
    return xs
  raise KeyError(f'no x-axis column in {list(keys)}')


def _metric_col(rows: Sequence[Dict[str, str]], preferred: Sequence[str]) -> str:
  keys = rows[0].keys()
  for name in preferred:
    if name in keys:
      return name
  raise KeyError(f'none of {preferred} in {list(keys)}')


def _rolling_smooth(y: np.ndarray, window: int) -> np.ndarray:
  if window <= 1 or y.size == 0:
    return y.copy()
  out = np.empty_like(y, dtype=np.float64)
  csum = np.cumsum(y, dtype=np.float64)
  for i in range(y.size):
    j0 = max(0, i - window + 1)
    total = csum[i] - (csum[j0 - 1] if j0 > 0 else 0.0)
    out[i] = total / (i - j0 + 1)
  return out


def _unique_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  order = np.argsort(x)
  x_u, y_u = x[order], y[order]
  uniq_x, inv = np.unique(x_u, return_inverse=True)
  uniq_y = np.zeros_like(uniq_x, dtype=np.float64)
  counts = np.zeros_like(uniq_x, dtype=np.float64)
  for i, yi in enumerate(y_u):
    uniq_y[inv[i]] += yi
    counts[inv[i]] += 1.0
  uniq_y /= np.maximum(counts, 1.0)
  return uniq_x, uniq_y


def _interpolate_to_grid(
    xs_list: List[np.ndarray],
    ys_list: List[np.ndarray],
    se_list: Optional[List[np.ndarray]] = None,
    n_grid: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Align seeds on a common x-grid; return mean and SE (or episode-std band)."""
  x_max = max(float(x.max()) for x in xs_list if x.size)
  x_min = min(float(x.min()) for x in xs_list if x.size)
  grid = np.linspace(x_min, x_max, n_grid)
  stacked = []
  for x, y in zip(xs_list, ys_list):
    if x.size < 2:
      continue
    xu, yu = _unique_xy(x, y)
    stacked.append(np.interp(grid, xu, yu))
  arr = np.asarray(stacked, dtype=np.float64)
  mean = arr.mean(axis=0)
  if arr.shape[0] > 1:
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
  elif se_list and se_list[0].size:
    xu, su = _unique_xy(xs_list[0], se_list[0])
    se = np.interp(grid, xu, su)
  else:
    se = np.zeros_like(mean)
  return grid, mean, se


def _format_steps(v, _pos) -> str:
  if v == 0:
    return '0'
  if v >= 1e6:
    s = f'{v / 1e6:.1f}M'
    return s.replace('.0M', 'M')
  if v >= 1e3:
    return f'{v / 1e3:.0f}K'
  return str(int(v))


def _zero_series(x_max: float, n: int = 50) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  xs = np.linspace(0.0, float(x_max), n)
  ys = np.zeros_like(xs)
  return [xs], [ys]


def _x_max_from_series(series: List[LabelSeries]) -> float:
  mx = 0.0
  for _label, xs_list, _ys, _se in series:
    for x in xs_list:
      if x.size:
        mx = max(mx, float(x.max()))
  return mx if mx > 0 else 1.0


def _style_axes(ax, title: str) -> None:
  ax.set_title(title, fontsize=TITLE_FS, pad=14, fontweight='semibold')
  ax.set_xlabel('Environment steps', fontsize=LABEL_FS)
  ax.set_ylabel('Success rate', fontsize=LABEL_FS)
  ax.tick_params(axis='both', labelsize=TICK_FS)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_steps))
  ax.set_ylim(-0.02, 1.05)
  ax.grid(True, axis='y', linestyle='--', alpha=0.35, linewidth=1.0)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  for spine in ('left', 'bottom'):
    ax.spines[spine].set_linewidth(1.4)
  ax.legend(
      loc='best',
      fontsize=LEGEND_FS,
      framealpha=0.95,
      edgecolor='#D6D3D1',
      fancybox=False,
  )


def _plot_methods(
    series: List[LabelSeries],
    title: str,
    out_path: str,
    smooth_window: int,
) -> None:
  # Fill missing Dirac / RND with flat zero over the observed x-range.
  x_max = _x_max_from_series(series)
  filled: List[LabelSeries] = []
  for label, xs_list, ys_list, se_list in series:
    if not xs_list and label in ('Dirac target', 'PPO+RND'):
      zx, zy = _zero_series(x_max)
      filled.append((label, zx, zy, None))
      print(f'[zero] {title}: {label} (flat zero baseline)')
    else:
      filled.append((label, xs_list, ys_list, se_list))

  fig, ax = plt.subplots(figsize=(11.5, 7.0))
  plotted = 0
  # Draw baselines first so Goal preimage sits on top.
  order = ['Dirac target', 'PPO+RND', 'Goal preimage']
  by_label = {s[0]: s for s in filled}

  for label in order:
    if label not in by_label:
      continue
    _lab, xs_list, ys_list, se_list = by_label[label]
    if not xs_list:
      print(f'[skip] {title}: no data for {label}')
      continue
    ys_s = [_rolling_smooth(y, smooth_window) for y in ys_list]
    se_s = None
    if se_list is not None:
      se_s = [_rolling_smooth(s, smooth_window) for s in se_list]
    grid, mean, se = _interpolate_to_grid(xs_list, ys_s, se_s)
    color = COLORS.get(label, '#334155')
    lw = 3.6 if label == 'Goal preimage' else 3.0
    # Distinct linestyles so flat-zero baselines stay visible in the legend.
    ls = {'Goal preimage': '-', 'Dirac target': '--', 'PPO+RND': ':'}.get(
        label, '-')
    ax.plot(
        grid, mean, color=color, linewidth=lw, linestyle=ls, label=label,
        zorder=3)
    if np.any(se > 1e-8):
      ax.fill_between(
          grid,
          np.clip(mean - se, -0.02, 1.05),
          np.clip(mean + se, -0.02, 1.05),
          color=color,
          alpha=0.22,
          linewidth=0,
          zorder=2,
      )
    plotted += 1
    n = len(xs_list)
    print(f'[ok] {title}: {label} ({n} seed{"s" if n != 1 else ""})')

  if plotted == 0:
    print(f'[warn] nothing to plot for {title}')
    plt.close(fig)
    return

  _style_axes(ax, title)
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
  fig.savefig(out_path, dpi=220, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  print(f'[saved] {out_path}')


def _load_eval_runs(
    patterns: Sequence[str],
    metric_prefs: Sequence[str],
    total_steps: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  xs_list, ys_list = [], []
  paths: List[str] = []
  for pat in patterns:
    paths.extend(sorted(glob.glob(pat)))
  seen = set()
  for path in paths:
    if path in seen:
      continue
    seen.add(path)
    csv_path = path
    if os.path.isdir(path):
      csv_path = os.path.join(path, 'logs', 'eval', 'logs.csv')
    if not os.path.isfile(csv_path):
      print(f'[warn] missing {csv_path}')
      continue
    rows = _read_csv(csv_path)
    if not rows:
      continue
    met = _metric_col(rows, metric_prefs)
    xs = _to_env_steps(rows, total_steps=total_steps)
    ys = np.asarray([float(r[met]) for r in rows], dtype=np.float64)
    xs_list.append(xs)
    ys_list.append(ys)
  return xs_list, ys_list


def _load_checkpoint_csvs(
    patterns: Sequence[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
  xs_list, ys_list, se_list = [], [], []
  for pat in patterns:
    for path in sorted(glob.glob(pat)):
      rows = _read_csv(path)
      if not rows:
        continue
      xs = np.asarray([float(r['global_step']) for r in rows], dtype=np.float64)
      ys = np.asarray([float(r['success_mean']) for r in rows], dtype=np.float64)
      if 'success_std' in rows[0]:
        se = np.asarray([float(r['success_std']) for r in rows], dtype=np.float64)
      else:
        se = np.zeros_like(ys)
      xs_list.append(xs)
      ys_list.append(ys)
      se_list.append(se)
      print(f'  loaded ckpt csv {os.path.basename(path)} ({len(rows)} pts)')
  return xs_list, ys_list, se_list


def plot_builderbench_task1(out_dir: str, smooth: int) -> None:
  # Two NF runs when available: seed0 (pd_nf) + seed1 (same tag or tau05 fallback).
  eval_dir = os.path.join(REPO, 'figs/builderbench/checkpoint_eval')
  ours_x, ours_y, ours_se = _load_checkpoint_csvs([
      os.path.join(
          eval_dir,
          'ppo_builderbench_creative3_task1_e1024_pd_nf_seed*_checkpoint_success.csv'),
      os.path.join(
          eval_dir,
          'ppo_builderbench_creative3_task1_e1024_pd_nf_tau05_seed1_checkpoint_success.csv'),
  ])
  # Deduplicate if seed1 matched both globs somehow — keep unique by basename.
  # _load already lists each file once per glob; second glob may add tau05 seed1
  # while seed1 of non-tau is missing — that is intentional for a 2-run band.
  series: List[LabelSeries] = [
      ('Goal preimage', ours_x, ours_y, ours_se if len(ours_x) == 1 else None),
      ('Dirac target', [], [], None),
      ('PPO+RND', [], [], None),
  ]
  _plot_methods(
      series,
      title='3 Cube Stacking',
      out_path=os.path.join(out_dir, 'builderbench_cube3_task1.png'),
      smooth_window=max(2, min(3, smooth)),
  )


def plot_builderbench_task2(out_dir: str, smooth: int) -> None:
  ours_x, ours_y, ours_se = _load_checkpoint_csvs([
      os.path.join(
          REPO,
          'figs/builderbench/checkpoint_eval/'
          'builderbench_creative3_task2_e1024_pd_nf_tau05_seed*_checkpoint_success.csv'),
  ])
  series: List[LabelSeries] = [
      ('Goal preimage', ours_x, ours_y, ours_se if len(ours_x) == 1 else None),
      ('Dirac target', [], [], None),
      ('PPO+RND', [], [], None),
  ]
  _plot_methods(
      series,
      title='2 Cube Stacking and One Cube Lifted',
      out_path=os.path.join(out_dir, 'builderbench_cube3_task2.png'),
      smooth_window=max(2, min(3, smooth)),
  )


def plot_sawyer_peg(out_dir: str, smooth: int) -> None:
  ours_x, ours_y = _load_eval_runs(
      [
          os.path.join(REPO, 'logs/ppo_peg_tau0p85_40m/ppo_sawyer_peg_*'),
          os.path.join(REPO, 'logs/ppo_peg_tau0p85_40m_consistency/ppo_sawyer_peg_*'),
      ],
      metric_prefs=('success_1000', 'success'),
      total_steps=40_000_000,
  )
  rnd_x, rnd_y = _load_eval_runs(
      [os.path.join(REPO, 'logs/ppo_rnd_peg_40m/ppo_rnd_sawyer_peg_*')],
      metric_prefs=('success', 'success_1000'),
      total_steps=40_000_000,
  )
  series: List[LabelSeries] = [
      ('Goal preimage', ours_x, ours_y, None),
      ('Dirac target', [], [], None),
      ('PPO+RND', rnd_x, rnd_y, None),
  ]
  _plot_methods(
      series,
      title='Sawyer Peg Insertion',
      out_path=os.path.join(out_dir, 'sawyer_peg.png'),
      smooth_window=smooth,
  )


def plot_point_impossible(out_dir: str, smooth: int) -> None:
  seeds = list(range(123, 129))
  ours_paths = [
      os.path.join(REPO, f'logs/ppo_impossible/ppo_point_Impossible_{s}')
      for s in seeds
  ]
  ours_x, ours_y = _load_eval_runs(
      ours_paths,
      metric_prefs=('success_1000', 'success'),
      total_steps=10_000_000,
  )
  rnd_x, rnd_y = _load_eval_runs(
      [os.path.join(REPO, 'logs/ppo_rnd_impossible_e8/ppo_rnd_point_Impossible_*')],
      metric_prefs=('success', 'success_1000'),
      total_steps=10_000_000,
  )
  series: List[LabelSeries] = [
      ('Goal preimage', ours_x, ours_y, None),
      ('Dirac target', [], [], None),  # filled as flat zero
      ('PPO+RND', rnd_x, rnd_y, None),
  ]
  _plot_methods(
      series,
      title='Point Maze Hard',
      out_path=os.path.join(out_dir, 'point_impossible.png'),
      smooth_window=max(5, smooth // 4),
  )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--output_dir',
      default=os.path.join(REPO, 'ready_plots'),
      help='Output folder for presentation PNGs.',
  )
  ap.add_argument(
      '--smooth_window',
      type=int,
      default=41,
      help='Rolling mean window (points) for dense eval curves.',
  )
  args = ap.parse_args()
  out = os.path.abspath(args.output_dir)
  os.makedirs(out, exist_ok=True)

  plt.rcParams.update({
      'font.size': TICK_FS,
      'axes.titlesize': TITLE_FS,
      'axes.labelsize': LABEL_FS,
      'xtick.labelsize': TICK_FS,
      'ytick.labelsize': TICK_FS,
      'legend.fontsize': LEGEND_FS,
      'axes.linewidth': 1.4,
  })

  print(f'[ready_plots] writing to {out}')
  plot_builderbench_task1(out, args.smooth_window)
  plot_builderbench_task2(out, args.smooth_window)
  plot_sawyer_peg(out, args.smooth_window)
  plot_point_impossible(out, args.smooth_window)
  print('[ready_plots] done')


if __name__ == '__main__':
  main()
