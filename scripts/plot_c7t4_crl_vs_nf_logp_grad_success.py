#!/usr/bin/env python3
"""c7t4 CRL vs NF compact-small: success, ∇_s, raw repr reward, NF log p.

Two columns, same task. Train success is raw ``train_success_1000``. Eval
success is faint raw + bold rolling mean (window=5). Remaining panels are
raw learner traces. CRL has no replay log p (those panels are blank).

  python scripts/plot_c7t4_crl_vs_nf_logp_grad_success.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c7t4_crl_vs_nf_logp_grad_success.png')
SPI = 1024 * 60  # num_envs × episode_length (ep60 jobs)
PLOT_MAX_PTS = 2500

# cols: panel_key -> learner CSV column (None = n/a for that method)
RUNS = (
    {
        'title': 'c7t4 · CRL τ=0.5 catwp',
        'log_dir': (
            'ppo_builderbench_creative7_task4_e1024_pd_crl_tau05'
            '_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl10'
        ),
        'run': 'ppo_builderbench_creative_7_task4_0',
        'cols': {
            'train_success_1000': 'train_success_1000',
            'grad_s': 'crl/crl_phi_psi_grad_s_norm_mean',
            'reward': 'reward_repr_raw_mean',
            'logp_mean': None,
            'logp_max': None,
            'logp_min': None,
        },
    },
    {
        'title': 'c7t4 · NF compact-small catwp',
        'log_dir': (
            'ppo_builderbench_creative7_task4_e1024_pd_nf_compact_small'
            '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
            '_minstd1e5_entanneal_ep60_300m_crl10'
        ),
        'run': 'ppo_builderbench_creative_7_task4_0',
        'cols': {
            'train_success_1000': 'train_success_1000',
            'grad_s': 'nf/nf_logp_grad_s_norm_mean',
            'reward': 'reward_repr_raw_mean',
            'logp_mean': 'nf/log_p_mean',
            'logp_max': 'nf/log_p_max',
            'logp_min': 'nf/log_p_min',
        },
    },
)

# (source, panel_key, ylabel, kind, color)
PANELS = (
    ('learner', 'train_success_1000', 'Train success (zoomed)', 'success',
     '#2A9D8F'),
    ('eval', 'success', 'Eval success (zoomed)', 'success', '#4C9BE8'),
    ('learner', 'grad_s', r'mean $\|\nabla_s\|$', 'linear', '#9B2226'),
    ('learner', 'reward', 'raw repr reward', 'linear', '#E07A3D'),
    ('learner', 'logp_mean', 'replay log p mean', 'linear', '#264653'),
    ('learner', 'logp_max', 'replay log p max', 'linear', '#1D6A9A'),
    ('learner', 'logp_min', 'replay log p min', 'linear', '#6B8E9F'),
)


def _csv_path(log_dir: str, run: str, kind: str) -> str:
  return os.path.join(
      REPO, 'logs', log_dir, run, 'logs', kind, 'logs.csv')


def _downsample(xs, ys, max_pts: int = PLOT_MAX_PTS):
  n = len(xs)
  if n <= max_pts:
    return xs, ys
  stride = max(1, n // max_pts)
  xs_d = list(xs[::stride])
  ys_d = list(ys[::stride])
  if xs_d[-1] != xs[-1]:
    xs_d.append(xs[-1])
    ys_d.append(ys[-1])
  return xs_d, ys_d


def _load_cols(
    path: str, x_col: str, y_cols: tuple[str, ...],
) -> dict[str, tuple[list, list]]:
  out = {c: ([], []) for c in y_cols}
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return out
  with open(path, newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader, None)
    if not header:
      return out
    try:
      x_i = header.index(x_col)
    except ValueError:
      return out
    y_is = [(c, header.index(c)) for c in y_cols if c in header]
    for row in reader:
      if len(row) <= x_i:
        continue
      x = base._coerce(row[x_i])
      if x is None:
        continue
      for c, i in y_is:
        if i >= len(row):
          continue
        y = base._coerce(row[i])
        if y is None:
          continue
        out[c][0].append(float(x))
        out[c][1].append(float(y))
  return out


def _plot_success_zoomed(ax, xs, ys, color, extras=()):
  """Autoscale y so 1e-3 train blips are visible; mark nonzero points."""
  ax.plot(xs, ys, color=color, linewidth=1.15, alpha=0.9, zorder=3)
  all_y = list(ys)
  nz = [(x, y) for x, y in zip(xs, ys) if y > 0]
  if nz:
    ax.scatter(
        [x for x, _ in nz], [y for _, y in nz],
        color=color, s=32, zorder=5, marker='o',
        edgecolors='white', linewidths=0.5)
  for exs, eys, ecol, elabel, els in extras:
    ax.plot(exs, eys, color=ecol, linewidth=1.0, alpha=0.85,
            linestyle=els, label=elabel, zorder=2)
    all_y.extend(eys)
    enz = [(x, y) for x, y in zip(exs, eys) if y > 0]
    if enz:
      ax.scatter(
          [x for x, _ in enz], [y for _, y in enz],
          color=ecol, s=28, zorder=5, marker='s',
          edgecolors='white', linewidths=0.5)
  ymax = max(all_y) if all_y else 0.0
  if ymax <= 0:
    ax.set_ylim(-0.05, 1.05)
    ax.text(
        0.98, 0.88, 'all 0', transform=ax.transAxes, ha='right',
        fontsize=8, color='0.45')
  else:
    ax.set_ylim(-0.12 * ymax, ymax * 1.55)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4g'))
    # one annotation per isolated spike cluster
    labeled = []
    for x, y in nz:
      if labeled and (x - labeled[-1][0]) < 2e6 and abs(y - labeled[-1][1]) < 1e-9:
        continue
      labeled.append((x, y))
      ax.annotate(
          f'{y:g}', (x, y), textcoords='offset points', xytext=(5, 7),
          fontsize=7, color=color)
  if extras:
    ax.legend(fontsize=7, frameon=False, loc='upper left')


def _summarize(col: str, xs, ys) -> None:
  if not ys:
    print(f'  {col}: no data')
    return
  peak_i = max(range(len(ys)), key=lambda i: ys[i])
  print(
      f'  {col}: n={len(ys)}  last={ys[-1]:.4g} @ {xs[-1]:.0f}  '
      f'peak={ys[peak_i]:.4g} @ {xs[peak_i]:.0f}')


def main() -> None:
  loaded = []
  for spec in RUNS:
    learn_cols = tuple(
        c for c in list(spec['cols'].values()) + ['train_success_mean'] if c)
    series = _load_cols(
        _csv_path(spec['log_dir'], spec['run'], 'learner'),
        'global_step', learn_cols)
    ev_raw = _load_cols(
        _csv_path(spec['log_dir'], spec['run'], 'eval'),
        'iteration', ('success',))
    ev_it, ev_y = ev_raw.get('success', ([], []))
    ev = ([it * SPI for it in ev_it], ev_y)
    print(spec['title'])
    for key, col in spec['cols'].items():
      if not col:
        print(f'  {key}: n/a')
        continue
      xs, ys = series.get(col, ([], []))
      _summarize(f'{key} ({col})', xs, ys)
    _summarize('eval success', ev[0], ev[1])
    loaded.append((spec, series, ev))

  fig, axes = plt.subplots(
      len(PANELS), 2, figsize=(12.4, 15.2), sharex='col')
  if len(PANELS) == 1:
    axes = [axes]

  for col_i, (spec, series, ev) in enumerate(loaded):
    for row_i, (src, key, ylabel, kind, color) in enumerate(PANELS):
      ax = axes[row_i][col_i]
      plotted = False
      if src == 'eval':
        xs, ys = ev
        if xs:
          _plot_success_zoomed(ax, xs, ys, color)
          plotted = True
      elif key == 'train_success_1000':
        col = spec['cols'].get(key)
        xs, ys = series.get(col, ([], [])) if col else ([], [])
        mx, my = series.get('train_success_mean', ([], []))
        extras = ()
        if mx and any(v > 0 for v in my):
          extras = ((mx, my, '#C45C26', 'train_success_mean', '--'),)
        if xs:
          _plot_success_zoomed(ax, xs, ys, color, extras=extras)
          plotted = True
      else:
        col = spec['cols'].get(key)
        if col:
          xs, ys = series.get(col, ([], []))
          if xs:
            xs, ys = _downsample(xs, ys)
            ax.plot(xs, ys, color=color, linewidth=1.25, alpha=0.92)
            plotted = True
      if not plotted and src == 'learner' and not spec['cols'].get(key):
        ax.text(
            0.5, 0.5, 'n/a (CRL has no log p)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color='0.45')
      if kind == 'success':
        pass  # ylim set in _plot_success_zoomed
      elif plotted:
        ax.axhline(0.0, color='0.75', lw=0.6, zorder=0)
      ax.spines[['top', 'right']].set_visible(False)
      ax.grid(axis='y', linestyle='--', alpha=0.4)
      if col_i == 0:
        ax.set_ylabel(ylabel, fontsize=8)
      if row_i == 0:
        ax.set_title(spec['title'], fontsize=10, fontweight='bold')
      if row_i == len(PANELS) - 1:
        ax.set_xlabel('Env Steps', fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout()
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
