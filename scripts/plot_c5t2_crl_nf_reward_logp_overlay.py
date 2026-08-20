#!/usr/bin/env python3
"""c5t2 CRL vs NF overlay: PPO / mean / min on a shared y-axis.

Left CRL is scaled to left NF; right CRL is scaled to right NF (same
column). Cyan/min uses the first-half mean; PPO and black use the
second-half mean. Traces are a centered time-window mean (faint raw +
bold). ylim clips NF PPO jumps below -40.

  python scripts/plot_c5t2_crl_nf_reward_logp_overlay.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench',
    'c5t2_crl_nf_reward_logp_overlay_mean_matched.png')
PLOT_MAX_PTS = 2500
RUN = 'ppo_builderbench_creative_5_task2_0'

CRL_SERIES = (
    ('reward_repr_raw_mean', 'PPO reward (raw)', '#d1495b'),
    ('crl/logits_pos', 'mean pos logit', '#264653'),
    ('crl/logits_neg', 'mean neg logit', '#2A9D8F'),
)
NF_SERIES = (
    ('reward_repr_raw_mean', 'PPO log p (raw)', '#d1495b'),
    ('nf/log_p_mean', 'mean log p', '#264653'),
    ('nf/log_p_min', 'min log p', '#2A9D8F'),
)

PANELS = (
    (
        'c5t2 CRL · no gradreg  (scaled ×; cyan 1st-half mean → NF)',
        'ppo_builderbench_creative5_task2_e1024_pd_crl_tau05_nopermute_'
        'catselect_ent0005to0001',
        CRL_SERIES,
        2,  # scale to NF no-gradreg (same column)
        'PPO reward vs mean pos / neg logit',
    ),
    (
        'c5t2 CRL · gradreg  (scaled ×; cyan 1st-half mean → NF)',
        'ppo_builderbench_creative5_task2_e1024_pd_crl_tau05_nopermute_norand_'
        'catselect_gradreg',
        CRL_SERIES,
        3,  # scale to NF dualgradreg (same column)
        'PPO reward vs mean pos / neg logit',
    ),
    (
        'c5t2 NF compact-small · no gradreg (τ=0.9)',
        'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau09_nopermute_fixedx01_catselect'
        '_minstd1e5_entanneal_ep60_300m_crl10',
        NF_SERIES,
        None,
        'PPO log p vs mean / min replay log p',
    ),
    (
        'c5t2 NF compact-small · dualgradreg',
        'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau05_nopermute_norand_catselect_dualgradreg',
        NF_SERIES,
        None,
        'PPO log p vs mean / min replay log p',
    ),
)
YLIM = (-45.0, 20.0)  # NF typical range; ignore PPO/min spikes below -40
SMOOTH_WINDOW_STEPS = 5.0e6  # centered time-window mean (env steps)


def _csv_path(log_dir: str) -> str:
  return os.path.join(
      REPO, 'logs', log_dir, RUN, 'logs', 'learner', 'logs.csv')


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


def _load_cols(path: str, y_cols: tuple[str, ...]) -> dict:
  out = {c: ([], []) for c in y_cols}
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return out
  with open(path, newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader, None)
    if not header:
      return out
    try:
      x_i = header.index('global_step')
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
  return {k: (np.asarray(v[0], dtype=float), np.asarray(v[1], dtype=float))
          for k, v in out.items()}


# Per CRL series: which half of the run to match to NF.
# Cyan/min: first half (NF min sits near -40 then rises).
# Red/black: second half (NF mean/rollout sit near +10..15).
SCALE_HALF = ('second', 'second', 'first')


def _half_mean(xs, ys, half: str) -> float:
  if len(xs) == 0:
    return float('nan')
  mid = 0.5 * xs[-1]
  mask = xs < mid if half == 'first' else xs >= mid
  return float(np.mean(ys[mask]))


def _time_rolling_mean(xs, ys, window_steps: float = SMOOTH_WINDOW_STEPS):
  """Centered rolling mean over a time window in env steps (xs sorted)."""
  xs = np.asarray(xs, dtype=float)
  ys = np.asarray(ys, dtype=float)
  n = len(xs)
  if n == 0 or window_steps <= 0:
    return ys
  half = 0.5 * window_steps
  out = np.empty(n, dtype=float)
  lo = 0
  hi = 0
  acc = 0.0
  for i, t in enumerate(xs):
    while hi < n and xs[hi] <= t + half:
      acc += ys[hi]
      hi += 1
    while lo < hi and xs[lo] < t - half:
      acc -= ys[lo]
      lo += 1
    out[i] = acc / float(hi - lo)
  return out


def _scales_for(crl_data, nf_data, nf_series) -> list[float]:
  scales = []
  for (col, label, _), half, nf_col in zip(CRL_SERIES, SCALE_HALF, nf_series):
    crl_mu = _half_mean(*crl_data[col], half)
    nf_mu = _half_mean(*nf_data[nf_col[0]], half)
    scale = nf_mu / crl_mu
    scales.append(scale)
    print(f'  {label:22s}  {half:6s}  {crl_mu:8.3f} → {nf_mu:8.3f}  ×{scale:.3f}')
  return scales


def main() -> None:
  loaded = []
  for title, log_dir, series, match_nf, subtitle in PANELS:
    cols = tuple(c for c, _, _ in series)
    data = _load_cols(_csv_path(log_dir), cols)
    loaded.append((title, series, match_nf, subtitle, data))

  panel_scales: list[list[float] | None] = [None] * len(loaded)
  for i, (title, series, match_nf, subtitle, data) in enumerate(loaded):
    if match_nf is None:
      continue
    nf_series = loaded[match_nf][1]
    nf_data = loaded[match_nf][4]
    print(f'{title.split("(")[0].strip()}  →  '
          f'{loaded[match_nf][0].split("(")[0].strip()}  (y ← k·y)')
    panel_scales[i] = _scales_for(data, nf_data, nf_series)

  print(f'\nTime-window means (centered {SMOOTH_WINDOW_STEPS/1e6:.0f}M steps)')
  print(f'  {"panel":44s} {"series":22s} {"1st-half":>9s} {"2nd-half":>9s}')
  for i, (title, series, match_nf, subtitle, data) in enumerate(loaded):
    short = title.split('(')[0].strip()
    scales = panel_scales[i]
    for si, (col, label, color) in enumerate(series):
      xs, ys = data[col]
      if len(xs) == 0:
        continue
      y_use = ys * scales[si] if scales is not None else ys
      sm = _time_rolling_mean(xs, y_use)
      m1 = _half_mean(xs, sm, 'first')
      m2 = _half_mean(xs, sm, 'second')
      print(f'  {short:44s} {label:22s} {m1:9.3f} {m2:9.3f}')

  fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), sharex=True, sharey=True)
  w_label = f'roll mean, window={SMOOTH_WINDOW_STEPS/1e6:.0f}M steps'

  for ax, (title, series, match_nf, subtitle, data), panel_i in zip(
      axes.ravel(), loaded, range(4)):
    scales = panel_scales[panel_i]
    for si, (col, label, color) in enumerate(series):
      xs, ys = data[col]
      if len(xs) == 0:
        continue
      y_plot = ys
      lab = label
      if scales is not None:
        y_plot = ys * scales[si]
        lab = f'{label}  ×{scales[si]:.2f}'
      sm = _time_rolling_mean(xs, y_plot)
      xs_r, ys_r = _downsample(xs, y_plot)
      xs_s, ys_s = _downsample(xs, sm)
      ax.plot(xs_r, ys_r, color=color, linewidth=0.8, alpha=0.28, zorder=2)
      ax.plot(xs_s, ys_s, color=color, linewidth=1.6, alpha=0.95,
              label=lab, zorder=3)
    ax.axhline(0.0, color='0.75', lw=0.6, zorder=0)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.text(
        0.5, 0.02, f'{subtitle}  ·  {w_label}', transform=ax.transAxes,
        fontsize=7.5, color='0.4', ha='center', va='bottom')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=7, frameon=False, loc='best')
    if panel_i >= 2:
      ax.set_xlabel('Env Steps', fontsize=9)
      ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
    if panel_i % 2 == 0:
      ax.set_ylabel('value', fontsize=9)

  axes[0, 0].set_ylim(*YLIM)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'ylim={YLIM}')
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
