#!/usr/bin/env python3
"""Object z, GAE advantage, episode length, and actor-reset events.

Covers Allegro throw jobs that are running or finished today. Train curves
are raw (not smoothed). Actor-reset marks come from slurm
``[ppo] actor reset at iter=`` lines, not from every short-episode tick.

  python scripts/plot_allegro_live_objz_adv_eplen.py
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_live_objz_adv_eplen.png')
OUT_ENVDENSE = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_envdense_objz_adv_eplen.png')
RESET_RE = re.compile(r'\[ppo\] actor reset at iter=(\d+)')
RESET_EP_LEN = 200.0
NOMINAL_EP = 300.0
TABLE_Z = 0.55
FALL_Z = 0.1
C = base.ACCENT_COLORS

ENVDENSE_RUN = (
    'ppo_allegro_kuka_throw_e1024_envdense_objxyrand_noshape_ep300_100m'
    '_actorreset_2h',
    'envdense (NVIDIA rew_buf)',
    C[2],
    '-',
)

# Today's NF jobs (envdense is plotted alone — overlay was unreadable).
RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_objxyrand_noshape_mixtaskg_mix75_actorreset_4h',
        'objxy mix75',
        C[0],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_tableside_objxyrand_noshape_mixtaskg_mix75_actorreset_4h',
        'tableside objxy',
        C[1],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_norand_mixtaskg_mix75'
        '_actorreset_4h',
        'throw mix75',
        C[3],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_stateonly'
        '_actorreset_4h',
        'palm',
        C[4],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_stateonly'
        '_norand_mixtaskg_actorreset_4h',
        'palm mix norand',
        C[5],
        '--',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palm022_stateonly'
        '_actorreset_4h',
        'palm 0.22',
        C[6],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palm022_stateonly'
        '_mixtaskg_actorreset_4h',
        'palm 0.22 mix',
        C[7],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palm022_stateonly'
        '_norand_mixtaskg_actorreset_4h',
        'palm 0.22 mix norand',
        C[8],
        '--',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palm025_stateonly'
        '_mixtaskg_actorreset_4h',
        'palm 0.25 mix',
        C[9],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palm025_stateonly'
        '_norand_mixtaskg_actorreset_4h',
        'palm 0.25 mix norand',
        C[10],
        '--',
    ),
)


def _curve(log_dir: str, col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, log_dir, split='learner',
      x_col='global_step', y_col=col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return None
  xs, mean, _ = base._subsample_curve(xs, mean, se)
  return xs, mean, n


def _iter_to_step(log_dir: str) -> dict[int, int]:
  return base._iter_to_env_steps_map(base.LOG_ROOT, log_dir)


def _slurm_logs(log_dir: str) -> list[str]:
  return sorted(glob.glob(os.path.join(REPO, 'slurm', f'{log_dir}_*.log')))


def _squeue_job_ids() -> set[str]:
  try:
    out = subprocess.check_output(
        ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%A'],
        text=True, timeout=10)
  except (OSError, subprocess.SubprocessError):
    return set()
  return {tok.strip() for tok in out.split() if tok.strip().isdigit()}


def _is_live(log_dir: str, live_ids: set[str]) -> bool:
  if not live_ids:
    return False
  for path in _slurm_logs(log_dir):
    base_name = os.path.basename(path)
    for job_id in live_ids:
      if f'_{job_id}_' in base_name or base_name.endswith(f'_{job_id}.log'):
        return True
  return False


def _reset_steps(log_dir: str) -> list[int]:
  mapping = _iter_to_step(log_dir)
  steps: list[int] = []
  for path in _slurm_logs(log_dir):
    try:
      with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
          m = RESET_RE.search(line)
          if not m:
            continue
          it = int(m.group(1))
          if it in mapping:
            steps.append(int(mapping[it]))
    except OSError:
      continue
  return sorted(set(steps))


def _y_at(xs, ys, x):
  if not xs:
    return None
  best_i = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
  if abs(xs[best_i] - x) > 2.0 * (xs[1] - xs[0] if len(xs) > 1 else 1):
    return ys[best_i]
  return ys[best_i]


def _style(ax, ylabel: str, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=9)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)


def main() -> None:
  live_ids = _squeue_job_ids()
  fig, axes = plt.subplots(3, 1, figsize=(12.2, 10.4), sharex=True)
  ax_z, ax_adv, ax_ep = axes

  legend_handles = []
  legend_labels = []
  adv_later = []
  for log_dir, label, color, ls in RUNS:
    if _is_live(log_dir, live_ids):
      label = f'{label} (live)'
    resets = _reset_steps(log_dir)
    print(f'{label}: {len(resets)} actor-resets')

    z = _curve(log_dir, 'object_z_mean')
    if z is None:
      print(f'  no object_z_mean (job predates xyz logging)')
    else:
      xs, mean, n = z
      h, = ax_z.plot(xs, mean, color=color, linestyle=ls, lw=1.6, alpha=0.95)
      for rx in resets:
        ry = _y_at(xs, mean, rx)
        if ry is not None:
          ax_z.scatter([rx], [ry], color=color, marker='x', s=26,
                       linewidths=1.1, zorder=5)
      print(f'  object_z last={mean[-1]:.3f} n={n} steps={xs[-1]:.0f}')

    adv = _curve(log_dir, 'advantage_mean')
    if adv is None:
      print('  no advantage_mean')
    else:
      xs, mean, n = adv
      h, = ax_adv.plot(xs, mean, color=color, linestyle=ls, lw=1.6, alpha=0.95)
      for rx in resets:
        ry = _y_at(xs, mean, rx)
        if ry is not None:
          ax_adv.scatter([rx], [ry], color=color, marker='x', s=26,
                         linewidths=1.1, zorder=5)
      # Init iters have |A|~10 and would squash the rest of training.
      later = [y for x, y in zip(xs, mean) if x >= 3e6]
      if later:
        adv_later.extend(later)
      print(f'  adv last={mean[-1]:.3f} n_pts={len(xs)}')

    ep = _curve(log_dir, 'ep_length_mean')
    if ep is None:
      print('  no ep_length_mean')
      continue
    xs, mean, n = ep
    h, = ax_ep.plot(xs, mean, color=color, linestyle=ls, lw=1.6, alpha=0.95,
                    label=label)
    legend_handles.append(h)
    legend_labels.append(label)
    for rx in resets:
      ry = _y_at(xs, mean, rx)
      if ry is not None:
        ax_ep.scatter([rx], [ry], color=color, marker='x', s=26,
                      linewidths=1.1, zorder=5)
    print(f'  eplen last={mean[-1]:.1f} n={n}')

  ax_z.axhline(TABLE_Z, color='0.55', lw=0.8, linestyle='--')
  ax_z.axhline(FALL_Z, color='0.55', lw=0.8, linestyle=':')
  ax_z.set_title(
      r'object $z$ mean  (dashed = table $\approx 0.55$m; dotted = fall $0.1$m; '
      r'$\times$ = actor-reset)',
      fontsize=10, fontweight='bold')
  _style(ax_z, r'object $z$ (m)')

  ax_adv.axhline(0.0, color='0.55', lw=0.7)
  if adv_later:
    lo, hi = min(adv_later), max(adv_later)
    pad = 0.08 * max(hi - lo, 0.05)
    ax_adv.set_ylim(lo - pad, hi + pad)
  ax_adv.set_title(
      r'GAE advantage mean (pre z-score; y-lim after 3M steps; $\times$ = actor-reset)',
      fontsize=10, fontweight='bold')
  _style(ax_adv, 'advantage')

  ax_ep.axhline(NOMINAL_EP, color='0.55', lw=0.8, linestyle='--')
  ax_ep.axhline(RESET_EP_LEN, color='0.55', lw=1.0, linestyle=':')
  ax_ep.set_title(
      rf'episode length  (dashed = 300; dotted = reset thresh {RESET_EP_LEN:.0f}; '
      r'$\times$ = last-layer actor-reset)',
      fontsize=10, fontweight='bold')
  _style(ax_ep, 'ep length', xlabel=True)
  ax_ep.legend(legend_handles, legend_labels, loc='lower left', fontsize=7.5,
               ncol=3, framealpha=0.95)

  fig.suptitle(
      'Allegro throw — live / today: object $z$, advantage, episode length '
      r'($\times$ = actor-reset)',
      fontsize=12, fontweight='bold', y=0.995)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
