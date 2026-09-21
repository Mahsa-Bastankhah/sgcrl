#!/usr/bin/env python3
"""Train/eval success for the four T=150 new Allegro baselines (RND + MPO).

One figure per working recipe. NF working recipe is a reference series.
RND is eval-only (no train_success_1000). Eval is faint raw + bold roll
mean, window=5.

  python scripts/plot_allegro_new_baselines.py
  python scripts/plot_allegro_new_baselines.py --watch 120 --watch-max-sec 3600
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_allegro_hidetable_running_success as hidetable  # noqa: E402
import plot_allegro_keeparm_nf_mpo_rnd as rndplot  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, 'figs', 'allegro_kuka_throw')
SAC_CRL_LOG_ROOT = '/n/fs/mislresearch/sgcrl-sac-crl/logs'
SAC_CRL_SLURM_DIR = '/n/fs/mislresearch/sgcrl-sac-crl/slurm'
# Isolated SAC-CRL uses Acme TerminalLogger, not the host `[ppo] eval` line.
SAC_CRL_EVAL_RE = re.compile(
    r'\[Eval\].*?Global Step = (?P<step>[-+eE0-9.nanINF]+)'
    r'.*?\|\s*Success = (?P<success>[-+eE0-9.nanINF]+)'
)
EVAL_WINDOW = hidetable.EVAL_WINDOW
C = base.ACCENT_COLORS

# (title, out_path, ((label, log_dir, kind, color, ls), ...))
FAMILIES = (
    (
        'hard hover  near10  (0.45,-0.60)  z=0.55  in-bucket',
        os.path.join(FIG_DIR, 'akt_new_bl_hard_hover_success.png'),
        (
            (
                'NF  T=80 q40  300M  (working)',
                'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
                'nf', C[3], '--',
            ),
            (
                'RND  T=150  300M (old)',
                'new_allegrobaselines/rnd_hard_hover_near10_z055_ep150_300m_seed0_5h',
                'rnd', C[4], ':',
            ),
            (
                'MPO  T=150 fut80  300M (old)',
                'new_allegrobaselines/mpo_hard_hover_near10_z055_ep150_fut80_300m_seed0_14h',
                'mpo', C[5], ':',
            ),
            (
                'RND  T=150  500M',
                'new_allegrobaselines/rnd_hard_hover_near10_z055_ep150_500m_seed0_6h',
                'rnd', C[2], '-',
            ),
            (
                'MPO  T=150 fut80  500M',
                'new_allegrobaselines/mpo_hard_hover_near10_z055_ep150_fut80_500m_seed0_14h',
                'mpo', C[0], '-',
            ),
        ),
    ),
    (
        'mid hover  near10  (0.50,-0.45)  z=0.55  in-bucket',
        os.path.join(FIG_DIR, 'akt_new_bl_mid_hover_success.png'),
        (
            (
                'NF  T=80 q40  300M  (working)',
                'successful_candidate/mid_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
                'nf', C[9], '--',
            ),
            (
                'RND  T=150  300M (old)',
                'new_allegrobaselines/rnd_mid_hover_near10_z055_ep150_300m_seed0_5h',
                'rnd', C[4], ':',
            ),
            (
                'MPO  T=150 fut80  300M (old)',
                'new_allegrobaselines/mpo_mid_hover_near10_z055_ep150_fut80_300m_seed0_14h',
                'mpo', C[5], ':',
            ),
            (
                'RND  T=150  500M',
                'new_allegrobaselines/rnd_mid_hover_near10_z055_ep150_500m_seed0_6h',
                'rnd', C[2], '-',
            ),
            (
                'MPO  T=150 fut80  500M',
                'new_allegrobaselines/mpo_mid_hover_near10_z055_ep150_fut80_500m_seed0_14h',
                'mpo', C[0], '-',
            ),
        ),
    ),
    (
        'NVIDIA-init  easy  (0.50,-0.30)  z=0.55  T=150  in-bucket',
        os.path.join(FIG_DIR, 'akt_new_bl_nvidiainit_success.png'),
        (
            (
                'NF  T=150 fut80 q05  500M  (working)',
                (
                    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                    '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed0_8h',
                    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                    '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed1_8h',
                    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                    '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed2_8h',
                ),
                'nf', C[6], '--',
            ),
            (
                'RND  T=150  300M (old)',
                'new_allegrobaselines/rnd_nvidiainit_z055_ep150_300m_seed0_5h',
                'rnd', C[4], ':',
            ),
            (
                'MPO  T=150 fut80  300M (old)',
                'new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_300m_seed0_14h',
                'mpo', C[5], ':',
            ),
            (
                'RND  T=150  500M',
                (
                    'new_allegrobaselines/rnd_nvidiainit_z055_ep150_500m_seed0_6h',
                    'new_allegrobaselines/rnd_nvidiainit_z055_ep150_500m_seed1_6h',
                    'new_allegrobaselines/rnd_nvidiainit_z055_ep150_500m_seed2_6h',
                ),
                'rnd', C[2], '-',
            ),
            (
                'MPO  T=150 fut80  500M',
                (
                    'new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed0_14h',
                    'new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed1_14h',
                    'new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed2_14h',
                ),
                'mpo', C[0], '-',
            ),
            (
                'SGCRL  T=150  400M',
                'sac_crl_allegro_kuka_throw_e1024_nvidiainit_z055_ep150_400m_nofut_seed0_10h',
                'sac_crl', '#D62728', '-.',
            ),
            (
                'SGCRL  T=150 fut80',
                (
                    'sac_crl_allegro_kuka_throw_e1024_nvidiainit_z055_ep150_400m_fut80_seed0_10h',
                    'sac_crl_allegro_kuka_throw_e1024_nvidiainit_z055_ep150_500m_fut80_seed1_12h',
                ),
                'sac_crl', '#C51B8A', ':',
            ),
        ),
    ),
    (
        'NVIDIA-init  NF seed 2  (0.50,-0.30)  z=0.55  T=150  500M  32G',
        os.path.join(FIG_DIR, 'akt_new_bl_nvidiainit_seed2_success.png'),
        (
            (
                'NF  seed 2  T=150 fut80 q05  500M',
                'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed2_8h',
                'nf', C[6], '-',
            ),
        ),
    ),
)


def _sac_crl_slurm_logs(log_dir: str) -> list[str]:
  """Match isolated slurm files even when the job stem drops the walltime token."""
  if not os.path.isdir(SAC_CRL_SLURM_DIR):
    return []
  stem = re.sub(r'_\d+h$', '', log_dir)
  out = []
  for name in os.listdir(SAC_CRL_SLURM_DIR):
    if not name.endswith('.log'):
      continue
    if name.startswith(log_dir) or name.startswith(stem + '_') or name.startswith(stem + '.'):
      out.append(os.path.join(SAC_CRL_SLURM_DIR, name))
  return sorted(out)


def _read_sac_crl_eval(log_dir: str):
  """Prefer isolated slurm `[Eval]` rows; CSV often has only the first flush."""
  by_x: dict[int, float] = {}
  for path in _sac_crl_slurm_logs(log_dir):
    try:
      with open(path, 'r', errors='replace') as f:
        for line in f:
          m = SAC_CRL_EVAL_RE.search(line)
          if not m:
            continue
          by_x[int(float(m.group('step')))] = float(m.group('success'))
    except OSError:
      continue
  if by_x:
    return [[(x, by_x[x]) for x in sorted(by_x)]]
  return base._read_eval_seed_series(SAC_CRL_LOG_ROOT, log_dir)


def _as_dirs(log_dir):
  if isinstance(log_dir, (list, tuple)):
    return [d for d in log_dir if d]
  return [log_dir]


def _read_split(log_dir: str, kind: str, split: str):
  if kind == 'rnd':
    if split == 'train':
      return []
    pts = rndplot._read_rnd_eval(log_dir)
    return [pts] if pts else []
  if kind == 'sac_crl':
    # No train_success_1000; show greedy eval on both panels.
    return _read_sac_crl_eval(log_dir)
  if split == 'train':
    return base._read_train_seed_series(base.LOG_ROOT, log_dir)
  return base._read_eval_seed_series(base.LOG_ROOT, log_dir)


def _plot_family(title, methods, out_path):
  fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.6), sharex=True)
  missing = []
  sgcrl_notes = []
  for label, log_dir, kind, color, ls in methods:
    dirs = _as_dirs(log_dir)
    print(f'{title} | {label}  {dirs}', flush=True)
    train = []
    for d in dirs:
      train.extend(_read_split(d, kind, 'train'))
    tx, tmean, tse, tn = base._aggregate_mean_stderr(train)
    z = 20 if kind == 'sac_crl' else 3
    lw = 2.8 if kind == 'sac_crl' else 2.0
    if tx:
      if kind == 'sac_crl':
        hidetable._plot_train(axes[0], tx, tmean, tse, tn, color,
                              label + '  (eval)', ls=ls)
        axes[0].lines[-1].set_zorder(z)
        axes[0].lines[-1].set_linewidth(lw)
      else:
        hidetable._plot_train(axes[0], tx, tmean, tse, tn, color, label, ls=ls)
    elif kind == 'rnd':
      print(f'{label} train: {kind} eval-only', flush=True)
    else:
      missing.append(label)
      print(f'{label} train: no data', flush=True)
    ev = []
    for d in dirs:
      ev.extend(_read_split(d, kind, 'eval'))
    ex, emean, ese, en = base._aggregate_mean_stderr(ev)
    if ex:
      if kind == 'sac_crl':
        sm = base._plot_eval_smoothed(
            axes[1], ex, emean, color=color,
            label=f'{label}  (n={en}, roll mean w={EVAL_WINDOW})',
            linestyle=ls, window=EVAL_WINDOW, zorder=z, linewidth=lw)
        ys = sm if sm else emean
        print(f'{label} eval: n={en} last={ys[-1]:.4f} peak={max(ys):.4g} '
              f'pts={len(ex)} steps={ex[-1]/1e6:.1f}M', flush=True)
        sgcrl_notes.append(f'{label}: {ys[-1]:.3f} @ {ex[-1]/1e6:.1f}M')
      else:
        hidetable._plot_eval(axes[1], ex, emean, ese, en, color, label, ls=ls)
    else:
      print(f'{label} eval: no data', flush=True)
  axes[0].set_title(
      f'{title} — train success (raw last-1000; RND/SGCRL eval-only)',
      fontsize=12, fontweight='bold')
  hidetable._style(axes[0], 'train_success_1000')
  axes[0].legend(loc='upper left', fontsize=8.0, framealpha=0.95)
  axes[1].set_title(
      f'eval success  (faint raw + bold rolling mean, window={EVAL_WINDOW})',
      fontsize=11, fontweight='bold')
  hidetable._style(axes[1], f'eval success (roll mean, w={EVAL_WINDOW})',
                   xlabel=True)
  axes[1].legend(loc='upper left', fontsize=8.0, framealpha=0.95)
  if missing:
    axes[0].text(
        0.99, 0.02,
        'no logs yet:\n' + '\n'.join('• ' + m for m in missing),
        transform=axes[0].transAxes, ha='right', va='bottom',
        fontsize=7.4, color='#555555')
  if sgcrl_notes:
    note = 'SGCRL (greedy eval, still running)\n' + '\n'.join(sgcrl_notes)
    for ax in axes:
      ax.text(
          0.98, 0.16, note, transform=ax.transAxes, ha='right', va='bottom',
          fontsize=8.0, color='#B41E5C', zorder=30,
          bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                    edgecolor='#B41E5C', alpha=0.95))
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=160, bbox_inches='tight')
  extra = out_path.replace('nvidiainit_success.png', 'nvidiainit_sgcrl_success.png')
  if extra != out_path:
    fig.savefig(extra, dpi=160, bbox_inches='tight')
    print(f'→ {extra}', flush=True)
  plt.close(fig)
  print(f'→ {out_path}', flush=True)
  return out_path


def run_once() -> None:
  for title, path, methods in FAMILIES:
    _plot_family(title, methods, path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--watch', type=int, default=0)
  parser.add_argument('--watch-max-sec', type=int, default=0)
  args = parser.parse_args()
  if args.watch <= 0:
    run_once()
    return
  t0 = time.time()
  print(
      f'[new baselines] watching every {args.watch}s '
      f'(max {args.watch_max_sec or "∞"}s)',
      flush=True)
  while True:
    try:
      run_once()
    except Exception as exc:  # noqa: BLE001
      print(f'[new baselines] error: {exc}', flush=True)
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[new baselines] watch-max-sec reached; exiting', flush=True)
      return
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
