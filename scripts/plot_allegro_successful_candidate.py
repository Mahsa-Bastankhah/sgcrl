#!/usr/bin/env python3
"""Train/eval success for successful-candidate Allegro groups, one figure each.

Families: hard hover, mid hover, hard arm-init, NVIDIA-init.
Other NF jobs stay on ``akt_live_queue_success.png``.

  python scripts/plot_allegro_successful_candidate.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_allegro_hidetable_running_success as hidetable  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, 'figs', 'allegro_kuka_throw')
OUT_PATH = os.path.join(FIG_DIR, 'akt_sc_hard_hover_success.png')
EVAL_WINDOW = hidetable.EVAL_WINDOW
C = base.ACCENT_COLORS

# One figure per family. Solid = parent (default NF goal z). Dashed = z=0.55.
# Parent ``1B`` is the step budget in the log-dir name, not episode length.
FAMILIES = (
    (
        'hard hover  near10  bucket (0.45,-0.60)  q40  in-bucket',
        os.path.join(FIG_DIR, 'akt_sc_hard_hover_success.png'),
        (
            (
                'z=0.55  T=80 q40  300M  s0',
                'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
                C[3],
                '--',
            ),
            (
                'z=0.55  T=80 q40  300M  s1',
                'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed1_4h',
                C[7],
                '--',
            ),
            (
                'z=0.55  T=80 q40  300M  s2',
                'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed2_4h',
                C[11],
                '--',
            ),
        ),
    ),
    (
        'mid hover  near10  bucket (0.50,-0.45)  in-bucket',
        os.path.join(FIG_DIR, 'akt_sc_mid_hover_success.png'),
        (
            (
                'z=0.55  T=80 q40  300M',
                'successful_candidate/mid_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
                C[9],
                '-',
            ),
        ),
    ),
    (
        'hard arm-init  keep-arm  bucket (0.45,-0.60)  q40  in-bucket',
        os.path.join(FIG_DIR, 'akt_sc_hard_arminit_success.png'),
        (
            (
                'parent  T=70 q40  1B-budget  goal z≈0.45',
                'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
                '_nvbucket_xy045m060_norand_ep70_1b_q40shift_noreplace_inbucket_seed0_14h',
                C[4],
                '-',
            ),
            (
                'z=0.55  T=80 q40  300M',
                'successful_candidate/hard_bucket_arm_init_z055_ep80_q40_300m_seed0_4h',
                C[1],
                '--',
            ),
        ),
    ),
    (
        'NVIDIA-init  bucket (0.50,-0.30)  q05  in-bucket',
        os.path.join(FIG_DIR, 'akt_sc_nvidiainit_success.png'),
        (
            (
                'T=100 fut50  1B  rand-init  z≈0.45  s0',
                'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                '_nvbucket_xy050030_noshape_ep100_fut50_1b_q05shift_noreplace_inbucket_seed0_14h',
                C[2],
                '-',
            ),
            (
                'T=100 fut50  1B  rand-init  z≈0.45  s1',
                'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                '_nvbucket_xy050030_noshape_ep100_fut50_1b_q05shift_noreplace_inbucket_seed1_14h',
                C[8],
                '-',
            ),
            (
                'T=100 fut50  500M  norand  z≈0.45  s0',
                'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                '_nvbucket_xy050030_noshape_ep100_fut50_500m_q05shift_noreplace_inbucket_norand_seed0_7h',
                C[5],
                '-',
            ),
            (
                'T=150 fut80  500M  norand  z=0.55  s0',
                'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed0_8h',
                C[6],
                '--',
            ),
        ),
    ),
    (
        'working recipes  (z=0.55 kids that solved)',
        os.path.join(FIG_DIR, 'akt_sc_working_recipes_success.png'),
        (
            (
                'hover-near10  hard (0.45,-0.60)  T=80 q40  s0',
                'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
                C[3],
                '-',
            ),
            (
                'hover-near10  mid (0.50,-0.45)  T=80 q40  s0',
                'successful_candidate/mid_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
                C[9],
                '-',
            ),
            (
                'keep-arm  hard (0.45,-0.60)  T=80 q40  s0',
                'successful_candidate/hard_bucket_arm_init_z055_ep80_q40_300m_seed0_4h',
                C[1],
                '-',
            ),
            (
                'NVIDIA-init  easy (0.50,-0.30)  T=150 fut80 q05  norand',
                'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed0_8h',
                C[6],
                '-',
            ),
        ),
    ),
    (
        'hover-near10  hard (0.45,-0.60)  T=150 fut80 q40  500M  z=0.55',
        os.path.join(FIG_DIR, 'akt_sc_hard_hover_ep150_fut80_500m_success.png'),
        (
            (
                's0',
                'successful_candidate/hard_bucket_hover_near10_z055_ep150_fut80_q40_500m_seed0_10h',
                C[3],
                '-',
            ),
            (
                's1',
                'successful_candidate/hard_bucket_hover_near10_z055_ep150_fut80_q40_500m_seed1_10h',
                C[7],
                '--',
            ),
        ),
    ),
    (
        'hover-near10  mid (0.50,-0.45)  T=150 fut80 q40  500M  z=0.55',
        os.path.join(FIG_DIR, 'akt_sc_mid_hover_ep150_fut80_500m_success.png'),
        (
            (
                's0',
                'successful_candidate/mid_bucket_hover_near10_z055_ep150_fut80_q40_500m_seed0_10h',
                C[9],
                '-',
            ),
            (
                's1',
                'successful_candidate/mid_bucket_hover_near10_z055_ep150_fut80_q40_500m_seed1_10h',
                C[0],
                '--',
            ),
        ),
    ),
    (
        'keep-arm  hard (0.45,-0.60)  T=150 fut80 q40  500M  z=0.55',
        os.path.join(FIG_DIR, 'akt_sc_hard_arminit_ep150_fut80_500m_success.png'),
        (
            (
                's0',
                'successful_candidate/hard_bucket_arm_init_z055_ep150_fut80_q40_500m_seed0_10h',
                C[1],
                '-',
            ),
            (
                's1',
                'successful_candidate/hard_bucket_arm_init_z055_ep150_fut80_q40_500m_seed1_10h',
                C[4],
                '--',
            ),
        ),
    ),
    (
        'NVIDIA-init  easy (0.50,-0.30)  T=150 fut80 q05  500M  z=0.55',
        os.path.join(FIG_DIR, 'akt_sc_nvidiainit_ep150_fut80_500m_success.png'),
        (
            (
                's0  existing',
                'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
                '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed0_8h',
                C[6],
                '-',
            ),
            (
                's1',
                'successful_candidate/nvidiainit_easy_z055_ep150_fut80_q05_500m_seed1_10h',
                C[2],
                '--',
            ),
        ),
    ),
)


def _expand_log_dirs(log_dir: str) -> list[str]:
  if '*' not in log_dir:
    return [log_dir]
  matches = sorted(glob.glob(os.path.join(base.LOG_ROOT, log_dir)))
  rel = [os.path.relpath(p, base.LOG_ROOT) for p in matches if os.path.isdir(p)]
  return rel or [log_dir]


def _read_split(log_dir: str, split: str):
  series = []
  for d in _expand_log_dirs(log_dir):
    if split == 'train':
      series.extend(base._read_train_seed_series(base.LOG_ROOT, d))
    else:
      series.extend(base._read_eval_seed_series(base.LOG_ROOT, d))
  return series


def _plot_family(title: str, methods, out_path: str) -> str:
  fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.6), sharex=True)
  missing = []
  for label, log_dir, color, ls in methods:
    print(f'{title} | {label}  {log_dir}', flush=True)
    train = _read_split(log_dir, 'train')
    tx, tmean, tse, tn = base._aggregate_mean_stderr(train)
    if tx:
      hidetable._plot_train(axes[0], tx, tmean, tse, tn, color, label, ls=ls)
    else:
      missing.append(label)
      print(f'{label} train: no data', flush=True)
    ev = _read_split(log_dir, 'eval')
    ex, emean, ese, en = base._aggregate_mean_stderr(ev)
    if ex:
      hidetable._plot_eval(axes[1], ex, emean, ese, en, color, label, ls=ls)
    else:
      print(f'{label} eval: no data', flush=True)
  axes[0].set_title(
      f'{title} — train success (raw last-1000)',
      fontsize=12, fontweight='bold')
  hidetable._style(axes[0], 'train_success_1000')
  axes[0].legend(loc='upper left', fontsize=8.2, framealpha=0.95)
  axes[1].set_title(
      f'eval success  (faint raw + bold rolling mean, window={EVAL_WINDOW})',
      fontsize=11, fontweight='bold')
  hidetable._style(axes[1], f'eval success (roll mean, w={EVAL_WINDOW})',
                   xlabel=True)
  axes[1].legend(loc='upper left', fontsize=8.2, framealpha=0.95)
  if missing:
    axes[0].text(
        0.99, 0.02,
        'no logs yet:\n' + '\n'.join('• ' + m for m in missing),
        transform=axes[0].transAxes, ha='right', va='bottom',
        fontsize=7.4, color='#555555')
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}', flush=True)
  return out_path


def run_once(out_path: str = OUT_PATH) -> str:
  last = out_path
  for title, path, methods in FAMILIES:
    last = _plot_family(title, methods, path)
  try:
    import plot_allegro_successful_candidate_nf_diag as diag  # noqa: E402
    diag.run_once()
  except Exception as exc:  # noqa: BLE001
    print(f'nf-diag plot skipped: {exc}', flush=True)
  return last


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
      f'[successful candidate] watching every {args.watch}s '
      f'(max {args.watch_max_sec or "∞"}s) → {FIG_DIR}/akt_sc_*_success.png',
      flush=True)
  while True:
    try:
      run_once()
    except Exception as exc:  # noqa: BLE001
      print(f'[successful candidate] error: {exc}', flush=True)
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[successful candidate] watch-max-sec reached; exiting',
            flush=True)
      return
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
