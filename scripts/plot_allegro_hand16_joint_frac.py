#!/usr/bin/env python3
"""Plot Allegro joint-fraction + MAE from learner/eval CSVs.

Eval (smoothed, window=5) when present:
  joint_frac_010 / joint_frac_020, mean_abs_joint_err, threshold success

Train (raw, no smooth):
  joint_frac @0.10/@0.20, hard success, and mean |q-q*|
  (prefer *_1000, else step_mean / train_success_mean)

Eval x-axis is iteration * (E * T).  Auto-detected from run_config.json
(ppo_num_envs * ppo_rollout_length) when present; override with
--steps-per-iter.

Sources (first match wins for eval):
  1) BraX seed CSV: logs/<dir>/*/logs/eval/logs.csv
  2) PPO-RND Allegro: logs/<dir>/eval_metrics.csv
     (keys like eval/joint_frac_010; hard success = eval/episode_success)

  python scripts/plot_allegro_hand16_joint_frac.py \\
    --log-dir <run_log_dir_name> --out figs/.../out.png --title '...'
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = (
    'ppo_allegro_hand16_control_sanity_e1024_nf_compact_small_200m_ep150_4h_jfrac')
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'hand16_joint_frac.png')
STEPS_PER_ITER = 1024 * 150  # E * ep_len

EVAL_COLS = [
    'joint_frac_010', 'joint_frac_020', 'mean_abs_joint_err',
    'success', 'easy_success', 'very_easy_success',
]
TRAIN_COLS = [
    'train_joint_frac_010_1000', 'train_joint_frac_020_1000',
    'train_joint_frac_010_step_mean', 'train_joint_frac_020_step_mean',
    'train_mean_abs_joint_err_1000',
    'train_mean_abs_joint_err_step_mean',
    'train_success_1000', 'train_success_mean',
]


def _infer_steps_per_iter(log_dir: str) -> int:
  """E * rollout T from run_config.json, else STEPS_PER_ITER (E=1024, T=150)."""
  root = os.path.join(base.LOG_ROOT, log_dir)
  if not os.path.isdir(root):
    return STEPS_PER_ITER
  for name in sorted(os.listdir(root)):
    cfg_path = os.path.join(root, name, 'run_config.json')
    if not os.path.isfile(cfg_path):
      continue
    try:
      cfg = json.load(open(cfg_path))
      flags = cfg.get('flags') or cfg.get('resolved_config') or cfg
      n_envs = int(flags.get('ppo_num_envs') or 0)
      ep = int(
          flags.get('ppo_rollout_length')
          or flags.get('isaacgym_episode_length')
          or 0)
      if n_envs > 0 and ep > 0:
        return n_envs * ep
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
      continue
  return STEPS_PER_ITER


def _seed_csv(split: str, log_dir: str) -> Optional[str]:
  root = os.path.join(base.LOG_ROOT, log_dir)
  if not os.path.isdir(root):
    return None
  for name in sorted(os.listdir(root)):
    path = os.path.join(root, name, 'logs', split, 'logs.csv')
    if os.path.isfile(path):
      return path
  return None


def _rnd_eval_csv(log_dir: str) -> Optional[str]:
  path = os.path.join(base.LOG_ROOT, log_dir, 'eval_metrics.csv')
  return path if os.path.isfile(path) else None


def _read_cols(path: str, x_col: str, y_cols: List[str],
               *, x_scale: float = 1.0
               ) -> Dict[str, Tuple[List[float], List[float]]]:
  out: Dict[str, Tuple[List[float], List[float]]] = {
      c: ([], []) for c in y_cols}
  with open(path, newline='') as fh:
    reader = csv.DictReader(fh)
    fields = set(reader.fieldnames or [])
    if x_col not in fields:
      return out
    for row in reader:
      try:
        x = float(row[x_col]) * x_scale
      except (TypeError, ValueError):
        continue
      for c in y_cols:
        if c not in fields:
          continue
        try:
          y = float(row[c])
        except (TypeError, ValueError):
          continue
        if y != y:  # NaN
          continue
        xs, ys = out[c]
        xs.append(x)
        ys.append(y)
  return out


def _read_rnd_eval(
    path: str, y_cols: List[str]
) -> Dict[str, Tuple[List[float], List[float]]]:
  """Read ppo-rnd Allegro eval_metrics.csv into the same keys as BraX eval."""
  # CSV uses eval/<name>; hard success is episode_success (levels['hard']).
  aliases = {
      'success': (
          'eval/episode_success', 'episode_success',
          'eval/success', 'success'),
      'easy_success': ('eval/easy_success', 'easy_success'),
      'very_easy_success': ('eval/very_easy_success', 'very_easy_success'),
      'joint_frac_010': ('eval/joint_frac_010', 'joint_frac_010'),
      'joint_frac_020': ('eval/joint_frac_020', 'joint_frac_020'),
      'mean_abs_joint_err': (
          'eval/mean_abs_joint_err', 'mean_abs_joint_err'),
  }
  out: Dict[str, Tuple[List[float], List[float]]] = {
      c: ([], []) for c in y_cols}
  with open(path, newline='') as fh:
    reader = csv.DictReader(fh)
    fields = set(reader.fieldnames or [])
    x_col = 'env_steps' if 'env_steps' in fields else None
    if x_col is None:
      return out
    for row in reader:
      try:
        x = float(row[x_col])
      except (TypeError, ValueError):
        continue
      for c in y_cols:
        src = None
        for cand in aliases.get(c, (c, f'eval/{c}')):
          if cand in fields:
            src = cand
            break
        if src is None:
          continue
        try:
          y = float(row[src])
        except (TypeError, ValueError):
          continue
        if y != y:
          continue
        xs, ys = out[c]
        xs.append(x)
        ys.append(y)
  return out


def run_once(
    *,
    log_dir: str,
    out: str,
    title: str,
    only_frac010: bool = False,
    steps_per_iter: Optional[int] = None,
) -> str:
  eval_path = _seed_csv('eval', log_dir)
  train_path = _seed_csv('learner', log_dir)
  rnd_path = _rnd_eval_csv(log_dir)
  if eval_path is None and train_path is None and rnd_path is None:
    raise SystemExit(
        f'no eval/learner/eval_metrics.csv under logs/{log_dir}/ yet')

  if steps_per_iter is None or int(steps_per_iter) <= 0:
    steps_per_iter = _infer_steps_per_iter(log_dir)
  print(f'steps_per_iter={int(steps_per_iter)} (eval x = iteration * this)')

  if eval_path is not None:
    eval_data = _read_cols(
        eval_path, 'iteration', EVAL_COLS, x_scale=float(steps_per_iter))
  elif rnd_path is not None:
    eval_data = _read_rnd_eval(rnd_path, EVAL_COLS)
    print(f'using RND eval_metrics.csv: {rnd_path}')
  else:
    eval_data = {c: ([], []) for c in EVAL_COLS}
    print(f'no eval CSV under logs/{log_dir}/ yet (train-only plot)')

  train_data = (
      _read_cols(train_path, 'global_step', TRAIN_COLS)
      if train_path else {c: ([], []) for c in TRAIN_COLS})

  C = base.ACCENT_COLORS
  fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.6), sharex=True)

  ax = axes[0]
  all_series = (
      ('joint_frac_010', 'eval joint_frac@0.10', C[0], '-'),
      ('joint_frac_020', 'eval joint_frac@0.20', C[1], '-'),
      ('success', 'eval hard (all ≤0.10)', C[3], '--'),
      ('easy_success', 'eval easy (all ≤0.20)', C[4], '--'),
      ('very_easy_success', 'eval very-easy (all ≤0.30)', C[2], ':'),
  )
  series = all_series[:1] if only_frac010 else all_series
  plotted_eval = False
  for key, label, color, ls in series:
    xs, ys = eval_data[key]
    if not xs:
      print(f'missing eval col: {key}')
      continue
    base._plot_eval_smoothed(
        ax, xs, ys, color=color, label=label, linestyle=ls)
    plotted_eval = True
    print(f'{key}: pts={len(xs)} last={ys[-1]:.4f} '
          f'peak={max(ys):.4f} at {xs[ys.index(max(ys))]/1e6:.1f}M')

  # Eval MAE on top panel twin when present.
  xs_eval_mae, ys_eval_mae = eval_data['mean_abs_joint_err']
  if xs_eval_mae and not only_frac010:
    ax_e2 = ax.twinx()
    sm = base._rolling_mean(ys_eval_mae)
    ax_e2.plot(xs_eval_mae, ys_eval_mae, color=C[5], lw=1.0, alpha=0.28)
    ax_e2.plot(
        xs_eval_mae, sm, color=C[5], lw=2.0,
        label='eval mean |q-q*| (rad)')
    ax_e2.set_ylabel('mean abs joint err (rad)', fontsize=10, color=C[5])
    ax_e2.spines[['top']].set_visible(False)
    print(f'mean_abs_joint_err: pts={len(xs_eval_mae)} '
          f'last={ys_eval_mae[-1]:.4f}')
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax_e2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc='best', fontsize=8,
              framealpha=0.95)
  elif plotted_eval:
    ax.legend(loc='best', fontsize=8, framealpha=0.95)
  else:
    ax.text(0.5, 0.5, 'no eval joint_frac yet',
            transform=ax.transAxes, ha='center', va='center')

  eval_title = (
      f'{title} — eval fraction of joints within 0.10 rad'
      if only_frac010 else
      f'{title} — eval joint fraction + MAE')
  ax.set_title(
      f'{eval_title} (rolling mean, window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=11, fontweight='bold')
  ax.set_ylabel(
      f'fraction / success (roll mean, w={base.EVAL_SMOOTH_WINDOW})',
      fontsize=10)
  ax.set_ylim(-0.02, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)

  ax = axes[1]
  # Prefer episode-max @1000; fall back to step mean early in the run.
  def _pick_train(pref: str, fallback: str):
    xs, ys = train_data[pref]
    if xs and any(y == y for y in ys):
      return pref, xs, ys
    return fallback, train_data[fallback][0], train_data[fallback][1]

  all_train_series = (
      ('train_joint_frac_010_1000', 'train_joint_frac_010_step_mean',
       'train joint_frac@0.10', C[0]),
      ('train_joint_frac_020_1000', 'train_joint_frac_020_step_mean',
       'train joint_frac@0.20', C[1]),
  )
  train_series = (
      all_train_series[:1] if only_frac010 else all_train_series)
  plotted = False
  for pref, fb, label, color in train_series:
    key, xs, ys = _pick_train(pref, fb)
    if not xs:
      print(f'missing train col: {pref}/{fb}')
      continue
    ax.plot(xs, ys, color=color, lw=2.0, label=f'{label} ({key})')
    plotted = True
    print(f'{key}: pts={len(xs)} last={ys[-1]:.4f}')

  if not only_frac010:
    succ_key, xs_s, ys_s = _pick_train(
        'train_success_1000', 'train_success_mean')
    if xs_s:
      ax.plot(xs_s, ys_s, color=C[3], lw=2.0, ls='--',
              label=f'train hard success ({succ_key})')
      plotted = True
      print(f'{succ_key}: pts={len(xs_s)} last={ys_s[-1]:.4f}')

  # Train MAE on twin axis (raw; prefer @1000, else step mean).
  xs_mae, ys_mae = ([], [])
  mae_key = ''
  if not only_frac010:
    mae_key, xs_mae, ys_mae = _pick_train(
        'train_mean_abs_joint_err_1000',
        'train_mean_abs_joint_err_step_mean')
  if xs_mae:
    ax2 = ax.twinx()
    ax2.plot(xs_mae, ys_mae, color=C[5], lw=2.0,
             label=f'train mean |q-q*| ({mae_key})')
    ax2.set_ylabel('mean abs joint err (rad)', fontsize=10, color=C[5])
    ax2.spines[['top']].set_visible(False)
    print(f'{mae_key}: pts={len(xs_mae)} last={ys_mae[-1]:.4f}')
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if plotted:
      ax.legend(handles + h2, labels + l2, loc='upper left', fontsize=8,
                framealpha=0.95)
    else:
      ax2.legend(loc='upper left', fontsize=8, framealpha=0.95)
  elif plotted:
    ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
  else:
    note = (
        'RND: no train joint_frac CSV (eval-only)'
        if rnd_path and not train_path else
        'no train joint_frac / MAE columns yet')
    ax.text(0.5, 0.5, note,
            transform=ax.transAxes, ha='center', va='center')

  ax.set_title(
      f'{title} — train fraction of joints within 0.10 rad (raw)'
      if only_frac010 else
      f'{title} — train joint fraction + MAE (raw)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('train joint fraction', fontsize=10)
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.set_ylim(-0.02, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  plt.close(fig)
  print(f'wrote {out}')
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--log-dir', default=LOG_DIR)
  parser.add_argument('--out', default=OUT)
  parser.add_argument('--title', default='hand16 NF')
  parser.add_argument(
      '--only-frac010', action='store_true',
      help='Plot only the fraction of joints within 0.10 rad.')
  parser.add_argument(
      '--steps-per-iter', type=int, default=0,
      help='Eval x-scale (E*T). 0 = auto from run_config.json.')
  parser.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds (CPU-only login-node watcher).')
  parser.add_argument(
      '--watch-max-sec', type=int, default=3600,
      help='Stop watcher after this many seconds (default 1h).')
  args = parser.parse_args()

  kwargs = dict(
      log_dir=args.log_dir, out=args.out, title=args.title,
      only_frac010=args.only_frac010,
      steps_per_iter=args.steps_per_iter)
  if args.watch <= 0:
    run_once(**kwargs)
    return

  t0 = time.time()
  print(
      f'[allegro_jfrac] watching every {args.watch}s '
      f'(max {args.watch_max_sec}s) → {args.out}',
      flush=True)
  while True:
    try:
      run_once(**kwargs)
    except SystemExit as exc:
      print(f'[allegro_jfrac] {exc}', flush=True)
    except Exception as exc:  # noqa: BLE001 — keep watcher alive
      print(f'[allegro_jfrac] error: {exc}', flush=True)
    if time.time() - t0 >= args.watch_max_sec:
      print('[allegro_jfrac] watch-max-sec reached; exiting', flush=True)
      break
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
