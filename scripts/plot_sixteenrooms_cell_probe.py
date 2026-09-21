#!/usr/bin/env python3
"""Plot and summarize one SixteenRooms fixed-cell probe run."""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _read_probe(path):
  with open(path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
  if not rows:
    raise ValueError(f'No probe rows in {path}')
  numeric = {}
  for name in rows[0]:
    numeric[name] = np.asarray([
        float(row[name]) if row[name] != '' else np.nan for row in rows
    ], dtype=np.float64)
  return numeric


def _slope_per_100k(steps, rewards):
  mask = np.isfinite(steps) & np.isfinite(rewards)
  if np.sum(mask) < 2 or np.ptp(steps[mask]) <= 0:
    return np.nan
  return float(np.polyfit(steps[mask], rewards[mask], 1)[0] * 100_000.0)


def _first_finite(values):
  finite = values[np.isfinite(values)]
  return float(finite[0]) if len(finite) else np.nan


def _fmt(value):
  return 'never' if not np.isfinite(value) else f'{value:g}'


def _summary(data, method):
  steps = data['global_step']
  near_reward = data['near_reward_raw']
  far_reward = data['far_reward_raw']
  near_first = _first_finite(data['near_first_visit_step'])
  far_first = _first_finite(data['far_first_visit_step'])
  far_pre = np.ones_like(steps, dtype=bool)
  far_post = np.zeros_like(steps, dtype=bool)
  if np.isfinite(far_first):
    far_pre = steps < far_first
    far_post = steps >= far_first

  near_occ = float(data['near_occupancy_steps'][-1])
  near_delta = float(near_reward[-1] - near_reward[0])
  return {
      'method': method,
      'near_cell': (
          f"({int(data['near_row'][0])},{int(data['near_col'][0])})"),
      'far_cell': f"({int(data['far_row'][0])},{int(data['far_col'][0])})",
      'final_global_step': int(steps[-1]),
      'near_first_visit_step': _fmt(near_first),
      'near_occupancy_steps': int(data['near_occupancy_steps'][-1]),
      'near_entry_count': int(data['near_entry_count'][-1]),
      'near_reward_initial': float(near_reward[0]),
      'near_reward_final': float(near_reward[-1]),
      'near_reward_delta': near_delta,
      'near_reward_slope_per_100k_steps': _slope_per_100k(
          steps, near_reward),
      'near_reward_delta_per_1000_occupancy_steps': (
          near_delta * 1000.0 / near_occ if near_occ > 0 else np.nan),
      'far_first_visit_step': _fmt(far_first),
      'far_occupancy_steps': int(data['far_occupancy_steps'][-1]),
      'far_entry_count': int(data['far_entry_count'][-1]),
      'far_reward_initial': float(far_reward[0]),
      'far_reward_at_last_previsit_probe': (
          float(far_reward[np.flatnonzero(far_pre)[-1]])
          if np.any(far_pre) else np.nan),
      'far_previsit_reward_slope_per_100k_steps': _slope_per_100k(
          steps[far_pre], far_reward[far_pre]),
      'far_postvisit_reward_slope_per_100k_steps': _slope_per_100k(
          steps[far_post], far_reward[far_post]),
      'far_reward_final': float(far_reward[-1]),
  }


def _write_summaries(path, summaries):
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)


def _first_env_success_step(learner_csv):
  """First global_step where batch mean env reward is > 0 (proxy for goal hit)."""
  if not learner_csv or not os.path.exists(learner_csv):
    return np.nan
  with open(learner_csv, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
      try:
        if float(row.get('reward_env_mean', 'nan')) > 0.0:
          return float(row['global_step'])
      except (TypeError, ValueError):
        continue
  return np.nan


def _near_occ_at_step(data, global_step):
  if not np.isfinite(global_step):
    return np.nan
  steps = np.asarray(data['global_step'], dtype=np.float64)
  occ = np.asarray(data['near_occupancy_steps'], dtype=np.float64)
  idx = np.flatnonzero(steps >= global_step)
  if len(idx) == 0:
    return np.nan
  return float(occ[idx[0]])


def _near_seen_vs_reward(data):
  """Keep one point per occupancy increase (x = times seen, y = raw reward)."""
  occ = np.asarray(data['near_occupancy_steps'], dtype=np.float64)
  rew = np.asarray(data['near_reward_raw'], dtype=np.float64)
  mask = np.isfinite(occ) & np.isfinite(rew)
  occ, rew = occ[mask], rew[mask]
  if len(occ) == 0:
    return occ, rew
  keep = np.ones(len(occ), dtype=bool)
  keep[1:] = occ[1:] > occ[:-1]
  return occ[keep], rew[keep]


def _method_color(method):
  return {
      'nf_compact_small': '#1f77b4',
      'crl': '#ff7f0e',
      'td3_logq': '#2ca02c',
      'tdinfonce': '#d62728',
      'nf_tiny': '#9467bd',
  }.get(method, None)


def _plot(data, method, output, learner_csv=None):
  near = (int(data['near_row'][0]), int(data['near_col'][0]))
  far = (int(data['far_row'][0]), int(data['far_col'][0]))
  far_first = _first_finite(data['far_first_visit_step'])
  success_step = _first_env_success_step(learner_csv)
  success_near_occ = _near_occ_at_step(data, success_step)
  near_x, near_y = _near_seen_vs_reward(data)
  step_m = data['global_step'] / 1e6
  color = _method_color(method)

  fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0))
  axes[0].plot(near_x, near_y, lw=2.0, color=color,
               label=f'near {near} raw reward')
  if np.isfinite(success_near_occ):
    axes[0].axvline(
        success_near_occ, color='black', ls='-.', lw=1.4,
        label=f'first goal success @ {success_near_occ:g} seen '
              f'({success_step:g} steps)')
  axes[0].set_xlabel(f'Near {near}: cumulative occupied steps (times seen)')
  axes[0].set_ylabel('Raw estimator reward')
  axes[0].legend()
  axes[0].grid(alpha=0.25)

  axes[1].plot(step_m, data['far_reward_raw'], lw=2.0, color=color,
               label=f'far {far} raw reward')
  if np.isfinite(far_first):
    axes[1].axvline(
        far_first / 1e6, color='black', ls='--', lw=1.2,
        label=f'far first visit @ {far_first:g} steps')
  else:
    axes[1].plot([], [], ' ', label='far cell: never visited')
  if np.isfinite(success_step):
    axes[1].axvline(
        success_step / 1e6, color='black', ls='-.', lw=1.4,
        label=f'first goal success @ {success_step:g} steps')
  axes[1].set_xlabel('Training environment steps (millions)')
  axes[1].set_ylabel('Raw estimator reward')
  axes[1].legend()
  axes[1].grid(alpha=0.25)

  fig.suptitle(
      f'{method}: seen vs unseen SixteenRooms cell probe\n'
      'Near: reward vs times seen; far: reward vs steps + first visit')
  fig.tight_layout()
  fig.savefig(output, dpi=180, bbox_inches='tight')
  plt.close(fig)


def _plot_near_panel(series, method_set, title, output, success_by_method,
                     yscale='linear'):
  fig, ax = plt.subplots(figsize=(8.0, 5.2))
  for method, data in series:
    if method not in method_set:
      continue
    color = _method_color(method)
    near_x, near_y = _near_seen_vs_reward(data)
    ax.plot(near_x, near_y, lw=2.0, color=color, label=method)
    success_step = success_by_method.get(method, np.nan)
    success_near_occ = _near_occ_at_step(data, success_step)
    if np.isfinite(success_near_occ):
      ax.axvline(
          success_near_occ, color=color, ls='-.', lw=1.5,
          label=f'{method} first success @ {success_near_occ:g} seen')
    else:
      ax.plot([], [], color=color, ls='-.', lw=1.0,
              label=f'{method}: no goal success')
  ax.set_title(title)
  ax.set_xlabel('Near (2,0): times seen (occupied steps)')
  ax.set_ylabel('Near (2,0) raw reward')
  if yscale == 'symlog':
    ax.set_yscale('symlog', linthresh=1.0)
  elif yscale != 'linear':
    ax.set_yscale(yscale)
  ax.grid(alpha=0.25)
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(output, dpi=180, bbox_inches='tight')
  plt.close(fig)


def _plot_far_panel(series, method_set, title, output, success_by_method,
                    yscale='linear'):
  fig, ax = plt.subplots(figsize=(8.0, 5.2))
  for method, data in series:
    if method not in method_set:
      continue
    color = _method_color(method)
    step_m = data['global_step'] / 1e6
    ax.plot(step_m, data['far_reward_raw'], lw=2.0, color=color, label=method)
    far_first = _first_finite(data['far_first_visit_step'])
    if np.isfinite(far_first):
      ax.axvline(
          far_first / 1e6, color=color, ls='--', lw=1.4,
          label=f'{method} far first visit @ {far_first:g}')
    else:
      ax.plot([], [], color=color, ls=':', lw=1.0,
              label=f'{method}: far not visited')
    success_step = success_by_method.get(method, np.nan)
    if np.isfinite(success_step):
      ax.axvline(
          success_step / 1e6, color=color, ls='-.', lw=1.5,
          label=f'{method} first success @ {success_step:g}')
    else:
      ax.plot([], [], color=color, ls='-.', lw=1.0,
              label=f'{method}: no goal success')
  ax.set_title(title)
  ax.set_xlabel('Training environment steps (millions)')
  ax.set_ylabel('Far (0,20) raw reward')
  if yscale == 'symlog':
    ax.set_yscale('symlog', linthresh=1.0)
  elif yscale != 'linear':
    ax.set_yscale(yscale)
  ax.grid(alpha=0.25)
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(output, dpi=180, bbox_inches='tight')
  plt.close(fig)


def _plot_comparison(series, output, success_by_method):
  """Write separate PNGs so reward scales stay readable."""
  out_dir = os.path.dirname(output) or '.'
  panels = (
      ('near_nf', {'nf_compact_small', 'nf_tiny'}, 'near', 'symlog',
       'Near (2,0): reward vs times seen — NF (symlog y)\n'
       'dash-dot = first goal success (from env reward)'),
      ('near_crl_tdinfonce', {'crl', 'tdinfonce'}, 'near', 'linear',
       'Near (2,0): reward vs times seen — CRL / TD-InfoNCE\n'
       'dash-dot = first goal success (from env reward)'),
      ('near_td3', {'td3_logq'}, 'near', 'linear',
       'Near (2,0): reward vs times seen — TD3 log-Q\n'
       'dash-dot = first goal success (from env reward)'),
      ('far_nf', {'nf_compact_small', 'nf_tiny'}, 'far', 'symlog',
       'Far (0,20): reward vs steps — NF (symlog y)\n'
       'dashed = far first visit; dash-dot = first goal success'),
      ('far_crl_tdinfonce', {'crl', 'tdinfonce'}, 'far', 'linear',
       'Far (0,20): reward vs steps — CRL / TD-InfoNCE\n'
       'dashed = far first visit; dash-dot = first goal success'),
      ('far_td3', {'td3_logq'}, 'far', 'linear',
       'Far (0,20): reward vs steps — TD3 log-Q\n'
       'dashed = far first visit; dash-dot = first goal success'),
  )
  paths = {}
  for key, methods, kind, yscale, title in panels:
    path = os.path.join(out_dir, f'cell_probe_{key}.png')
    paths[key] = path
    if kind == 'near':
      _plot_near_panel(
          series, methods, title, path, success_by_method, yscale=yscale)
    else:
      _plot_far_panel(
          series, methods, title, path, success_by_method, yscale=yscale)
  for stale in (
      'cell_probe_near_crl_td3_tdinfonce.png',
      'cell_probe_far_crl_td3_tdinfonce.png',
  ):
    stale_path = os.path.join(out_dir, stale)
    if os.path.exists(stale_path):
      os.remove(stale_path)
  return paths


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--probe-csv')
  parser.add_argument(
      '--log-root',
      help='Aggregate the five standard method subdirectories instead.')
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--method')
  parser.add_argument(
      '--learner-csv',
      help='Optional learner logs.csv for first-goal-success marker.')
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  if args.log_root:
    methods = ('nf_compact_small', 'crl', 'td3_logq', 'tdinfonce', 'nf_tiny')
    series = []
    summaries = []
    success_by_method = {}
    for method in methods:
      run_dir = os.path.join(
          args.log_root, method, 'ppo_point_SixteenRooms_0')
      path = os.path.join(run_dir, 'cell_probe.csv')
      learner = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
      data = _read_probe(path)
      series.append((method, data))
      summaries.append(_summary(data, method))
      success_by_method[method] = _first_env_success_step(learner)
    summary_path = os.path.join(
        args.output_dir, 'cell_probe_all_methods_summary.csv')
    plot_path = os.path.join(
        args.output_dir, 'cell_probe_all_methods.png')
    _write_summaries(summary_path, summaries)
    plot_paths = _plot_comparison(series, plot_path, success_by_method)
    print(f'Wrote {summary_path}')
    for key, path in plot_paths.items():
      print(f'Wrote {key}: {path}')
    for method, step in success_by_method.items():
      print(f'first_goal_success[{method}]={_fmt(step)}')
    return

  if not args.probe_csv or not args.method:
    parser.error(
        'single-run mode requires both --probe-csv and --method; '
        'otherwise use --log-root')
  data = _read_probe(args.probe_csv)
  summary = _summary(data, args.method)
  summary_path = os.path.join(args.output_dir, 'cell_probe_summary.csv')
  plot_path = os.path.join(args.output_dir, 'cell_probe.png')
  learner_csv = args.learner_csv
  if learner_csv is None:
    # Default: <run>/logs/learner/logs.csv when probe is <run>/cell_probe.csv
    run_dir = os.path.dirname(os.path.abspath(args.probe_csv))
    candidate = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
    if os.path.exists(candidate):
      learner_csv = candidate
  _write_summaries(summary_path, [summary])
  _plot(data, args.method, plot_path, learner_csv=learner_csv)
  print(f'Wrote {summary_path}')
  print(f'Wrote {plot_path}')
  print(f'first_goal_success={_fmt(_first_env_success_step(learner_csv))}')
  for key, value in summary.items():
    print(f'{key}={value}')


if __name__ == '__main__':
  main()
