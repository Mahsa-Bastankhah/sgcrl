#!/usr/bin/env python3
"""Plot creative-5-task3 success across ablation setups (mean ± stderr).

Overlays completed catselect/extrew1 runs from ``logs/`` that have matching
``slurm/`` logs. Multiple seeds → mean curve with shaded ±1 standard error.

Legend lists only non-default knobs (e.g. nopermute+norand, extrew=1).
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

FIGS_DIR = os.path.join(base.FIGS_DIR, 'creative5_task3_setups')
GROUP = 'creative5_task3'

# Defaults from ppo_contrastive / BB job conventions. Only deviations go in
# the legend (permute+random start is the default — do not mention it).
_DEFAULT_MIN_STD = 0.01
_DEFAULT_EXTREW = False
_DEFAULT_PERMUTE = True
_DEFAULT_FIXED_X = -1.0
_DEFAULT_NORM_OBS = False
_DEFAULT_ANNEAL_ENT = False

LOG_DIRS: list[str] = [
    'ppo_builderbench_creative5_task3_e1024_pd_crl_tau05_actorreset_nopermute_norand_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task3_e1024_pd_crl_tau05_actorreset_nopermute_norand_minstd1e5_entanneal_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task3_e1024_pd_crl_tau05_actorreset_nopermute_norand_normobs_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task3_e1024_pd_crl_tau05_actorreset_permute_rand_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task3_e1024_pd_td3_logq_tau05_actorreset_nopermute_norand_minstd1e5_entanneal_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task3_e1024_pd_td3_logq_tau05_actorreset_nopermute_norand_normobs_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task3_e1024_pd_td3_logq_tau05_actorreset_permute_rand_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task3_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_norand_minstd1e5_entanneal_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task3_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_permute_rand_evalvid_catselect_extrew1',
]


def _cfg_get(cfg: dict, key: str):
  if key in cfg:
    return cfg[key]
  for nest in ('resolved_config', 'flags', 'ppo_env_defaults'):
    d = cfg.get(nest)
    if isinstance(d, dict) and key in d:
      return d[key]
  return None


def _load_run_config(log_dir_name: str) -> dict:
  base_dir = os.path.join(base.LOG_ROOT, log_dir_name)
  try:
    runs = sorted(os.listdir(base_dir))
  except OSError:
    return {}
  for run in runs:
    path = os.path.join(base_dir, run, 'run_config.json')
    if os.path.isfile(path):
      try:
        with open(path) as f:
          return json.load(f)
      except Exception:
        continue
  return {}


def _method_name(log_dir_name: str, cfg: dict | None = None) -> str:
  if '_td3_' in log_dir_name:
    return 'TD3'
  if '_nf_' in log_dir_name:
    return f'NF ({_nf_size_tag(log_dir_name, cfg or {})})'
  if '_crl_' in log_dir_name:
    return 'CRL'
  return 'PPO'


def _nf_size_tag(log_dir_name: str, cfg: dict) -> str:
  """e.g. sa3x256 r64 b6 w256 from config or dir name."""
  rep = _cfg_get(cfg, 'nf_rep_size')
  blocks = _cfg_get(cfg, 'nf_num_blocks')
  width = _cfg_get(cfg, 'nf_coupling_width')
  sa_h = _cfg_get(cfg, 'nf_sa_hidden')
  sa_l = _cfg_get(cfg, 'nf_sa_num_layers')
  # Fallback parse: ..._nf_compact_sa3x256_r64_b6_w256_...
  m = re.search(
      r'nf_compact_sa(?P<sa_l>\d+)x(?P<sa_h>\d+)_r(?P<rep>\d+)_b(?P<blocks>\d+)_w(?P<width>\d+)',
      log_dir_name,
  )
  if m:
    sa_l = sa_l if sa_l is not None else int(m.group('sa_l'))
    sa_h = sa_h if sa_h is not None else int(m.group('sa_h'))
    rep = rep if rep is not None else int(m.group('rep'))
    blocks = blocks if blocks is not None else int(m.group('blocks'))
    width = width if width is not None else int(m.group('width'))
  if None in (rep, blocks, width, sa_h, sa_l):
    return 'compact'
  return f'sa{int(sa_l)}x{int(sa_h)} r{int(rep)} b{int(blocks)} w{int(width)}'


def _fmt_float(x: float) -> str:
  if abs(x - round(x)) < 1e-12:
    return str(int(round(x)))
  s = f'{x:.6g}'
  return s


def _label_from_config(log_dir_name: str) -> str:
  """Build legend: method + non-default knobs only."""
  cfg = _load_run_config(log_dir_name)
  parts = [_method_name(log_dir_name, cfg)]

  permute = _cfg_get(cfg, 'builderbench_permute_start_boxes')
  if permute is None:
    permute = 'nopermute' not in log_dir_name
  fixed_x = _cfg_get(cfg, 'builderbench_fixed_start_x')
  if fixed_x is None:
    fixed_x = 0.1 if 'norand' in log_dir_name or 'fixedx' in log_dir_name else -1.0
  try:
    fixed_x = float(fixed_x)
  except (TypeError, ValueError):
    fixed_x = _DEFAULT_FIXED_X
  if (not bool(permute)) or fixed_x >= 0.0:
    tag = 'nopermute+norand'
    if fixed_x >= 0.0:
      tag += f' (fixed_x={_fmt_float(fixed_x)})'
    parts.append(tag)

  min_std = _cfg_get(cfg, 'ppo_actor_min_std')
  if min_std is None and 'minstd1e5' in log_dir_name:
    min_std = 1e-5
  try:
    min_std = float(min_std) if min_std is not None else _DEFAULT_MIN_STD
  except (TypeError, ValueError):
    min_std = _DEFAULT_MIN_STD
  if abs(min_std - _DEFAULT_MIN_STD) > 1e-12:
    parts.append(f'minstd={_fmt_float(min_std)}')

  anneal = _cfg_get(cfg, 'ppo_anneal_ent_coef')
  if anneal is None:
    anneal = 'entanneal' in log_dir_name
  if bool(anneal):
    final = _cfg_get(cfg, 'ppo_ent_coef_final')
    if final is None:
      parts.append('ent anneal')
    else:
      parts.append(f'ent anneal→{_fmt_float(float(final))}')

  norm_obs = _cfg_get(cfg, 'ppo_norm_obs')
  if norm_obs is None:
    norm_obs = 'normobs' in log_dir_name
  if bool(norm_obs):
    parts.append('obsnorm')

  use_ext = _cfg_get(cfg, 'ppo_use_external_reward')
  if use_ext is None:
    use_ext = 'extrew' in log_dir_name
  if bool(use_ext):
    scale = _cfg_get(cfg, 'ppo_external_reward_scale')
    if scale is None:
      m = re.search(r'extrew(\d+(?:\.\d+)?)', log_dir_name)
      scale = float(m.group(1)) if m else 1.0
    parts.append(f'extrew={_fmt_float(float(scale))}')

  # tau is shared across these runs; mention only if present in name / config
  # and useful for distinguishing (all c5t3 ablations use tau=0.5).
  return ', '.join(parts)


def _available_runs() -> list[tuple[str, str]]:
  out = []
  for name in LOG_DIRS:
    has_train = base._has_csv(base.LOG_ROOT, name, 'learner', min_size=200)
    has_eval = base._has_csv(base.LOG_ROOT, name, 'eval', min_size=100)
    if has_train or has_eval:
      label = _label_from_config(name)
      print(f'  label: {label}')
      out.append((name, label))
    else:
      print(f'  skip (no csv): {name}')
  return out


def _plot(runs: list[tuple[str, str]], *, split: str) -> str | None:
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker

  fig, ax = plt.subplots(figsize=(11.5, 5.2))
  plotted = 0
  for i, (log_dir_name, label) in enumerate(runs):
    if split == 'eval':
      seed_series = base._read_eval_seed_series(
          base.LOG_ROOT, log_dir_name, slurm_dir=base.SLURM_DIR)
      out_suffix = 'eval_success'
      title_metric = f'eval {base.EVAL_METRIC}'
      ylabel = 'Eval Success'
    elif split == 'train_mean':
      seed_series = base._read_csv_seed_series(
          base.LOG_ROOT, log_dir_name,
          split='learner', x_col='global_step', y_col='train_success_mean')
      out_suffix = 'train_success_mean'
      title_metric = 'train train_success_mean'
      ylabel = 'Train Success Mean'
    else:
      seed_series = base._read_train_seed_series(
          base.LOG_ROOT, log_dir_name, slurm_dir=base.SLURM_DIR)
      out_suffix = 'train_success1000'
      title_metric = f'train {base.TRAIN_METRIC}'
      ylabel = 'Train Success (last 1000)'

    xs, mean, se, n_seeds = base._aggregate_mean_stderr(seed_series)
    if not xs:
      print(f'  skip {label}: no {split} points')
      continue
    xs, mean, se = base._subsample_curve(xs, mean, se)
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    kwargs = dict(
        color=color, linewidth=2.0, label=f'{label} (n={n_seeds})',
        alpha=0.95, zorder=3 + i,
    )
    if split == 'eval':
      kwargs.update(
          marker=base.EVAL_MARKERS[i % len(base.EVAL_MARKERS)],
          markersize=6.5,
          linestyle=base.EVAL_LINESTYLES[i % len(base.EVAL_LINESTYLES)],
          markerfacecolor=color,
          markeredgecolor='white',
          markeredgewidth=0.6,
      )
    ax.plot(xs, mean, **kwargs)
    if n_seeds > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(
          xs, lo, hi, color=color, alpha=0.22, linewidth=0, zorder=2 + i)
    print(
        f'  {label}: n_seeds={n_seeds} pts={len(xs)} '
        f'y=[{min(mean):.3f}..{max(mean):.3f}] last={mean[-1]:.3f}'
    )
    plotted += 1

  if plotted == 0:
    plt.close(fig)
    return None

  ax.set_title(
      f'BuilderBench Creative 5 Task 3 — {title_metric} (mean ± stderr)',
      fontsize=12, fontweight='bold',
  )
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel(ylabel, fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  leg = ax.legend(
      loc='upper left', bbox_to_anchor=(1.01, 1.0),
      fontsize=9.5, framealpha=1.0, ncol=1,
      edgecolor='#333333', fancybox=False, borderpad=0.6,
      handlelength=2.4, labelspacing=0.4, borderaxespad=0.0,
  )
  leg.get_frame().set_facecolor('white')
  leg.get_frame().set_linewidth(1.2)

  os.makedirs(FIGS_DIR, exist_ok=True)
  out_path = os.path.join(FIGS_DIR, f'{GROUP}_{out_suffix}.png')
  fig.subplots_adjust(right=0.58)
  fig.savefig(
      out_path, dpi=160, bbox_inches='tight', bbox_extra_artists=(leg,))
  plt.close(fig)
  return out_path


def _plot_by_method(runs: list[tuple[str, str]], *, split: str) -> list[str]:
  """One panel per method so legends stay readable."""
  families = {
      'crl': 'CRL',
      'td3': 'TD3',
      'nf': 'NF compact',
  }
  outs = []
  for key, title in families.items():
    subset = [(n, l) for n, l in runs if f'_{key}_' in n or f'_{key}' in n]
    # NF dirs contain nf_compact; avoid matching nothing
    if key == 'nf':
      subset = [(n, l) for n, l in runs if '_nf_' in n]
    elif key == 'td3':
      subset = [(n, l) for n, l in runs if '_td3_' in n]
    else:
      subset = [(n, l) for n, l in runs if '_crl_' in n]
    if not subset:
      continue
    print(f'--- {title} / {split} ---')
    # temporarily reuse _plot with filtered runs; rewrite title via wrapper
    path = _plot_family(subset, split=split, family_title=title)
    if path:
      outs.append(path)
      print(f'  → {path}')
  return outs


def _plot_family(runs: list[tuple[str, str]], *, split: str,
                 family_title: str) -> str | None:
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker

  fig, ax = plt.subplots(figsize=(10.0, 4.8))
  plotted = 0
  for i, (log_dir_name, label) in enumerate(runs):
    if split == 'eval':
      seed_series = base._read_eval_seed_series(
          base.LOG_ROOT, log_dir_name, slurm_dir=base.SLURM_DIR)
      out_suffix = f'eval_success_{family_title.lower().replace(" ", "_")}'
      title_metric = f'eval {base.EVAL_METRIC}'
      ylabel = 'Eval Success'
    else:
      seed_series = base._read_train_seed_series(
          base.LOG_ROOT, log_dir_name, slurm_dir=base.SLURM_DIR)
      out_suffix = (
          f'train_success1000_{family_title.lower().replace(" ", "_")}'
      )
      title_metric = f'train {base.TRAIN_METRIC}'
      ylabel = 'Train Success (last 1000)'

    xs, mean, se, n_seeds = base._aggregate_mean_stderr(seed_series)
    if not xs:
      continue
    xs, mean, se = base._subsample_curve(xs, mean, se)
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    kwargs = dict(
        color=color, linewidth=2.2, label=f'{label} (n={n_seeds})',
        alpha=0.95, zorder=3 + i,
    )
    if split == 'eval':
      kwargs.update(
          marker=base.EVAL_MARKERS[i % len(base.EVAL_MARKERS)],
          markersize=7,
          linestyle=base.EVAL_LINESTYLES[i % len(base.EVAL_LINESTYLES)],
          markerfacecolor=color,
          markeredgecolor='white',
          markeredgewidth=0.6,
      )
    ax.plot(xs, mean, **kwargs)
    if n_seeds > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(
          xs, lo, hi, color=color, alpha=0.25, linewidth=0, zorder=2 + i)
    plotted += 1

  if plotted == 0:
    plt.close(fig)
    return None

  ax.set_title(
      f'BuilderBench Creative 5 Task 3 [{family_title}] — '
      f'{title_metric} (mean ± stderr)',
      fontsize=12, fontweight='bold',
  )
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel(ylabel, fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  leg = ax.legend(
      loc='upper left', bbox_to_anchor=(1.01, 1.0),
      fontsize=10, framealpha=1.0, ncol=1,
      edgecolor='#333333', fancybox=False, borderpad=0.6,
      handlelength=2.4, labelspacing=0.45, borderaxespad=0.0,
  )
  leg.get_frame().set_facecolor('white')
  leg.get_frame().set_linewidth(1.2)

  os.makedirs(FIGS_DIR, exist_ok=True)
  out_path = os.path.join(FIGS_DIR, f'{GROUP}_{out_suffix}.png')
  fig.subplots_adjust(right=0.62)
  fig.savefig(
      out_path, dpi=160, bbox_inches='tight', bbox_extra_artists=(leg,))
  plt.close(fig)
  return out_path


def main() -> None:
  runs = _available_runs()
  print(f'Plotting {len(runs)} c5t3 setups (one figure per method)')
  for key, title in [('crl', 'CRL'), ('td3', 'TD3'), ('nf', 'NF')]:
    if key == 'nf':
      subset = [(n, l) for n, l in runs if '_nf_' in n]
    elif key == 'td3':
      subset = [(n, l) for n, l in runs if '_td3_' in n]
    else:
      subset = [(n, l) for n, l in runs if '_crl_' in n]
    if not subset:
      continue
    print(f'--- {title} / train_success_mean ---')
    path = _plot_family_train_mean(subset, family_title=title)
    if path:
      print(f'  → {path}')
    print(f'--- {title} / eval_success ---')
    path = _plot_family(subset, split='eval', family_title=title)
    if path:
      print(f'  → {path}')


def _plot_family_train_mean(runs: list[tuple[str, str]], *,
                            family_title: str) -> str | None:
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker

  fig, ax = plt.subplots(figsize=(10.0, 4.8))
  plotted = 0
  for i, (log_dir_name, label) in enumerate(runs):
    seed_series = base._read_csv_seed_series(
        base.LOG_ROOT, log_dir_name,
        split='learner', x_col='global_step', y_col='train_success_mean')
    xs, mean, se, n_seeds = base._aggregate_mean_stderr(seed_series)
    if not xs:
      continue
    xs, mean, se = base._subsample_curve(xs, mean, se)
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    ax.plot(
        xs, mean, color=color, linewidth=2.2,
        label=f'{label} (n={n_seeds})', alpha=0.95, zorder=3 + i)
    if n_seeds > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(
          xs, lo, hi, color=color, alpha=0.25, linewidth=0, zorder=2 + i)
    plotted += 1
  if plotted == 0:
    plt.close(fig)
    return None

  ax.set_title(
      f'BuilderBench Creative 5 Task 3 [{family_title}] — '
      f'train success mean (mean ± stderr)',
      fontsize=12, fontweight='bold',
  )
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel('Train Success Mean', fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  leg = ax.legend(
      loc='upper left', bbox_to_anchor=(1.01, 1.0),
      fontsize=10, framealpha=1.0, ncol=1,
      edgecolor='#333333', fancybox=False, borderpad=0.6,
      handlelength=2.4, labelspacing=0.45, borderaxespad=0.0,
  )
  leg.get_frame().set_facecolor('white')
  leg.get_frame().set_linewidth(1.2)
  os.makedirs(FIGS_DIR, exist_ok=True)
  out_path = os.path.join(
      FIGS_DIR,
      f'{GROUP}_train_success_mean_{family_title.lower().replace(" ", "_")}.png',
  )
  fig.subplots_adjust(right=0.62)
  fig.savefig(
      out_path, dpi=160, bbox_inches='tight', bbox_extra_artists=(leg,))
  plt.close(fig)
  return out_path


if __name__ == '__main__':
  main()
