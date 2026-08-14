#!/usr/bin/env python3
"""Pick best NF / CRL / TD3 / TD-InfoNCE run per BuilderBench creative task.

For each task in {c3t1,c4t1,c4t2,c4t3,c5t2,c5t3,c5t4,c7t2} and each density
estimator family, ranks historical ``logs/ppo_builderbench_*`` configs by a
**late-training success** score (not mid-run peaks):

  Primary score
  -------------
  Per seed:
    late_mean = mean(train_success_1000 over last LATE_FRAC of step span)
                (fallback: last LATE_MIN_PTS points if the %-window is thin)
    end_mean  = mean(train_success_1000 over last END_PTS logged points)
    seed_score = LATE_WEIGHT * late_mean + END_WEIGHT * end_mean
                 (defaults: 0.5 / 0.5)
  Across seeds: mean ± stderr of those per-seed scores.

  Secondary (reported, used only for tie-breaks / ambiguity flags)
  ----------------------------------------------------------------
  - Per-seed peak ``train_success_1000`` (mean ± stderr)
  - Late-window within-seed std (stability inside the window)
  - Spike/collapse flag: peak − late_mean or late_mean − end_mean large

Clear winners are plotted against each other; ambiguous near-ties are flagged
for a human decision unless locked in ``USER_OVERRIDES`` (resolved picks are
not re-asked on subsequent runs).

Displayed train/eval curves use a centered **time** moving average (window
in env steps; default 10M train / 20M eval). Scores in the legend are still
computed on the raw series.

Writes:
  figs/builderbench/active_train_eval/best_method_compare/
    {task}_best_nf_crl_td3_tdinfonce_train_eval.png
    summary.csv
    ambiguities.md
    picks.json
    BEST_RUNS.md   (human-readable reference; flags n_seeds==1)

Usage:
  python scripts/plot_builderbench_best_method_compare.py
  python scripts/plot_builderbench_best_method_compare.py --watch 1800
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

TASKS = [
    'creative3_task1',
    'creative4_task1',
    'creative4_task2',
    'creative4_task3',
    'creative5_task2',
    'creative5_task3',
    'creative5_task4',
    'creative7_task2',
]
# Human-readable BuilderBench task names for plot titles / overview.
TASK_DISPLAY_NAMES = {
    'creative3_task1': 'Creative 3 Task 1',
    'creative4_task1': 'Creative 4 Task 1',
    'creative4_task2': 'two towers',
    'creative4_task3': 'maximum overhang 4',
    'creative5_task2': 'pyramid',
    'creative5_task3': 'underspecified towers',
    'creative5_task4': 'maximum overhang 5',
    'creative7_task2': 'Creative 7 Task 2',
}
METHODS = ('nf', 'crl', 'td3', 'tdinfonce')
METHOD_TITLE = {
    'nf': 'NF',
    'crl': 'CRL',
    'td3': 'TD3',
    'tdinfonce': 'TD-InfoNCE',
}


def _task_display(task: str) -> str:
  return TASK_DISPLAY_NAMES.get(task, task.replace('_', ' '))


METHOD_COLORS = {
    'nf': '#4C9BE8',
    'crl': '#E8834C',
    'td3': '#4CE87A',
    'tdinfonce': '#E84C6F',
}
EXCLUDE_SUBSTR = (
    'frozen_crl_reward',
    'frozen_nf_reward',
    'frozen_td3',
    'frozen_tdinfonce',
    'reward_probe',
)
OUT_DIR = os.path.join(base.FIGS_DIR, 'best_method_compare')

# Drop in-progress extra seeds whose last logged x is far behind the longest
# seed of the same config. Otherwise late/end scoring treats mid-run as "end"
# and mean±SE bands get pulled down by unfinished jobs.
MIN_SEED_HORIZON_FRAC = 0.90

# Display-only centered moving average over env-step time (not point count).
# Eval is sparser/noisier so it uses a wider window. 0 disables.
SMOOTH_WINDOW_TRAIN = 10_000_000
SMOOTH_WINDOW_EVAL = 20_000_000

# Late-window scoring: last LATE_FRAC of each seed's x-span; if that yields
# fewer than LATE_MIN_PTS points, take the last LATE_MIN_PTS logged points.
# End plateau = mean of the last END_PTS logged points (true end of training).
LATE_FRAC = 0.15
LATE_MIN_PTS = 5
END_PTS = 5
# Per-seed primary = LATE_WEIGHT * late_window_mean + END_WEIGHT * end_plateau.
LATE_WEIGHT = 0.5
END_WEIGHT = 0.5

# Ambiguity heuristics (on seed-mean primary late/end score).
NEAR_TIE = 0.03
SATURATED = 0.95
HIGH_SEED_STD = 0.30
MULTI_SEED_BONUS_GAP = 0.15  # single-seed leader vs multi-seed runner-up
FAILED = 0.05  # all-failed families: no human ask needed
MIN_ALT_SCORE = 0.10  # don't call near-zero alts "more consistent"
# Peak much higher than late/end → spike-then-collapse / unstable plateau.
SPIKE_COLLAPSE_GAP = 0.20
# Peak vs late disagreement between leader and a competitor.
PEAK_LATE_DISAGREE = 0.10
# Late mean high but end plateau collapsed (or vice versa) within a seed.
END_COLLAPSE_GAP = 0.20

# Locked human decisions (Aug 2026 late/end ranking). Force these runs as the
# pick and suppress re-flagging as ambiguous. ``collapsed_reference`` keeps a
# high-peak mid-train spike that collapsed by the end, for plot reference only.
USER_OVERRIDES: dict[tuple[str, str], dict] = {
    # --- resolved ambiguous cases ---
    ('creative4_task1', 'td3'): {
        'name': (
            'ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_'
            'actorreset_nopermute_catselect_minstd1e4_fixedx01_extrew1'
        ),
        'note': 'user: keep provisional near-tie winner',
    },
    ('creative4_task1', 'tdinfonce'): {
        'name': (
            'ppo_builderbench_creative4_task1_e1024_pd_tdinfonce_tau05_'
            'actorreset_nopermute_catselect_minstd1e4_extrew1'
        ),
        'note': 'user: keep provisional near-tie winner',
    },
    ('creative4_task2', 'crl'): {
        'name': (
            'ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_'
            'catselect_stateonly_extrew1'
        ),
        'note': 'user: keep provisional n=1 saturated over multi-seed alt',
    },
    ('creative5_task3', 'td3'): {
        'name': (
            'ppo_builderbench_creative5_task3_e1024_pd_td3_logq_tau05_'
            'actorreset_permute_rand_evalvid_catselect_extrew1'
        ),
        'note': 'user: KEEP inconsistent multi-seed run (do not mark failed)',
    },
    ('creative5_task4', 'crl'): {
        'name': (
            'ppo_builderbench_creative5_task4_e1024_pd_crl_tau05_'
            'actorreset_nopermute_norand_normobs_evalvid_catselect_extrew1'
        ),
        'note': 'user: KEEP mean~0.5 (seeds 0 and 1); do not mark failed',
    },
    ('creative7_task2', 'crl'): {
        'name': (
            'ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_'
            'nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl10'
        ),
        'note': (
            'user: KEEP high-peak collapsed run for reference '
            '(peak≈0.94; late/end≈0)'
        ),
        'collapsed_reference': True,
    },
    # --- lock remaining clear picks ("the rest is good as it is") ---
    ('creative3_task1', 'nf'): {
        'name': 'ppo_builderbench_creative3_task1_e1024_pd_nf_tau05_catselect_extrew1',
        'note': 'user: lock clear pick',
    },
    ('creative3_task1', 'crl'): {
        'name': 'ppo_builderbench_creative3_task1_e1024_pd_crl_tau05_catselect_extrew1',
        'note': 'user: lock clear pick',
    },
    ('creative3_task1', 'td3'): {
        'name': 'ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05_catselect_extrew1',
        'note': 'user: lock clear pick',
    },
    ('creative3_task1', 'tdinfonce'): {
        'name': 'ppo_builderbench_creative3_task1_e1024_pd_tdinfonce_tau05_catselect',
        'note': 'user: lock clear pick',
    },
    ('creative4_task1', 'nf'): {
        'name': (
            'ppo_builderbench_creative4_task1_e1024_pd_nf_tau05_'
            'actorreset_nopermute_catselect_minstd1e4_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative4_task1', 'crl'): {
        'name': (
            'ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_'
            'actorreset_nopermute_norand_normobs_catselect_minstd1e4_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative4_task2', 'nf'): {
        'name': (
            'ppo_builderbench_creative4_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_'
            'tau05_actorreset_nopermute_fixedx01_catselect_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative4_task2', 'td3'): {
        'name': 'ppo_builderbench_creative4_task2_e1024_pd_td3_logq_tau05_catselect_extrew1',
        'note': 'user: lock clear pick',
    },
    ('creative4_task2', 'tdinfonce'): {
        'name': 'ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_catselect_extrew1',
        'note': 'user: lock clear pick',
    },
    ('creative4_task3', 'nf'): {
        'name': (
            'ppo_builderbench_creative4_task3_e1024_pd_nf_compact_sa3x256_r64_b6_w256_'
            'tau05_actorreset_nopermute_norand_minstd1e5_entanneal_evalvid_catselect_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative4_task3', 'crl'): {
        'name': (
            'ppo_builderbench_creative4_task3_e1024_pd_crl_tau05_'
            'actorreset_nopermute_norand_normobs_evalvid_catselect_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative5_task2', 'nf'): {
        'name': (
            'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_'
            'tau05_actorreset_nopermute_norand_evalvid_catselect_stateonly_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative5_task2', 'crl'): {
        'name': (
            'ppo_builderbench_creative5_task2_e1024_pd_crl_tau05_'
            'actorreset_evalvid_catselect_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative5_task2', 'td3'): {
        'name': (
            'ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_'
            'actorreset_nopermute_normobs_evalvid_catselect'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative5_task2', 'tdinfonce'): {
        'name': 'ppo_builderbench_creative5_task2_e1024_pd_tdinfonce_tau05_catselect_extrew1',
        'note': 'user: lock clear pick',
    },
    ('creative5_task3', 'nf'): {
        'name': (
            'ppo_builderbench_creative5_task3_e1024_pd_nf_compact_sa3x256_r64_b6_w256_'
            'tau05_actorreset_nopermute_norand_minstd1e5_entanneal_evalvid_catselect_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative5_task3', 'crl'): {
        'name': (
            'ppo_builderbench_creative5_task3_e1024_pd_crl_tau05_'
            'actorreset_nopermute_norand_normobs_evalvid_catselect_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative5_task4', 'nf'): {
        'name': (
            'ppo_builderbench_creative5_task4_e1024_pd_nf_compact_sa3x256_r64_b6_w256_'
            'tau05_actorreset_nopermute_norand_minstd1e5_entanneal_evalvid_catselect_extrew1'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative7_task2', 'nf'): {
        'name': (
            'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_'
            'tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m'
        ),
        'note': 'user: lock clear pick',
    },
    ('creative7_task2', 'td3'): {
        'name': (
            'ppo_builderbench_creative7_task2_e1024_pd_td3_logq_tau05_'
            'actorreset_nopermute_normobs_evalvid_catwp_tol0013'
        ),
        'note': 'user: lock clear pick',
    },
}


def _short_name(name: str) -> str:
  for t in TASKS:
    for pref in (f'ppo_builderbench_{t}_', f'ppo_rnd_builderbench_{t}_'):
      if name.startswith(pref):
        tag = 'rnd_' if 'rnd' in pref else ''
        return tag + name[len(pref):]
  return name


def _family(name: str) -> str | None:
  fam = base._method_family(name)
  if fam == 'nf_td':
    return 'nf'  # NF+TD still uses an NF density estimator
  if fam in METHODS:
    return fam
  return None


def _mean_std_se(vals: list[float]) -> tuple[float, float, float]:
  if not vals:
    return float('nan'), float('nan'), float('nan')
  m = sum(vals) / len(vals)
  if len(vals) == 1:
    return m, 0.0, 0.0
  var = sum((y - m) ** 2 for y in vals) / (len(vals) - 1)
  std = math.sqrt(var)
  return m, std, std / math.sqrt(len(vals))


def _peak_of_mean(seed_series):
  xs, mean, se, n = base._aggregate_mean_stderr(seed_series)
  if not mean:
    return None
  i = max(range(len(mean)), key=lambda j: mean[j])
  return {
      'peak_mean': mean[i],
      'peak_se': se[i],
      'peak_x': xs[i],
      'final_mean': mean[-1],
      'final_se': se[-1],
      'n_seeds': n,
      'max_x': xs[-1],
      'n_pts': len(mean),
  }


def _per_seed_peaks(seed_series) -> list[float]:
  out = []
  for pts in seed_series:
    if pts:
      out.append(max(y for _, y in pts))
  return out


def _late_window_stats(
    pts: list[tuple[int, float]],
    *,
    late_frac: float | None = None,
    min_pts: int | None = None,
) -> dict | None:
  """Per-seed late-window mean / std of success.

  Window = points with x >= x_max - late_frac * (x_max - x_min).
  If that set has < min_pts, use the last min_pts logged points instead.
  """
  if late_frac is None:
    late_frac = LATE_FRAC
  if min_pts is None:
    min_pts = LATE_MIN_PTS
  if not pts:
    return None
  pts = sorted(pts, key=lambda p: p[0])
  xs = [p[0] for p in pts]
  x_min, x_max = xs[0], xs[-1]
  span = x_max - x_min
  if span <= 0:
    late = pts
    cutoff = x_min
    mode = 'all_same_x'
  else:
    cutoff = x_max - late_frac * span
    late = [(x, y) for x, y in pts if x >= cutoff]
    mode = f'last_{late_frac:.0%}_steps'
    if len(late) < min_pts:
      late = pts[-min(min_pts, len(pts)):]
      mode = f'last_{len(late)}_pts'
      cutoff = late[0][0]
  ys = [y for _, y in late]
  m, std, _se = _mean_std_se(ys)
  end_pts = pts[-min(END_PTS, len(pts)):]
  end_ys = [y for _, y in end_pts]
  end_m, _, _ = _mean_std_se(end_ys)
  peak = max(y for _, y in pts)
  seed_score = LATE_WEIGHT * m + END_WEIGHT * end_m
  # Collapse = ended much worse than peak, or late window looked OK but
  # the true end plateau dropped. A mid-window dip that recovers (high end)
  # is NOT treated as collapse.
  return {
      'late_mean': m,
      'end_mean': end_m,
      'seed_score': seed_score,  # primary per-seed value
      'late_std': std,  # within-seed variation inside the window
      'n_late_pts': len(ys),
      'n_end_pts': len(end_ys),
      'cutoff_x': cutoff,
      'max_x': x_max,
      'peak': peak,
      'spike_collapse': (
          (peak - end_m) >= SPIKE_COLLAPSE_GAP
          or (m - end_m) >= END_COLLAPSE_GAP
      ),
      'window_mode': mode,
  }


def _moving_average_time(
    xs: list, mean: list, se: list | None = None, *, window_steps: int = 0,
) -> tuple:
  """Centered moving average over a time window in env-step units.

  For each x_i, averages every point with |x - x_i| <= window/2. The SE band
  is smoothed the same way so it tracks the displayed mean. No-op if the
  window is 0 or the series is shorter than 3 points. Scoring is unaffected.
  """
  n = len(xs)
  if n < 3 or not window_steps or window_steps <= 0:
    return xs, mean, se
  half = float(window_steps) / 2.0
  out_mean = [0.0] * n
  out_se = [0.0] * n if se is not None else None
  j0 = 0
  j1 = 0
  for i, x in enumerate(xs):
    lo, hi = float(x) - half, float(x) + half
    while j0 < n and float(xs[j0]) < lo:
      j0 += 1
    if j1 < j0:
      j1 = j0
    while j1 < n and float(xs[j1]) <= hi:
      j1 += 1
    k = j1 - j0
    if k <= 0:
      out_mean[i] = mean[i]
      if out_se is not None:
        out_se[i] = se[i]
      continue
    out_mean[i] = sum(mean[j0:j1]) / k
    if out_se is not None:
      out_se[i] = sum(se[j0:j1]) / k
  return xs, out_mean, out_se


def _seed_max_x(pts: list[tuple[int, float]]) -> int:
  return max(p[0] for p in pts) if pts else 0


def _drop_short_horizon_seeds(
    seed_series: list[list[tuple[int, float]]],
    *,
    name: str | None = None,
    split: str = 'train',
) -> list[list[tuple[int, float]]]:
  """Keep seeds that reached ≥ MIN_SEED_HORIZON_FRAC of the longest seed."""
  nonempty = [pts for pts in seed_series if pts]
  if len(nonempty) <= 1:
    return nonempty
  max_xs = [_seed_max_x(pts) for pts in nonempty]
  horizon = max(max_xs)
  cutoff = MIN_SEED_HORIZON_FRAC * horizon
  kept, dropped = [], []
  for pts, mx in zip(nonempty, max_xs):
    if mx >= cutoff:
      kept.append(pts)
    else:
      dropped.append(mx)
  if dropped and name:
    print(
        f'    drop incomplete {split} seeds for {name}: '
        f'{len(dropped)} seed(s) max_x={dropped} < {cutoff:.0f} '
        f'(horizon={horizon})'
    )
  return kept


def _score_run(name: str) -> dict | None:
  train = _drop_short_horizon_seeds(
      base._read_train_seed_series(base.LOG_ROOT, name),
      name=name, split='train')
  evals = _drop_short_horizon_seeds(
      base._read_eval_seed_series(base.LOG_ROOT, name),
      name=name, split='eval')
  if not train and not evals:
    return None
  tr = _peak_of_mean(train) if train else None
  ev = _peak_of_mean(evals) if evals else None

  # Primary: per-seed blend of late-window + end-plateau; fall back to eval.
  seed_stats = []
  metric = None
  if train:
    for pts in train:
      st = _late_window_stats(pts)
      if st is not None:
        seed_stats.append(st)
    if seed_stats:
      metric = 'train_late_end_blend'
  if not seed_stats and evals:
    for pts in evals:
      st = _late_window_stats(pts)
      if st is not None:
        seed_stats.append(st)
    if seed_stats:
      metric = 'eval_late_end_blend'
  if not seed_stats:
    return None

  seed_scores = [s['seed_score'] for s in seed_stats]
  late_means = [s['late_mean'] for s in seed_stats]
  end_means = [s['end_mean'] for s in seed_stats]
  peaks = [s['peak'] for s in seed_stats]
  late_within_stds = [s['late_std'] for s in seed_stats]
  m, std, se = _mean_std_se(seed_scores)
  late_m, _, late_se = _mean_std_se(late_means)
  end_m, _, end_se = _mean_std_se(end_means)
  peak_m, peak_std, peak_se = _mean_std_se(peaks)
  within_m, _, _ = _mean_std_se(late_within_stds)
  n_spike = sum(1 for s in seed_stats if s['spike_collapse'])
  modes = sorted({s['window_mode'] for s in seed_stats})
  return {
      'name': name,
      'short': _short_name(name),
      'fam': _family(name),
      'label': base._friendly_label(name),
      'metric': metric,
      'late_frac': LATE_FRAC,
      'late_min_pts': LATE_MIN_PTS,
      'end_pts': END_PTS,
      'window_modes': modes,
      'seed_scores': seed_scores,
      'seed_late_means': late_means,
      'seed_end_means': end_means,
      'seed_peaks': peaks,
      'seed_late_stds': late_within_stds,
      'score': m,  # primary: mean of per-seed (0.5 late + 0.5 end)
      'score_std': std,
      'score_se': se,
      'late_score': late_m,
      'late_score_se': late_se,
      'end_score': end_m,
      'end_score_se': end_se,
      'peak_score': peak_m,  # secondary
      'peak_score_std': peak_std,
      'peak_score_se': peak_se,
      'late_within_std_mean': within_m,
      'n_spike_collapse': n_spike,
      'spike_collapse': n_spike > 0,
      'peak_late_gap': peak_m - late_m,
      'late_end_gap': late_m - end_m,
      'n_seeds': len(seed_scores),
      'train': tr,
      'eval': ev,
      'train_curve_peak': tr['peak_mean'] if tr else None,
      'train_curve_final': tr['final_mean'] if tr else None,
      'eval_curve_peak': ev['peak_mean'] if ev else None,
  }


def _discover() -> dict[str, dict[str, list[dict]]]:
  by: dict[str, dict[str, list[dict]]] = {
      t: {m: [] for m in METHODS} for t in TASKS
  }
  names = set()
  if os.path.isdir(base.LOG_ROOT):
    for n in os.listdir(base.LOG_ROOT):
      names.add(n)
  names |= set(base._log_dirs_from_slurm_files(base.SLURM_DIR))

  for name in sorted(names):
    m = base.DIR_RE.match(name)
    if not m:
      continue
    task = f"creative{m.group('creative')}_task{m.group('task')}"
    if task not in by:
      continue
    if any(s in name for s in EXCLUDE_SUBSTR):
      continue
    fam = _family(name)
    if fam is None:
      continue
    has_csv = base._has_csv(base.LOG_ROOT, name, 'learner', min_size=200)
    has_slurm = bool(base._read_train_from_slurm(base.SLURM_DIR, name))
    if not has_csv and not has_slurm:
      continue
    scored = _score_run(name)
    if scored is None:
      continue
    by[task][fam].append(scored)
  return by


def _rank_key(r: dict):
  # Higher late mean, more seeds, lower across-seed std, lower within-window
  # std, higher secondary peak, higher eval curve peak.
  ev = r['eval_curve_peak'] if r['eval_curve_peak'] is not None else -1.0
  within = r.get('late_within_std_mean') or 0.0
  peak = r.get('peak_score') if r.get('peak_score') is not None else -1.0
  return (r['score'], r['n_seeds'], -r['score_std'], -within, peak, ev)


def _fmt_floats(vals: list[float]) -> str:
  return '[' + ','.join(f'{x:.2f}' for x in vals) + ']'


def _apply_user_override(
    task: str,
    fam: str,
    runs: list[dict],
    best: dict | None,
    reason: str | None,
) -> tuple[dict | None, str | None, dict | None]:
  """Force USER_OVERRIDES pick; clear ambiguity. Returns (run, reason, ov)."""
  ov = USER_OVERRIDES.get((task, fam))
  if ov is None:
    return best, reason, None
  wanted = ov['name']
  chosen = next((r for r in runs if r['name'] == wanted), None)
  if chosen is None:
    # Override name missing from discover — fall back but keep asking.
    print(f'  WARNING: USER_OVERRIDES missing run for {task}/{fam}: {wanted}')
    return best, reason, ov
  chosen = dict(chosen)  # shallow copy so we can attach flags
  chosen['user_locked'] = True
  chosen['user_note'] = ov.get('note', '')
  chosen['collapsed_reference'] = bool(ov.get('collapsed_reference', False))
  # Locked → not ambiguous for ask-list / provisional markers.
  return chosen, None, ov


def _pick_and_flag(runs: list[dict]) -> tuple[dict | None, list[dict], str | None]:
  """Return (provisional_best, ambiguity_options, reason_or_None)."""
  if not runs:
    return None, [], None
  ranked = sorted(runs, key=_rank_key, reverse=True)
  best = ranked[0]
  if len(ranked) == 1:
    reasons_solo = []
    if best['n_seeds'] >= 2 and best['score_std'] >= HIGH_SEED_STD:
      reasons_solo.append(
          f'only config; high seed-std={best["score_std"]:.2f} '
          f'(scores={_fmt_floats(best["seed_scores"])}) — '
          f'reported as best-available')
    if best.get('spike_collapse'):
      reasons_solo.append(
          f'spike/collapse: peak={best["peak_score"]:.3f} '
          f'late={best["late_score"]:.3f} end={best["end_score"]:.3f} '
          f'(n_spike_seeds={best["n_spike_collapse"]}/{best["n_seeds"]})')
    if reasons_solo:
      return best, [best], '; '.join(reasons_solo)
    return best, [], None

  # All failed on primary → still ask if some runs had high peaks (collapse).
  if best['score'] < FAILED:
    peaked = [r for r in ranked if (r.get('peak_score') or 0) >= 0.5]
    if peaked:
      opts = []
      seen = set()
      for r in [best] + peaked:
        if r['name'] not in seen:
          opts.append(r)
          seen.add(r['name'])
        if len(opts) >= 4:
          break
      return best, opts, (
          f'all late/end scores < {FAILED:.2f}, but {len(peaked)} run(s) '
          f'had peak≥0.5 (likely mid-training spike then collapse) — '
          f'pick intentionally or treat as failed')
    return best, [], None

  second = ranked[1]
  reasons = []
  both_saturated = (
      best['score'] >= SATURATED and second['score'] >= SATURATED)

  # Meaningful near-tie on primary late/end score.
  if (abs(best['score'] - second['score']) <= NEAR_TIE
      and not both_saturated
      and second['score'] < SATURATED):
    reasons.append(
        f"near-tie on late/end score ({best['score']:.3f} vs "
        f"{second['score']:.3f})")

  # Best is inconsistent across seeds; second is more consistent AND viable.
  if (best['n_seeds'] >= 2 and best['score_std'] >= HIGH_SEED_STD
      and second['score'] >= MIN_ALT_SCORE
      and second['score_std'] < best['score_std'] - 0.05
      and second['score'] >= best['score'] - 0.25):
    reasons.append(
        f"leader seed-std={best['score_std']:.2f} high "
        f"(scores={_fmt_floats(best['seed_scores'])}); "
        f"runner-up more consistent "
        f"(std={second['score_std']:.2f}, score={second['score']:.3f}, "
        f"scores={_fmt_floats(second['seed_scores'])})")

  # High-variance multi-seed leader: always flag (even without a good alt).
  elif best['n_seeds'] >= 2 and best['score_std'] >= HIGH_SEED_STD:
    if second['score'] >= MIN_ALT_SCORE:
      reasons.append(
          f"leader seed-std={best['score_std']:.2f} high "
          f"(scores={_fmt_floats(best['seed_scores'])}); compare vs "
          f"{second['short']} score={second['score']:.3f} "
          f"std={second['score_std']:.2f} "
          f"scores={_fmt_floats(second['seed_scores'])})")
    else:
      reasons.append(
          f"leader seed-std={best['score_std']:.2f} high "
          f"(scores={_fmt_floats(best['seed_scores'])}); "
          f"no strong multi-seed alternative")

  # Single-seed leader vs solid multi-seed runner-up.
  if (best['n_seeds'] == 1 and second['n_seeds'] >= 2
      and second['score'] >= best['score'] - MULTI_SEED_BONUS_GAP
      and not both_saturated):
    reasons.append(
        f"leader is n=1 (score={best['score']:.3f}); runner-up has "
        f"n={second['n_seeds']} with score={second['score']:.3f}")

  # Saturated n=1 leader vs multi-seed within tiny gap.
  if (best['n_seeds'] == 1 and second['n_seeds'] >= 2
      and both_saturated
      and second['score'] >= best['score'] - 0.03):
    reasons.append(
        f"both saturated late/end but leader is n=1; runner-up "
        f"n={second['n_seeds']} ({second['score']:.3f}) — prefer more seeds?")

  # Single-seed saturated leader vs multi-seed nearly-as-good.
  if (best['n_seeds'] == 1 and best['score'] >= SATURATED):
    for r in ranked[1:]:
      if (r['n_seeds'] >= 2 and r['score'] >= best['score'] - MULTI_SEED_BONUS_GAP
          and r['score_std'] < 0.15):
        reasons.append(
            f"leader n=1 saturated late/end; strong multi-seed alt "
            f"n={r['n_seeds']} score={r['score']:.3f}±{r['score_se']:.3f} "
            f"(std={r['score_std']:.2f})")
        break

  # Spike-then-collapse on the provisional leader.
  if best.get('spike_collapse'):
    reasons.append(
        f"leader spike/collapse: peak={best['peak_score']:.3f} "
        f"late={best['late_score']:.3f} end={best['end_score']:.3f} "
        f"(n_spike={best['n_spike_collapse']}/{best['n_seeds']})")

  # Peak vs primary ranking disagreement with a close competitor.
  for r in ranked[1:4]:
    if r['score'] < MIN_ALT_SCORE:
      continue
    late_gap = best['score'] - r['score']
    peak_gap = (best.get('peak_score') or 0.0) - (r.get('peak_score') or 0.0)
    if peak_gap < -PEAK_LATE_DISAGREE and late_gap >= 0:
      reasons.append(
          f"peak vs late/end disagree vs {r['short']}: "
          f"leader score={best['score']:.3f}/peak={best['peak_score']:.3f} "
          f"vs alt score={r['score']:.3f}/peak={r['peak_score']:.3f}")
      break
    if abs(late_gap) <= NEAR_TIE and peak_gap > PEAK_LATE_DISAGREE:
      reasons.append(
          f"near late/end-tie but peak favors {r['short']}: "
          f"leader score={best['score']:.3f}/peak={best['peak_score']:.3f} "
          f"vs alt score={r['score']:.3f}/peak={r['peak_score']:.3f}")
      break

  if not reasons:
    return best, [], None

  # Collect competitors for the ask-list.
  opts = [best]
  for r in ranked[1:]:
    close = abs(r['score'] - best['score']) <= max(NEAR_TIE, MULTI_SEED_BONUS_GAP)
    multi_alt = (
        best['n_seeds'] == 1 and r['n_seeds'] >= 2
        and r['score'] >= best['score'] - MULTI_SEED_BONUS_GAP)
    high_var_alt = (
        best['n_seeds'] >= 2 and best['score_std'] >= HIGH_SEED_STD
        and r['score'] >= MIN_ALT_SCORE)
    spike_alt = best.get('spike_collapse') and r['score'] >= MIN_ALT_SCORE
    peak_alt = (
        (r.get('peak_score') or 0) > (best.get('peak_score') or 0) + PEAK_LATE_DISAGREE
        and r['score'] >= best['score'] - MULTI_SEED_BONUS_GAP)
    if close or multi_alt or high_var_alt or spike_alt or peak_alt:
      opts.append(r)
    if len(opts) >= 4:
      break
  return best, opts, '; '.join(dict.fromkeys(reasons))  # dedupe, keep order


def _fmt_run(r: dict) -> str:
  scores = _fmt_floats(r['seed_scores'])
  lates = _fmt_floats(r['seed_late_means'])
  ends = _fmt_floats(r['seed_end_means'])
  peaks = _fmt_floats(r['seed_peaks'])
  tr = (f"curve_train_peak={r['train_curve_peak']:.3f}"
        if r['train_curve_peak'] is not None else 'curve_train=NA')
  trf = (f"curve_train_final={r['train_curve_final']:.3f}"
         if r.get('train_curve_final') is not None else '')
  ev = (f"curve_eval_peak={r['eval_curve_peak']:.3f}"
        if r['eval_curve_peak'] is not None else 'curve_eval=NA')
  spike = (f" spike_seeds={r['n_spike_collapse']}/{r['n_seeds']}"
           if r.get('n_spike_collapse') else '')
  return (
      f"{r['short']}\n"
      f"    score={r['score']:.3f}±{r['score_se']:.3f} "
      f"(std={r['score_std']:.3f}, n={r['n_seeds']}) "
      f"seed_scores={scores} | late={r['late_score']:.3f} "
      f"end={r['end_score']:.3f} peak={r['peak_score']:.3f} | "
      f"seed_late={lates} seed_end={ends} seed_peaks={peaks} | {tr}"
      + (f" | {trf}" if trf else '')
      + f" | {ev}{spike}"
  )


def _legend_tag(run: dict, fam: str, ambiguous: set[str]) -> str:
  if run.get('collapsed_reference'):
    return ' [collapsed ref]'
  if fam in ambiguous:
    return ' [provisional]'
  return ''


def _plot_task(task: str, picks: dict[str, dict | None],
               ambiguous: set[str]) -> str | None:
  present = [(m, picks[m]) for m in METHODS if picks.get(m) is not None]
  if not present:
    print(f'  {task}: no methods to plot')
    return None

  fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
  task_label = _task_display(task)

  for ax, split, title_metric, ylabel in (
      (axes[0], 'train', 'train_success_1000', 'train success (1000)'),
      (axes[1], 'eval', 'eval success', 'eval success'),
  ):
    for i, (fam, run) in enumerate(present):
      name = run['name']
      if split == 'train':
        series = _drop_short_horizon_seeds(
            base._read_train_seed_series(base.LOG_ROOT, name),
            name=name, split='train')
      else:
        series = _drop_short_horizon_seeds(
            base._read_eval_seed_series(base.LOG_ROOT, name),
            name=name, split='eval')
      xs, mean, se, n_seeds = base._aggregate_mean_stderr(series)
      if not xs:
        continue
      win = SMOOTH_WINDOW_TRAIN if split == 'train' else SMOOTH_WINDOW_EVAL
      xs, mean, se = _moving_average_time(xs, mean, se, window_steps=win)
      xs, mean, se = base._subsample_curve(xs, mean, se)
      color = METHOD_COLORS[fam]
      tag = _legend_tag(run, fam, ambiguous)
      peak_bit = f" peak={run['peak_score']:.2f}"
      if run.get('collapsed_reference'):
        peak_bit = f" peak={run['peak_score']:.2f} (collapsed)"
      label = (
          f"{METHOD_TITLE[fam]}{tag} (n={n_seeds})\n"
          f"  score={run['score']:.2f}±{run['score_se']:.2f} "
          f"(late={run['late_score']:.2f} end={run['end_score']:.2f}"
          f"{peak_bit})"
      )
      # Collapsed-reference curves: dashed so they read as non-primary.
      lw = 2.0 if run.get('collapsed_reference') else 2.4
      ls = '--' if run.get('collapsed_reference') and split == 'train' else '-'
      alpha = 0.85 if run.get('collapsed_reference') else 0.95
      kw = dict(color=color, linewidth=lw, linestyle=ls, label=label,
                alpha=alpha, zorder=3 + i)
      ax.plot(xs, mean, **kw)
      if n_seeds > 1 and any(s > 0 for s in se):
        lo = [m - s for m, s in zip(mean, se)]
        hi = [m + s for m, s in zip(mean, se)]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0,
                        zorder=2 + i)
      # Shade late window on train panel using mean curve x-span.
      if split == 'train' and xs:
        x0, x1 = xs[0], xs[-1]
        span = x1 - x0
        if span > 0:
          cut = x1 - LATE_FRAC * span
          ax.axvspan(cut, x1, color=color, alpha=0.06, zorder=1)
    win = SMOOTH_WINDOW_TRAIN if split == 'train' else SMOOTH_WINDOW_EVAL
    win_m = win / 1e6
    ax.set_title(
        f'{title_metric}  (centered MA {win_m:.0f}M steps)',
        fontsize=11, fontweight='bold')
    ax.set_xlabel('env steps', fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
    ax.set_ylim(-0.05, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if split == 'train':
      ax.set_ylabel(ylabel, fontsize=11)

  axes[1].legend(
      loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9,
      framealpha=1.0, edgecolor='#333333', fancybox=False,
      handlelength=2.2, labelspacing=0.55,
  )
  fig.suptitle(
      f'BuilderBench {task_label} — best run per density estimator '
      f'(late/end score ± stderr; curves time-smoothed)',
      fontsize=12, fontweight='bold', y=1.02,
  )
  fig.tight_layout()
  out = os.path.join(
      OUT_DIR, f'{task}_best_nf_crl_td3_tdinfonce_train_eval.png')
  fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  print(f'  wrote {out}')
  return out


def _fmt_seed_list(vals) -> str:
  if not vals:
    return '—'
  return ', '.join(f'{x:.3f}' for x in vals)


def _write_best_runs_md(path: str, picks: dict, amb_map: dict) -> None:
  """Human-readable durable reference of current best picks; flags n_seeds==1."""
  lines: list[str] = []
  lines.append('# BuilderBench best-method picks')
  lines.append('')
  lines.append(
      'Durable reference for the current best NF / CRL / TD3 / TD-InfoNCE '
      'run per creative task. Regenerated by '
      '`scripts/plot_builderbench_best_method_compare.py`.')
  lines.append('')
  lines.append('## Scoring metric')
  lines.append('')
  lines.append(
      f'- **Primary:** mean over seeds of '
      f'`{LATE_WEIGHT}*late_mean + {END_WEIGHT}*end_mean` of '
      '`train_success_1000`')
  lines.append(
      f'- **Late window:** last {LATE_FRAC:.0%} of each seed\'s step span '
      f'(fallback last {LATE_MIN_PTS} points)')
  lines.append(f'- **End plateau:** mean of last {END_PTS} logged points')
  lines.append('- **Secondary / reported:** per-seed peak `train_success_1000`')
  lines.append('')
  lines.append(
      'Machine-readable mirrors: [`summary.csv`](summary.csv) '
      '(includes `n_seeds`, `single_seed`), [`picks.json`](picks.json).')
  lines.append('')

  single_seed: list[tuple[str, str, str]] = []
  multi_seed: list[tuple[str, str, int]] = []
  missing: list[tuple[str, str]] = []
  for task in TASKS:
    fams = picks.get(task) or {}
    for fam in METHODS:
      r = fams.get(fam)
      if r is None:
        missing.append((task, fam))
        continue
      n = int(r.get('n_seeds') or 0)
      short = r.get('short') or _short_name(r['name'])
      if n == 1:
        single_seed.append((task, fam, short))
      else:
        multi_seed.append((task, fam, n))

  lines.append('## Single-seed best runs (`n_seeds == 1`)')
  lines.append('')
  lines.append(
      f'**{len(single_seed)}** of the current best picks rest on a single '
      'seed — treat score ± SE as unreliable until more seeds land:')
  lines.append('')
  for task, fam, short in single_seed:
    lines.append(
        f'- **`{task}` / {METHOD_TITLE[fam]}** — `{short}`')
  lines.append('')

  lines.append('## Multi-seed best runs')
  lines.append('')
  if not multi_seed:
    lines.append('_None._')
  else:
    for task, fam, n in multi_seed:
      lines.append(f'- `{task}` / {METHOD_TITLE[fam]} — **n_seeds={n}**')
  lines.append('')

  if missing:
    lines.append('## Missing (no run found)')
    lines.append('')
    for task, fam in missing:
      lines.append(f'- `{task}` / {METHOD_TITLE[fam]}')
    lines.append('')

  lines.append('## All picks by task')
  lines.append('')
  for task in TASKS:
    display = _task_display(task)
    lines.append(f'### `{task}` — {display}')
    lines.append('')
    fams = picks.get(task) or {}
    for fam in METHODS:
      r = fams.get(fam)
      title = METHOD_TITLE[fam]
      if r is None:
        lines.append(f'#### {title} — *missing*')
        lines.append('')
        lines.append('- **Status:** no runs found')
        lines.append('')
        continue

      n = int(r.get('n_seeds') or 0)
      short = r.get('short') or _short_name(r['name'])
      locked = bool(r.get('user_locked'))
      collapsed = bool(r.get('collapsed_reference'))
      amb = bool(task in amb_map and fam in amb_map[task])
      if collapsed:
        status = 'collapsed_reference'
      elif locked:
        status = 'locked'
      elif amb:
        status = 'ambiguous / provisional'
      else:
        status = 'clear'

      seed_flag = ' **← SINGLE-SEED**' if n == 1 else ''
      lines.append(f'#### {title}{seed_flag}')
      lines.append('')
      lines.append(f'- **Display:** {title}')
      lines.append(f'- **Full run:** `{r["name"]}`')
      lines.append(f'- **Short suffix:** `{short}`')
      lines.append(
          f'- **Score:** {r["score"]:.4f} ± {r["score_se"]:.4f} '
          f'(std={r["score_std"]:.4f})')
      lines.append(
          f'- **Late / end / peak:** {r["late_score"]:.4f} / '
          f'{r["end_score"]:.4f} / {r["peak_score"]:.4f}')
      lines.append(f'- **n_seeds:** **{n}**{seed_flag}')
      lines.append(
          f'- **Seed scores:** {_fmt_seed_list(r.get("seed_scores"))}')
      lines.append(
          f'- **Seed late means:** {_fmt_seed_list(r.get("seed_late_means"))}')
      lines.append(
          f'- **Seed end means:** {_fmt_seed_list(r.get("seed_end_means"))}')
      lines.append(
          f'- **Seed peaks:** {_fmt_seed_list(r.get("seed_peaks"))}')
      lines.append(f'- **Status:** {status}')
      note = r.get('user_note') or (amb_map.get(task, {}).get(fam, {}) or {}).get(
          'reason') or r.get('reason')
      if note:
        lines.append(f'- **Notes:** {note}')
      if collapsed:
        lines.append(
            '- **Flag:** collapsed reference (high mid-train peak, '
            'late/end ≈ 0) — kept for plot reference only')
      lines.append('')

  lines.append('---')
  lines.append('')
  lines.append(
      f'_Auto-generated. Metric late_frac={LATE_FRAC}, end_pts={END_PTS}, '
      f'weights={LATE_WEIGHT}/{END_WEIGHT}._')
  lines.append('')

  with open(path, 'w') as f:
    f.write('\n'.join(lines))


def main():
  global LATE_FRAC, LATE_MIN_PTS, SMOOTH_WINDOW_TRAIN, SMOOTH_WINDOW_EVAL
  ap = argparse.ArgumentParser()
  ap.add_argument('--tasks', nargs='*', default=TASKS)
  ap.add_argument('--late-frac', type=float, default=LATE_FRAC,
                  help='Fraction of each seed step-span used as late window')
  ap.add_argument('--late-min-pts', type=int, default=LATE_MIN_PTS,
                  help='Min logged points in late window (else take last N)')
  ap.add_argument(
      '--smooth-train-steps', type=float, default=SMOOTH_WINDOW_TRAIN,
      help='Centered moving-average window in env steps for train curves '
           '(0 disables). Default 10M.')
  ap.add_argument(
      '--smooth-eval-steps', type=float, default=SMOOTH_WINDOW_EVAL,
      help='Centered moving-average window in env steps for eval curves '
           '(0 disables). Default 20M.')
  ap.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds (CPU-only; login node).')
  args = ap.parse_args()
  tasks = args.tasks
  LATE_FRAC = float(args.late_frac)
  LATE_MIN_PTS = int(args.late_min_pts)
  SMOOTH_WINDOW_TRAIN = int(args.smooth_train_steps)
  SMOOTH_WINDOW_EVAL = int(args.smooth_eval_steps)

  if args.watch <= 0:
    run_once(tasks)
    return

  print(f'[best_method_compare] watching every {args.watch}s '
        f'→ {OUT_DIR}', flush=True)
  while True:
    t0 = time.time()
    try:
      run_once(tasks)
      print(f'[best_method_compare] refreshed in {time.time() - t0:.1f}s',
            flush=True)
    except Exception as exc:
      print(f'[best_method_compare] ERROR: {exc}', flush=True)
    time.sleep(args.watch)


def run_once(tasks):
  os.makedirs(OUT_DIR, exist_ok=True)
  print(f'Scoring runs with late_frac={LATE_FRAC}, '
        f'late_min_pts={LATE_MIN_PTS} (this can take a few minutes)...')
  print(f'Display smoothing: train MA={SMOOTH_WINDOW_TRAIN/1e6:.0f}M steps, '
        f'eval MA={SMOOTH_WINDOW_EVAL/1e6:.0f}M steps')
  by = _discover()

  picks: dict[str, dict[str, dict | None]] = {}
  amb_map: dict[str, dict[str, dict]] = defaultdict(dict)
  summary_rows = []
  amb_lines = ['# Ambiguous best-run picks', '']
  amb_lines.append(
      f'Ranking metric: **mean over seeds of** '
      f'`{LATE_WEIGHT:.1f}·late_mean + {END_WEIGHT:.1f}·end_mean` of '
      f'`train_success_1000`, where late_mean = last {LATE_FRAC:.0%} of each '
      f'seed\'s step span (fallback last {LATE_MIN_PTS} pts) and end_mean = '
      f'last {END_PTS} logged points. Peak is secondary. Flagged when '
      f'peak−end ≥ {SPIKE_COLLAPSE_GAP:.2f} or late−end ≥ {END_COLLAPSE_GAP:.2f}.')
  amb_lines.append('')
  amb_lines.append(
      'Human decisions are locked in `USER_OVERRIDES` inside '
      '`scripts/plot_builderbench_best_method_compare.py` and are **not** '
      're-flagged as needing a choice.')
  amb_lines.append('')
  resolved_lines: list[str] = ['# Resolved / locked picks', '']

  for task in tasks:
    print('=' * 72)
    print(task)
    picks[task] = {}
    amb_fams: set[str] = set()
    for fam in METHODS:
      runs = by[task][fam]
      best, opts, reason = _pick_and_flag(runs)
      auto_reason = reason
      best, reason, ov = _apply_user_override(task, fam, runs, best, reason)
      picks[task][fam] = best
      if best is None:
        print(f'  {fam}: (no runs)')
        summary_rows.append({
            'task': task, 'method': fam, 'status': 'missing',
            'run': '', 'score': '', 'score_se': '', 'score_std': '',
            'n_seeds': '', 'single_seed': '', 'seed_scores': '',
            'seed_late_means': '', 'seed_end_means': '', 'seed_peaks': '',
            'ambiguous': '', 'reason': 'no runs found',
        })
        continue

      locked = bool(best.get('user_locked'))
      collapsed = bool(best.get('collapsed_reference'))
      if collapsed:
        status_key = 'collapsed_reference'
        status = 'LOCKED-COLLAPSED-REF'
      elif locked:
        status_key = 'locked'
        status = 'LOCKED'
      elif reason:
        status_key = 'ambiguous'
        status = 'AMBIGUOUS'
      else:
        status_key = 'clear'
        status = 'CLEAR'
      print(f'  {fam}: {status} → score={best["score"]:.3f}±{best["score_se"]:.3f} '
            f'(late={best["late_score"]:.3f} end={best["end_score"]:.3f} '
            f'peak={best["peak_score"]:.3f}) n={best["n_seeds"]}  '
            f'{_short_name(best["name"])}')
      if locked and best.get('user_note'):
        print(f'    locked: {best["user_note"]}')
        resolved_lines.append(f'## {task} / {METHOD_TITLE[fam]}')
        resolved_lines.append(f'- **Decision:** {best["user_note"]}')
        resolved_lines.append(f'- **Locked pick:** `{best["name"]}`')
        if collapsed:
          resolved_lines.append(
              f'- **Annotation:** collapsed-for-reference '
              f'(late/end≈{best["score"]:.3f}, peak={best["peak_score"]:.3f})')
        if auto_reason:
          resolved_lines.append(
              f'- **Would have been ambiguous:** {auto_reason}')
        resolved_lines.append('')

      if reason:
        amb_fams.add(fam)
        amb_map[task][fam] = {
            'reason': reason,
            'provisional': best['name'],
            'options': [o['name'] for o in opts],
        }
        print(f'    reason: {reason}')
        amb_lines.append(f'## {task} / {METHOD_TITLE[fam]}')
        amb_lines.append(f'- **Why unclear:** {reason}')
        amb_lines.append(
            f'- **Provisional pick (used in plot):** `{best["name"]}`')
        amb_lines.append('- **Options:**')
        for o in opts:
          mark = ' ← provisional' if o['name'] == best['name'] else ''
          amb_lines.append(f'  1. `{o["name"]}`{mark}')
          amb_lines.append(f'     {_fmt_run(o).split(chr(10), 1)[1]}')
        amb_lines.append('')

      note = best.get('user_note') or reason or ''
      n_seeds = int(best['n_seeds'])
      summary_rows.append({
          'task': task,
          'method': fam,
          'status': status_key,
          'run': best['name'],
          'short': best['short'],
          'score': f'{best["score"]:.4f}',
          'score_se': f'{best["score_se"]:.4f}',
          'score_std': f'{best["score_std"]:.4f}',
          'late_score': f'{best["late_score"]:.4f}',
          'end_score': f'{best["end_score"]:.4f}',
          'peak_score': f'{best["peak_score"]:.4f}',
          'peak_score_se': f'{best["peak_score_se"]:.4f}',
          'peak_late_gap': f'{best["peak_late_gap"]:.4f}',
          'late_end_gap': f'{best["late_end_gap"]:.4f}',
          'late_within_std_mean': f'{best["late_within_std_mean"]:.4f}',
          'n_spike_collapse': best['n_spike_collapse'],
          'n_seeds': n_seeds,
          'single_seed': 'yes' if n_seeds == 1 else 'no',
          'seed_scores': ';'.join(f'{x:.3f}' for x in best['seed_scores']),
          'seed_late_means': ';'.join(
              f'{x:.3f}' for x in best['seed_late_means']),
          'seed_end_means': ';'.join(
              f'{x:.3f}' for x in best['seed_end_means']),
          'seed_peaks': ';'.join(f'{x:.3f}' for x in best['seed_peaks']),
          'train_curve_peak': (f'{best["train_curve_peak"]:.4f}'
                               if best['train_curve_peak'] is not None else ''),
          'train_curve_final': (f'{best["train_curve_final"]:.4f}'
                                if best.get('train_curve_final') is not None
                                else ''),
          'eval_curve_peak': (f'{best["eval_curve_peak"]:.4f}'
                              if best['eval_curve_peak'] is not None else ''),
          'late_frac': LATE_FRAC,
          'end_pts': END_PTS,
          'ambiguous': 'yes' if reason else 'no',
          'collapsed_reference': 'yes' if collapsed else 'no',
          'user_locked': 'yes' if locked else 'no',
          'reason': note,
      })

    _plot_task(task, picks[task], amb_fams)

  # Multipanel overview (train only).
  fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
  axes_flat = axes.ravel()
  for ax, task in zip(axes_flat, tasks):
    for i, fam in enumerate(METHODS):
      run = picks[task].get(fam)
      if run is None:
        continue
      series = _drop_short_horizon_seeds(
          base._read_train_seed_series(base.LOG_ROOT, run['name']),
          name=run['name'], split='train')
      xs, mean, se, n_seeds = base._aggregate_mean_stderr(series)
      if not xs:
        continue
      xs, mean, se = _moving_average_time(
          xs, mean, se, window_steps=SMOOTH_WINDOW_TRAIN)
      xs, mean, se = base._subsample_curve(xs, mean, se, max_pts=300)
      color = METHOD_COLORS[fam]
      if run.get('collapsed_reference'):
        tag = '†'
        ls, lw, alpha = '--', 1.6, 0.85
      elif task in amb_map and fam in amb_map[task]:
        tag = '*'
        ls, lw, alpha = '-', 2.0, 0.95
      else:
        tag = ''
        ls, lw, alpha = '-', 2.0, 0.95
      ax.plot(xs, mean, color=color, linewidth=lw, linestyle=ls,
              label=f'{METHOD_TITLE[fam]}{tag} n={n_seeds}', alpha=alpha)
      if n_seeds > 1 and any(s > 0 for s in se):
        lo = [m - s for m, s in zip(mean, se)]
        hi = [m + s for m, s in zip(mean, se)]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.set_title(_task_display(task), fontsize=10, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    if ax in axes[:, 0]:
      ax.set_ylabel('train success')
    if ax in axes[-1, :]:
      ax.set_xlabel('env steps')
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
  fig.suptitle(
      'Best density-estimator run per task '
      f'(0.5·late[{LATE_FRAC:.0%}] + 0.5·end[{END_PTS}pts] of '
      'train_success_1000; * = provisional, † = collapsed ref)',
      fontsize=13, fontweight='bold')
  fig.tight_layout()
  overview = os.path.join(OUT_DIR, 'overview_best_methods_train.png')
  fig.savefig(overview, dpi=150, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  print(f'wrote {overview}')

  csv_path = os.path.join(OUT_DIR, 'summary.csv')
  fields = [
      'task', 'method', 'status', 'run', 'short', 'score', 'score_se',
      'score_std', 'late_score', 'end_score', 'peak_score', 'peak_score_se',
      'peak_late_gap', 'late_end_gap', 'late_within_std_mean',
      'n_spike_collapse', 'n_seeds', 'single_seed', 'seed_scores',
      'seed_late_means', 'seed_end_means', 'seed_peaks', 'train_curve_peak',
      'train_curve_final', 'eval_curve_peak', 'late_frac', 'end_pts',
      'ambiguous', 'collapsed_reference', 'user_locked', 'reason',
  ]
  with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(summary_rows)
  print(f'wrote {csv_path}')

  amb_path = os.path.join(OUT_DIR, 'ambiguities.md')
  if len(amb_map) == 0:
    amb_lines.append('_No open ambiguous cases — all prior asks are locked '
                     'in USER_OVERRIDES._')
  amb_lines.append('')
  amb_lines.extend(resolved_lines)
  with open(amb_path, 'w') as f:
    f.write('\n'.join(amb_lines) + '\n')
  print(f'wrote {amb_path}')

  overrides_ser = {
      f'{t}/{m}': {
          'name': v['name'],
          'note': v.get('note', ''),
          'collapsed_reference': bool(v.get('collapsed_reference', False)),
      }
      for (t, m), v in USER_OVERRIDES.items()
  }
  picks_ser = {
      task: {
          fam: (None if r is None else {
              'name': r['name'],
              'score': r['score'],
              'score_se': r['score_se'],
              'score_std': r['score_std'],
              'late_score': r['late_score'],
              'end_score': r['end_score'],
              'peak_score': r['peak_score'],
              'peak_score_se': r['peak_score_se'],
              'peak_late_gap': r['peak_late_gap'],
              'late_end_gap': r['late_end_gap'],
              'n_spike_collapse': r['n_spike_collapse'],
              'n_seeds': r['n_seeds'],
              'single_seed': int(r['n_seeds']) == 1,
              'seed_scores': r['seed_scores'],
              'seed_late_means': r['seed_late_means'],
              'seed_end_means': r['seed_end_means'],
              'seed_peaks': r['seed_peaks'],
              'short': r.get('short') or _short_name(r['name']),
              'late_frac': LATE_FRAC,
              'end_pts': END_PTS,
              'ambiguous': bool(task in amb_map and fam in amb_map[task]),
              'user_locked': bool(r.get('user_locked')),
              'collapsed_reference': bool(r.get('collapsed_reference')),
              'user_note': r.get('user_note'),
              'reason': (amb_map.get(task, {}).get(fam, {}) or {}).get('reason'),
          })
          for fam, r in fams.items()
      }
      for task, fams in picks.items()
  }
  picks_path = os.path.join(OUT_DIR, 'picks.json')
  with open(picks_path, 'w') as f:
    json.dump({
        'metric': {
            'primary': (
                f'mean_over_seeds({LATE_WEIGHT}*late_mean + '
                f'{END_WEIGHT}*end_mean of train_success_1000); '
                f'late_mean = last {LATE_FRAC:.0%} of per-seed step span '
                f'(fallback last {LATE_MIN_PTS} pts); '
                f'end_mean = last {END_PTS} logged points'
            ),
            'secondary': 'mean_over_seeds(per_seed_peak(train_success_1000))',
            'late_frac': LATE_FRAC,
            'late_min_pts': LATE_MIN_PTS,
            'end_pts': END_PTS,
            'late_weight': LATE_WEIGHT,
            'end_weight': END_WEIGHT,
            'spike_collapse_gap': SPIKE_COLLAPSE_GAP,
            'end_collapse_gap': END_COLLAPSE_GAP,
        },
        'user_overrides': overrides_ser,
        'picks': picks_ser,
        'ambiguities': amb_map,
    }, f, indent=2)
  print(f'wrote {picks_path}')

  # Also serialize short names onto picks for the markdown writer.
  for task, fams in picks.items():
    for fam, r in fams.items():
      if r is not None and 'short' not in r:
        r['short'] = _short_name(r['name'])
  best_md = os.path.join(OUT_DIR, 'BEST_RUNS.md')
  _write_best_runs_md(best_md, picks, amb_map)
  print(f'wrote {best_md}')


if __name__ == '__main__':
  main()
