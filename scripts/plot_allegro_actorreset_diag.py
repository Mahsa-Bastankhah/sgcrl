#!/usr/bin/env python3
"""Diagnostics for the three Allegro actor-reset 300M runs.

Train curves are raw (not smoothed). Eval is off.

  python scripts/plot_allegro_actorreset_diag.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_actorreset_diag.png')
OUT_PPO = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_actorreset_ppo.png')
OUT_ACT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_actorreset_actions.png')
OUT_GRAD = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_actorreset_gradreg.png')
ACTION_CACHE = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', '_akt_ar_action_stats.npz')
RESET_EP_FRAC = 0.8
GRAD_REG_C = 100.0  # default ppo_crl_grad_reg_c; λ=0 so not in the NF loss
NOMINAL_EP = 300.0
RESET_EP_LEN = RESET_EP_FRAC * NOMINAL_EP  # 240: last-layer actor reset
C = base.ACCENT_COLORS
# NVIDIA kuka_allegro_touch_sensor.urdf revolute order (23-D PD targets in [-1,1]).
ACTION_DIM_LABELS = (
    'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7',
    'idx0', 'idx1', 'idx2', 'idx3',
    'mid0', 'mid1', 'mid2', 'mid3',
    'rng0', 'rng1', 'rng2', 'rng3',
    'th0', 'th1', 'th2', 'th3',
)
ARM_DIMS = 7

RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_palmgoal_stateonly_actorreset_4h',
        'palm + AR (rand)',
        C[0],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_palmgoal_stateonly_norand_mixtaskg_actorreset_4h',
        'palm mix-g + AR (no rand)',
        C[1],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_norand_mixtaskg_mix75_actorreset_4h',
        'throw mix75 + AR (no rand)',
        C[2],
        '-',
    ),
)

PANELS = (
    ('train_success_1000', 'train success (last 1000)', False),
    ('ep_length_mean', 'episode length  (nominal 300; × = actor-reset)', False),
    ('reward_repr_raw_mean', 'raw NF reward  (pre-norm, tau=0.5 log p)', True),
    ('reward_repr_mean', 'NF reward after return-norm  (PPO sees this)', True),
    ('reward_return_norm_std', r'return-norm $\sigma$  (PPO divides $r$ by this)', False),
    ('nf/density_loss', 'NF density loss  (NLL)', True),
    ('nf/log_p_mean', r'NF $\log p$ mean', True),
    ('nf/log_p_min', r'NF $\log p$ min (solid) / max (dashed)', True),
    ('ppo/entropy', r'PPO policy entropy  ($\times$ = last-layer actor-reset)', False),
    ('ppo/pg_loss', r'clipped surrogate pg_loss  ($\times$ = actor-reset)', True),
    ('ppo/policy_scale_mean',
     r'pre-tanh $\sigma$ mean solid / min dashed  ($\times$ = reset)', False),
    ('ppo/approx_kl', r'approx KL $E[r-1-\log r]$  ($\times$ = actor-reset)', False),
    ('advantage_mean', 'GAE mean  (pre z-score; dotted = std; × = reset)', True),
    ('nf/goal_mean_0', r'NF goal $\mu_0$  (palm $x$ / throw object $x$)', False),
    ('repr_rT_minus_r0_mean_fail', r'$r_T-r_0$  (fail solid / NF-diag succ dashed)', True),
)
RESET_MARK_COLS = frozenset((
    'ep_length_mean', 'ppo/entropy', 'ppo/pg_loss', 'ppo/policy_scale_mean',
    'ppo/approx_kl', 'advantage_mean',
))


def _curve_full(log_dir: str, col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, log_dir, split='learner',
      x_col='global_step', y_col=col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return None
  return xs, mean, n


def _curve(log_dir: str, col: str):
  got = _curve_full(log_dir, col)
  if got is None:
    return None
  xs, mean, n = got
  xs, mean, _ = base._subsample_curve(xs, mean, [0.0] * len(xs))
  return xs, mean, n


def _reset_points(log_dir: str, y_col: str):
  """(xs, ys) on ticks where ep_length_mean < 0.8 T (the actor-reset trigger)."""
  ep = _curve_full(log_dir, 'ep_length_mean')
  yv = _curve_full(log_dir, y_col)
  if ep is None or yv is None:
    return [], []
  xs_e, ye, _ = ep
  xs_y, yy, _ = yv
  ymap = dict(zip(xs_y, yy))
  rx, ry = [], []
  for x, e in zip(xs_e, ye):
    if e < RESET_EP_LEN and x in ymap:
      rx.append(x)
      ry.append(ymap[x])
  return rx, ry


def _mark_resets(ax, log_dir: str, color: str, y_col: str, *, legend: bool):
  rx, ry = _reset_points(log_dir, y_col)
  if not rx:
    return 0
  ax.scatter(rx, ry, color=color, marker='x', s=28, linewidths=1.1,
             zorder=5, label='actor-reset (ep<240)' if legend else None)
  return len(rx)


def _latest_pkl(log_dir: str) -> str:
  base_dir = os.path.join(base.LOG_ROOT, log_dir)
  try:
    runs = sorted(os.listdir(base_dir))
  except OSError:
    return ''
  for run_name in runs:
    path = os.path.join(base_dir, run_name, 'checkpoints', 'latest.pkl')
    if os.path.isfile(path):
      return path
  return ''


def _load_ckpt_numpy(path: str):
  """Unpickle a PPO ckpt without restoring JAX arrays (version-safe)."""
  import pickle

  def _reconstruct_np(fun, args, arr_state, aval_state):
    return np.asarray(fun(*args))

  class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
      if name == '_reconstruct_array':
        return _reconstruct_np
      return super().find_class(module, name)

  with open(path, 'rb') as fh:
    return _Unpickler(fh).load()


def _replay_action_stats(log_dir: str):
  """Mean / std / mean|a| of executed actions in latest.pkl replay.

  Milestone ckpts do not store replay; only latest.pkl does. This is the
  occupancy-weighted action at the end of training (stochastic samples).
  """

  os.makedirs(os.path.dirname(ACTION_CACHE), exist_ok=True)
  cache_key = log_dir.replace('/', '_')
  if os.path.isfile(ACTION_CACHE):
    cached = np.load(ACTION_CACHE, allow_pickle=True)
    if cache_key + '_mean' in cached.files:
      return {
          'mean': cached[cache_key + '_mean'],
          'std': cached[cache_key + '_std'],
          'mean_abs': cached[cache_key + '_mean_abs'],
          'n': int(cached[cache_key + '_n'][0]),
      }
  path = _latest_pkl(log_dir)
  if not os.path.isfile(path):
    print(f'no latest.pkl for {log_dir}')
    return None
  print(f'loading replay actions from {path} ({os.path.getsize(path)/1e9:.2f} GB)',
        flush=True)
  ckpt = _load_ckpt_numpy(path)
  extra = ckpt.get('extra_state') or {}
  replay = extra.get('replay') or {}
  act = replay.get('action')
  del ckpt
  if act is None:
    print(f'{log_dir}: latest.pkl has no extra_state.replay.action')
    return None
  act = np.asarray(act, dtype=np.float32)
  if act.ndim != 2:
    print(f'{log_dir}: unexpected action shape {act.shape}')
    return None
  stats = {
      'mean': act.mean(axis=0),
      'std': act.std(axis=0),
      'mean_abs': np.abs(act).mean(axis=0),
      'n': int(act.shape[0]),
  }
  payload = {}
  if os.path.isfile(ACTION_CACHE):
    old = np.load(ACTION_CACHE, allow_pickle=True)
    payload.update({k: old[k] for k in old.files})
  payload[cache_key + '_mean'] = stats['mean']
  payload[cache_key + '_std'] = stats['std']
  payload[cache_key + '_mean_abs'] = stats['mean_abs']
  payload[cache_key + '_n'] = np.asarray([stats['n']])
  tmp = ACTION_CACHE + '.tmp.npz'
  np.savez(tmp, **payload)
  os.replace(tmp, ACTION_CACHE)
  print(f'{log_dir}: action mean from n={stats["n"]}  shape={act.shape}',
        flush=True)
  return stats


def _curve_diff(log_dir: str, col_a: str, col_b: str):
  """col_a − col_b, aligned by env step (inner join)."""
  a = _curve(log_dir, col_a)
  b = _curve(log_dir, col_b)
  if a is None or b is None:
    return None
  xs_a, ya, n = a
  xs_b, yb, _ = b
  mb = dict(zip(xs_b, yb))
  xs, ys = [], []
  for x, y in zip(xs_a, ya):
    if x in mb:
      xs.append(x)
      ys.append(y - mb[x])
  if not xs:
    return None
  return xs, ys, n


def _style(ax, xlabel: bool = False) -> None:
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)


def _save(fig, path: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, path)
  plt.close(fig)
  print(f'→ {path}')


def main() -> None:
  fig, axes = plt.subplots(5, 3, figsize=(14.2, 16.4), sharex=True)
  axes = axes.ravel()
  reset_legend = True
  for ax, (col, title, hline0) in zip(axes, PANELS):
    any_pts = False
    for log_dir, label, color, ls in RUNS:
      got = _curve(log_dir, col)
      if got is None:
        print(f'{label}: no {col}')
        continue
      xs, mean, n = got
      ax.plot(xs, mean, color=color, linestyle=ls, lw=1.7, alpha=0.95,
              label=label if ax is axes[0] else None)
      any_pts = True
      print(f'{label} {col}: n={n} last={mean[-1]:.4g} steps={xs[-1]:.0f}')
      if col == 'nf/log_p_min':
        got_max = _curve(log_dir, 'nf/log_p_max')
        if got_max is not None:
          xs_m, mean_m, _ = got_max
          ax.plot(xs_m, mean_m, color=color, linestyle='--', lw=1.15,
                  alpha=0.8)
      if col == 'repr_rT_minus_r0_mean_fail':
        got_s = _curve(log_dir, 'repr_rT_minus_r0_mean_succ')
        if got_s is not None:
          xs_s, mean_s, n_s = got_s
          ax.plot(xs_s, mean_s, color=color, linestyle='--', lw=1.2,
                  alpha=0.75)
          print(f'{label} rT-r0 succ: n={n_s} last={mean_s[-1]:.4g}')
      if col == 'ppo/policy_scale_mean':
        got_min = _curve(log_dir, 'ppo/policy_scale_min')
        if got_min is not None:
          xs_m, ym, _ = got_min
          ax.plot(xs_m, ym, color=color, linestyle='--', lw=1.15, alpha=0.85)
      if col == 'advantage_mean':
        got_s = _curve(log_dir, 'advantage_std')
        if got_s is not None:
          xs_s, ys, _ = got_s
          ax.plot(xs_s, ys, color=color, linestyle=':', lw=1.1, alpha=0.7)
      if col in RESET_MARK_COLS:
        n_r = _mark_resets(ax, log_dir, color, col, legend=reset_legend)
        if n_r:
          print(f'{label} {col}: {n_r} actor-reset ticks')
          reset_legend = False
    if not any_pts:
      ax.text(0.5, 0.5, f'no {col}', ha='center', va='center',
              transform=ax.transAxes, color='0.5')
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    if hline0:
      ax.axhline(0.0, color='0.55', lw=0.7)
    if col == 'ep_length_mean':
      ax.axhline(300.0, color='0.55', lw=0.7, linestyle='--')
      ax.axhline(240.0, color='0.55', lw=0.7, linestyle=':')
    if col == 'ppo/policy_scale_mean':
      ax.axhline(1e-5, color='0.55', lw=0.7, linestyle=':')
    if col == 'nf/log_p_min':
      ax.set_ylim(-80.0, 12.0)
    if col == 'reward_repr_mean':
      ax.set_ylim(-0.08, 0.01)
    if col == 'advantage_mean':
      ax.set_ylim(-1.0, 1.0)
    if col == 'ppo/approx_kl':
      ax.set_ylim(0.0, 0.03)
    if col == 'ppo/pg_loss':
      ax.set_ylim(-0.04, 0.01)
    _style(ax, xlabel=ax in axes[-3:])
  axes[0].legend(loc='upper right', fontsize=8, framealpha=0.95)
  fig.suptitle(
      'Allegro actor-reset 300M — NF + PPO. × = last-layer actor-reset '
      '(ep_length_mean < 240). Train curves raw; eval off.',
      fontsize=11, fontweight='bold', y=0.995)
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  _save(fig, OUT)

  # ---- PPO internals + r_T-r_0 gap ----------------------------------------
  ppo_panels = (
      ('ep_length_mean',
       rf'episode length  ($\times$ if $<{RESET_EP_LEN:.0f}$)', False),
      ('ppo/pg_loss', r'clipped surrogate pg_loss  ($\times$ = reset)', True),
      ('ppo/policy_scale_mean',
       r'pre-tanh $\sigma$  (mean solid / min dashed)', False),
      ('ppo/approx_kl', r'approx KL  $E[r-1-\log r]$', False),
      ('advantage_mean', 'GAE mean  (pre z-score; dotted = std)', True),
      ('rT_r0_diff',
       r'$(r_T-r_0)$ succ $-$ fail  (NF diagnostic, not task success)', True),
  )
  fig2, axes2 = plt.subplots(2, 3, figsize=(14.2, 7.6), sharex=True)
  axes2 = axes2.ravel()
  reset_legend = True
  for ax, (col, title, hline0) in zip(axes2, ppo_panels):
    any_pts = False
    for log_dir, label, color, ls in RUNS:
      if col == 'rT_r0_diff':
        got = _curve_diff(
            log_dir,
            'repr_rT_minus_r0_mean_succ',
            'repr_rT_minus_r0_mean_fail')
      else:
        got = _curve(log_dir, col)
      if got is None:
        print(f'{label}: no {col}')
        continue
      xs, mean, n = got
      ax.plot(xs, mean, color=color, linestyle=ls, lw=1.7, alpha=0.95,
              label=label if ax is axes2[0] else None)
      any_pts = True
      print(f'{label} {col}: n={n} last={mean[-1]:.4g} steps={xs[-1]:.0f}')
      if col == 'ppo/policy_scale_mean':
        got_min = _curve(log_dir, 'ppo/policy_scale_min')
        if got_min is not None:
          xs_m, ym, _ = got_min
          ax.plot(xs_m, ym, color=color, linestyle='--', lw=1.15, alpha=0.85)
      if col == 'advantage_mean':
        got_s = _curve(log_dir, 'advantage_std')
        if got_s is not None:
          xs_s, ys, _ = got_s
          ax.plot(xs_s, ys, color=color, linestyle=':', lw=1.1, alpha=0.7)
      y_col = 'ep_length_mean' if col == 'rT_r0_diff' else col
      if y_col in RESET_MARK_COLS or col == 'rT_r0_diff':
        mark_col = col if col != 'rT_r0_diff' else 'ep_length_mean'
        if col == 'rT_r0_diff':
          # Place × at reset steps on the gap curve (y = gap at nearest x).
          rx, _ = _reset_points(log_dir, 'ep_length_mean')
          if rx:
            gmap = dict(zip(xs, mean))
            ry = [gmap[min(gmap, key=lambda t: abs(t - x))] for x in rx]
            ax.scatter(rx, ry, color=color, marker='x', s=28, linewidths=1.1,
                       zorder=5, label='actor-reset (ep<240)' if reset_legend else None)
            reset_legend = False
        else:
          n_r = _mark_resets(ax, log_dir, color, mark_col, legend=reset_legend)
          if n_r:
            reset_legend = False
    if not any_pts:
      ax.text(0.5, 0.5, f'no {col}', ha='center', va='center',
              transform=ax.transAxes, color='0.5')
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    if hline0:
      ax.axhline(0.0, color='0.55', lw=0.7)
    if col == 'ep_length_mean':
      ax.axhline(NOMINAL_EP, color='0.55', lw=0.7, linestyle='--')
      ax.axhline(RESET_EP_LEN, color='0.55', lw=1.1, linestyle=':')
    if col == 'ppo/policy_scale_mean':
      ax.axhline(1e-5, color='0.55', lw=0.7, linestyle=':')
    if col == 'advantage_mean':
      ax.set_ylim(-1.0, 1.0)
    if col == 'ppo/approx_kl':
      ax.set_ylim(0.0, 0.03)
    if col == 'ppo/pg_loss':
      ax.set_ylim(-0.04, 0.01)
    _style(ax, xlabel=ax in axes2[-3:])
  axes2[0].legend(loc='lower right', fontsize=8, framealpha=0.95)
  fig2.suptitle(
      r'Allegro actor-reset — PPO internals. $\times$ = last-layer reset. '
      'Dotted on GAE = std.',
      fontsize=11, fontweight='bold', y=0.995)
  fig2.tight_layout(rect=[0, 0, 1, 0.94])
  _save(fig2, OUT_PPO)

  # ---- ∇_s log p  (logged even with nf_grad_reg off / λ=0) ----------------
  grad_panels = (
      ('nf/nf_logp_grad_s_norm_mean',
       r'mean $\|\nabla_s \log p\|$  (c=100 dashed)', False),
      ('nf/nf_logp_grad_s_norm_max',
       r'max $\|\nabla_s \log p\|$', False),
      ('nf/nf_logp_grad_s_frac_above_c',
       r'frac $\|\nabla_s \log p\| > c{=}100$', False),
      ('nf/nf_grad_reg_raw',
       r'hinge mean $([\|score\|-c]_+)$  (not in loss, $\lambda=0$)', False),
  )
  figg, axesg = plt.subplots(2, 2, figsize=(11.6, 7.2), sharex=True)
  axesg = axesg.ravel()
  for ax, (col, title, hline0) in zip(axesg, grad_panels):
    any_pts = False
    for log_dir, label, color, ls in RUNS:
      got = _curve(log_dir, col)
      if got is None:
        print(f'{label}: no {col}')
        continue
      xs, mean, n = got
      ax.plot(xs, mean, color=color, linestyle=ls, lw=1.7, alpha=0.95,
              label=label if ax is axesg[0] else None)
      any_pts = True
      print(f'{label} {col}: n={n} last={mean[-1]:.4g} steps={xs[-1]:.0f}')
    if not any_pts:
      ax.text(0.5, 0.5, f'no {col}', ha='center', va='center',
              transform=ax.transAxes, color='0.5')
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    if hline0:
      ax.axhline(0.0, color='0.55', lw=0.7)
    if col == 'nf/nf_logp_grad_s_norm_mean':
      ax.axhline(GRAD_REG_C, color='0.55', lw=0.8, linestyle='--')
    if col == 'nf/nf_logp_grad_s_frac_above_c':
      ax.set_ylim(0.0, 1.0)
    _style(ax, xlabel=ax in axesg[-2:])
  axesg[0].legend(loc='upper left', fontsize=8, framealpha=0.95)
  figg.suptitle(
      r'Allegro actor-reset — NF $\nabla_s\log p$ (gradreg OFF, $\lambda=0$). '
      r'Threshold $c=100$ is the default hinge, not applied.',
      fontsize=11, fontweight='bold', y=0.995)
  figg.tight_layout(rect=[0, 0, 1, 0.94])
  _save(figg, OUT_GRAD)

  # ---- per-dim executed actions from latest.pkl replay --------------------
  fig3, (ax_m, ax_a) = plt.subplots(2, 1, figsize=(12.4, 7.2), sharex=True)
  xs = np.arange(len(ACTION_DIM_LABELS))
  any_act = False
  for log_dir, label, color, ls in RUNS:
    stats = _replay_action_stats(log_dir)
    if stats is None:
      continue
    any_act = True
    mu, sd, mag, n = stats['mean'], stats['std'], stats['mean_abs'], stats['n']
    print(f'{label} action n={n} mean_abs={mag.mean():.3f} '
          f'arm={mu[:ARM_DIMS].mean():.3f} hand={mu[ARM_DIMS:].mean():.3f}')
    ax_m.plot(xs, mu, color=color, linestyle=ls, lw=1.6, marker='o', ms=4,
              label=f'{label}  (n={n/1e6:.1f}M)')
    ax_m.fill_between(xs, mu - sd, mu + sd, color=color, alpha=0.12, lw=0)
    ax_a.plot(xs, mag, color=color, linestyle=ls, lw=1.6, marker='o', ms=4)
  ax_m.axhline(0.0, color='0.55', lw=0.7)
  ax_m.axvline(ARM_DIMS - 0.5, color='0.6', lw=0.8, linestyle='--')
  ax_a.axvline(ARM_DIMS - 0.5, color='0.6', lw=0.8, linestyle='--')
  ax_m.set_ylim(-1.05, 1.05)
  ax_a.set_ylim(0.0, 1.05)
  ax_m.set_ylabel(r'mean $a$  (band = $\pm$1 std)')
  ax_a.set_ylabel(r'mean $|a|$')
  ax_a.set_xticks(xs)
  ax_a.set_xticklabels(ACTION_DIM_LABELS, rotation=60, ha='right', fontsize=8)
  ax_m.set_title(
      'Executed action per dim  (latest.pkl replay, stochastic samples)',
      fontsize=10, fontweight='bold')
  ax_a.set_title(r'mean $|a|$ per dim', fontsize=10, fontweight='bold')
  ax_m.text(2.5, 0.92, 'arm (iiwa 1–7)', ha='center', fontsize=8, color='0.35')
  ax_m.text(14.5, 0.92, 'hand (index / mid / ring / thumb × 4)',
            ha='center', fontsize=8, color='0.35')
  for ax in (ax_m, ax_a):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
  if any_act:
    ax_m.legend(loc='lower right', fontsize=8, framealpha=0.95)
  else:
    ax_m.text(0.5, 0.5, 'no replay actions in latest.pkl', ha='center',
              va='center', transform=ax_m.transAxes, color='0.5')
  fig3.suptitle(
      'Allegro actor-reset — avg action along each of 23 PD dims. '
      'End-of-training replay only (milestones have no action buffer).',
      fontsize=11, fontweight='bold', y=0.995)
  fig3.tight_layout(rect=[0, 0, 1, 0.95])
  _save(fig3, OUT_ACT)


if __name__ == '__main__':
  main()
