"""Render BuilderBench traj video with live online φ·ψ reward strip.

Produces a stacked frame: reward timeline on top + MuJoCo render below, with
a moving cursor tracking the current (online) reward.

Examples:
  # Single checkpoint
  python scripts/render_frozen_crl_traj_reward_video.py \\
      --checkpoint=.../ckpt_iter_0001800.pkl --tag=c3t1_ckpt1800

  # All milestones in a run (reuses compile)
  python scripts/render_frozen_crl_traj_reward_video.py \\
      --checkpoint_dir=.../checkpoints --tag_prefix=c3t1_crl_stateonly \\
      --allow_no_success
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BB = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BB not in sys.path:
  sys.path.insert(0, _BB)

import sgcrl_jax_acme_compat  # noqa: F401

import importlib.util as _ilu

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

from contrastive import ContrastiveConfig
from contrastive import ppo_learner
from envs.builderbench_utils import (
    filter_pd_policy_state_obs,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

# Lazy-load BuilderBench rollout helpers so Sawyer/MetaWorld scripts can import
# strip/compose utilities from this module without pulling mujoco/builderbench.
VIDEO_FIXED_START_X = 0.1
_bbv = None
_build_networks = None
_load_train_ctx = None
_make_bb_env = None
_make_policy_fn = None
_maybe_fix_target = None
_run_config_path_for_checkpoint = None
force_video_nopermute_norand = None


def _ensure_bbv() -> None:
  """Load ``ppo_builderbench_rollout_video`` on first BuilderBench use."""
  global _bbv, _build_networks, _load_train_ctx, _make_bb_env
  global _make_policy_fn, _maybe_fix_target, _run_config_path_for_checkpoint
  global force_video_nopermute_norand, VIDEO_FIXED_START_X
  if _bbv is not None:
    return
  name = 'ppo_builderbench_rollout_video'
  if name in sys.modules:
    _bbv = sys.modules[name]
  else:
    _bbv_path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_rollout_video.py')
    _bbv_spec = _ilu.spec_from_file_location(name, _bbv_path)
    _bbv = _ilu.module_from_spec(_bbv_spec)
    assert _bbv_spec.loader is not None
    _bbv_spec.loader.exec_module(_bbv)
  _build_networks = _bbv._build_networks
  _load_train_ctx = _bbv._load_train_ctx
  _make_bb_env = _bbv._make_bb_env
  _make_policy_fn = _bbv._make_policy_fn
  _maybe_fix_target = _bbv._maybe_fix_target
  _run_config_path_for_checkpoint = _bbv._run_config_path_for_checkpoint
  force_video_nopermute_norand = _bbv.force_video_nopermute_norand
  VIDEO_FIXED_START_X = _bbv.VIDEO_FIXED_START_X


def _apply_hit_bonus_from_run_config(cfg: ContrastiveConfig, ckpt_path: str) -> str:
  """Copy ``ppo_crl_hit_bonus*`` from run_config into ``cfg``. Returns hit mode."""
  _ensure_bbv()
  cfg_path = _run_config_path_for_checkpoint(ckpt_path)
  if not cfg_path:
    return ''
  with open(cfg_path, 'r', encoding='utf-8') as fh:
    run_cfg = json.load(fh)
  flags = run_cfg.get('flags', {}) or {}
  resolved = run_cfg.get('resolved_config', {}) or {}
  hit = str(flags.get(
      'ppo_crl_hit_bonus',
      resolved.get('ppo_crl_hit_bonus', '')) or '').strip().lower()
  if hit not in ('', 'sf', 'goal'):
    print(f'[vid] WARNING: ignoring unknown ppo_crl_hit_bonus={hit!r}',
          flush=True)
    return ''
  cfg.ppo_crl_hit_bonus = hit
  tol = flags.get('ppo_crl_hit_bonus_tol',
                  resolved.get('ppo_crl_hit_bonus_tol', 0.01))
  scale = flags.get('ppo_crl_hit_bonus_scale',
                    resolved.get('ppo_crl_hit_bonus_scale', 1.0))
  if tol is not None and float(tol) >= 0.0:
    cfg.ppo_crl_hit_bonus_tol = float(tol)
  if scale is not None and float(scale) >= 0.0:
    cfg.ppo_crl_hit_bonus_scale = float(scale)
  if hit == 'goal':
    g = resolved.get('ppo_crl_hit_bonus_goal')
    if g is None:
      g = run_cfg.get('fixed_start_end')
    if g is None:
      raise ValueError(
          f"run_config {cfg_path} has ppo_crl_hit_bonus=goal but no "
          f"ppo_crl_hit_bonus_goal / fixed_start_end")
    cfg.ppo_crl_hit_bonus_goal = np.asarray(g, dtype=np.float32).reshape(-1)
  print(f'[vid] hit_bonus mode={hit!r} tol={cfg.ppo_crl_hit_bonus_tol} '
        f'scale={cfg.ppo_crl_hit_bonus_scale} '
        f'task_goal_set={getattr(cfg, "ppo_crl_hit_bonus_goal", None) is not None}',
        flush=True)
  return hit


def _reward_title_ylabel(hit_mode: str, state_only: bool, tag: str,
                         title_override: str = ''):
  if title_override:
    return f'{title_override}  ·  {tag}', r'$r$'
  phi = r'\varphi(s)' if state_only else r'\varphi(s,a)'
  base = rf'{phi}\cdot\psi(g)'
  if hit_mode == 'sf':
    title = (
        'online CRL  $r=' + base
        + r'+1\{\|s_g-g\|<\mathrm{tol}\}$  ·  ' + tag)
    ylabel = rf'$r={base}+1_{{\mathrm{{sf}}}}$'
  elif hit_mode == 'goal':
    title = (
        'online CRL  $r=' + base
        + r'+1\{\|s_g-g^\star\|<\mathrm{tol}\}$  ·  ' + tag)
    ylabel = rf'$r={base}+1_{{\mathrm{{goal}}}}$'
  else:
    title = rf'online CRL  $r={base}$  ·  {tag}'
    ylabel = rf'$r={base}$'
  return title, ylabel

import argparse

DEFAULT_ENV = 'builderbench_creative_3_task1'
DEFAULT_CKPT = (
    'logs/ppo_builderbench_creative3_task1_e1024_pd_crl_tau05_catselect_extrew1/'
    'ppo_builderbench_creative_3_task1_0/checkpoints/latest.pkl')
OUT_DIR = 'figs/builderbench/frozen_crl_reward_probe'
SEED = 0
MAX_TRIES = 1  # single det-policy rollout (env reset seed only)
FPS = 8
HOLD_LAST = 8


def _font(size: int):
  for path in (
      '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
      '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
      '/usr/share/fonts/liberation/LiberationSans-Bold.ttf',
  ):
    if os.path.isfile(path):
      return ImageFont.truetype(path, size=size)
  return ImageFont.load_default()


def _compile_rollout_and_states(policy, env, ep_len, fixed_goal, mocap_targets,
                                num_cubes, filter_policy_obs):
  _ensure_bbv()
  @jax.jit
  def _run(key):
    env_key, key = jax.random.split(key)
    state = env.reset(jax.random.split(env_key, 1))
    state = _maybe_fix_target(state, fixed_goal, mocap_targets, num_cubes)

    def step(carry, _):
      state, key = carry
      key, act_key = jax.random.split(key)
      goals = state.info['target_goal']
      action, _ = policy(state.obs, goals, act_key)
      if filter_policy_obs:
        policy_obs = filter_pd_policy_state_obs(state.obs, num_cubes)
      else:
        policy_obs = state.obs
      packed = jnp.concatenate([policy_obs, goals], axis=-1)
      next_state = env.step(state, action)
      next_state = _maybe_fix_target(
          next_state, fixed_goal, mocap_targets, num_cubes)
      succ = jnp.asarray(next_state.metrics['success']).reshape(-1)[0]
      out = {
          'packed': packed[0],
          'action': action[0],
          'success': succ,
      }
      return (next_state, key), (out, next_state)

    _, (traj, states) = jax.lax.scan(step, (state, key), (), length=ep_len)
    return traj, states

  return _run


def make_phi_psi_grad_norm_fn(networks, cfg):
  """‖∇_s (φ(s,a)·ψ(g))‖ and ‖∇_a (φ(s,a)·ψ(g))‖, g held fixed.

  Matches ``make_reward_fn`` for the raw CRL dot product (no hit bonus).
  ``ψ(g)`` does not depend on ``s`` or ``a``, so these are the encoder
  Jacobian norms of φ against the frozen goal embedding.
  """
  obs_dim = int(cfg.obs_dim)

  def _dot(q_params, s, a, g):
    packed = jnp.concatenate([s, g], axis=-1)[None]
    _, phi, psi = networks.q_network.apply(q_params, packed, a[None])
    return jnp.sum(phi * psi)

  @jax.jit
  def grad_norm_fn(q_params, packed, action):
    s = packed[:, :obs_dim]
    g = packed[:, obs_dim:]

    def one(s_i, a_i, g_i):
      gs = jax.grad(_dot, argnums=1)(q_params, s_i, a_i, g_i)
      ga = jax.grad(_dot, argnums=2)(q_params, s_i, a_i, g_i)
      return jnp.linalg.norm(gs), jnp.linalg.norm(ga)

    gs_n, ga_n = jax.vmap(one)(s, action, g)
    return gs_n, ga_n

  return grad_norm_fn


def _render_reward_strip(
    rewards: np.ndarray,
    success: np.ndarray,
    t: int,
    width: int,
    height: int = 220,
    *,
    title: str = '',
    ylabel: str = r'$r=\varphi\!\cdot\!\psi$',
    line_color: str = '#5ec8ff',
    ylim: tuple[float, float] | None = None,
) -> np.ndarray:
  T = len(rewards)
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  r_now = float(rewards[t])
  if ylim is None:
    r_min = float(np.min(rewards)) - 0.35
    r_max = float(np.max(rewards)) + 0.35
  else:
    r_min, r_max = float(ylim[0]), float(ylim[1])

  dpi = 120
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi,
                   facecolor='#0f1419')
  ax = fig.add_axes([0.07, 0.22, 0.72, 0.62])
  ax.set_facecolor('#0f1419')
  xs = np.arange(T)
  ax.plot(xs, rewards, color='#5a6a7a', lw=1.6, alpha=0.45, zorder=1)
  ax.plot(xs[: t + 1], rewards[: t + 1], color=line_color, lw=2.4, zorder=2)
  ax.fill_between(xs[: t + 1], rewards[: t + 1], r_min,
                   color=line_color, alpha=0.12, zorder=1)
  ax.scatter([t], [r_now], s=55, color='#ffe566', edgecolors='#1a1a1a',
             linewidths=0.8, zorder=4)
  ax.axvline(t, color='#ffe566', ls=':', lw=1.0, alpha=0.7, zorder=3)
  if first_succ >= 0:
    ax.axvline(first_succ, color='#ff8a4c', ls='--', lw=1.2, alpha=0.85,
               zorder=2)
    if t >= first_succ:
      ax.scatter([first_succ], [rewards[first_succ]], s=40,
                 color='#ff8a4c', zorder=3)
  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(r_min, r_max)
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel(ylabel, color='#c8d0d8', fontsize=9)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6)

  ax_txt = fig.add_axes([0.80, 0.22, 0.18, 0.62])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  succ_now = bool(success[t] >= 0.5)
  ax_txt.text(0.05, 0.78, 'reward now', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.52, f'{r_now:+.3f}', transform=ax_txt.transAxes,
              color='#ffe566' if not succ_now else '#7dffb0',
              fontsize=16, fontweight='bold', va='center',
              family='DejaVu Sans')
  ax_txt.text(0.05, 0.28, f't = {t}/{T - 1}', transform=ax_txt.transAxes,
              color='#c8d0d8', fontsize=10, va='center')
  ax_txt.text(0.05, 0.10,
              'SUCCESS' if succ_now else 'no success',
              transform=ax_txt.transAxes,
              color='#7dffb0' if succ_now else '#ff8a4c',
              fontsize=11, fontweight='bold', va='center')
  fig.suptitle(title or r'online CRL reward  $r=\varphi\!\cdot\!\psi$',
               color='#e8eef4', fontsize=10, y=0.96)
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def _render_grad_norm_strip(
    grad_s: np.ndarray,
    grad_a: np.ndarray,
    success: np.ndarray,
    t: int,
    width: int,
    height: int = 220,
    *,
    title: str = '',
    ylabel_s: str = r'$\Vert\nabla_s(\varphi\cdot\psi)\Vert$',
    ylabel_a: str = r'$\Vert\nabla_a(\varphi\cdot\psi)\Vert$',
) -> np.ndarray:
  """Episode timeline of ‖∇_s r‖ and ‖∇_a r‖ (twin axes)."""
  T = len(grad_s)
  gs_now = float(grad_s[t])
  ga_now = float(grad_a[t])
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  dpi = 120
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi,
                   facecolor='#0f1419')
  ax = fig.add_axes([0.07, 0.22, 0.72, 0.62])
  ax.set_facecolor('#0f1419')
  xs = np.arange(T)
  ax.plot(xs, grad_s, color='#3a6a7a', lw=1.6, alpha=0.45, zorder=1)
  ax.plot(xs[: t + 1], grad_s[: t + 1], color='#5ec8ff', lw=2.4, zorder=2)
  ax.scatter([t], [gs_now], s=55, color='#ffe566', edgecolors='#1a1a1a',
             linewidths=0.8, zorder=4)
  ax.axvline(t, color='#ffe566', ls=':', lw=1.0, alpha=0.7, zorder=3)
  if first_succ >= 0:
    ax.axvline(first_succ, color='#ff8a4c', ls='--', lw=1.2, alpha=0.85,
               zorder=2)
  gs_min = float(np.min(grad_s))
  gs_max = float(np.max(grad_s))
  gs_pad = 0.08 * max(gs_max - gs_min, 1e-6)
  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(gs_min - gs_pad, gs_max + gs_pad)
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel(ylabel_s, color='#5ec8ff', fontsize=9)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  ax.tick_params(axis='y', colors='#5ec8ff')
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6)

  ax2 = ax.twinx()
  ax2.plot(xs, grad_a, color='#6a4a7a', lw=1.4, alpha=0.45, zorder=1)
  ax2.plot(xs[: t + 1], grad_a[: t + 1], color='#e0c3ff', lw=2.2, zorder=2)
  ax2.scatter([t], [ga_now], s=40, color='#e0c3ff', edgecolors='#1a1a1a',
              linewidths=0.6, zorder=4)
  ga_min = float(np.min(grad_a))
  ga_max = float(np.max(grad_a))
  ga_pad = 0.08 * max(ga_max - ga_min, 1e-6)
  ax2.set_ylim(ga_min - ga_pad, ga_max + ga_pad)
  ax2.set_ylabel(ylabel_a, color='#e0c3ff', fontsize=9)
  ax2.tick_params(colors='#e0c3ff', labelsize=8)

  ax_txt = fig.add_axes([0.80, 0.22, 0.18, 0.62])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  ax_txt.text(0.05, 0.88, r'$\Vert\nabla_s\Vert$', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.70, f'{gs_now:.3f}', transform=ax_txt.transAxes,
              color='#5ec8ff', fontsize=14, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.48, r'$\Vert\nabla_a\Vert$', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.30, f'{ga_now:.3f}', transform=ax_txt.transAxes,
              color='#e0c3ff', fontsize=14, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.10, f't = {t}/{T - 1}', transform=ax_txt.transAxes,
              color='#c8d0d8', fontsize=10, va='center')
  fig.suptitle(
      title or r'$\Vert\nabla_{s,a}(\varphi(s,a)\cdot\psi(g))\Vert$',
      color='#e8eef4', fontsize=10, y=0.96)
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def _render_policy_loc_scale_strip(
    scale_mean: np.ndarray,
    scale_min: np.ndarray,
    loc_abs_mean: np.ndarray,
    success: np.ndarray,
    t: int,
    width: int,
    height: int = 220,
    *,
    title: str = '',
) -> np.ndarray:
  """Pre-tanh policy σ (mean/min over xyz+yaw) and mean |μ| vs macro step."""
  T = len(scale_mean)
  sm = float(scale_mean[t])
  smin = float(scale_min[t])
  lm = float(loc_abs_mean[t])
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  dpi = 120
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi,
                   facecolor='#0f1419')
  ax = fig.add_axes([0.07, 0.22, 0.72, 0.62])
  ax.set_facecolor('#0f1419')
  xs = np.arange(T)
  ax.plot(xs, scale_mean, color='#3a6a7a', lw=1.4, alpha=0.4, zorder=1)
  ax.plot(xs[: t + 1], scale_mean[: t + 1], color='#5ec8ff', lw=2.2, zorder=2)
  ax.plot(xs, scale_min, color='#6a3a2a', lw=1.2, alpha=0.4, zorder=1)
  ax.plot(xs[: t + 1], scale_min[: t + 1], color='#E8834C', lw=2.0, zorder=2)
  ax.scatter([t], [sm], s=50, color='#ffe566', edgecolors='#1a1a1a',
             linewidths=0.8, zorder=4)
  ax.axvline(t, color='#ffe566', ls=':', lw=1.0, alpha=0.7, zorder=3)
  if first_succ >= 0:
    ax.axvline(first_succ, color='#ff8a4c', ls='--', lw=1.2, alpha=0.85,
               zorder=2)
  y0 = float(min(np.min(scale_mean), np.min(scale_min)))
  y1 = float(max(np.max(scale_mean), np.max(scale_min)))
  pad = 0.08 * max(y1 - y0, 1e-6)
  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(y0 - pad, y1 + pad)
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel(r'policy $\sigma$ (pre-tanh xyz+yaw)', color='#5ec8ff',
                fontsize=8)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  ax.tick_params(axis='y', colors='#5ec8ff')
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6)

  ax2 = ax.twinx()
  ax2.plot(xs, loc_abs_mean, color='#6a4a7a', lw=1.2, alpha=0.4, zorder=1)
  ax2.plot(xs[: t + 1], loc_abs_mean[: t + 1], color='#e0c3ff', lw=2.0,
           zorder=2)
  ax2.scatter([t], [lm], s=36, color='#e0c3ff', edgecolors='#1a1a1a',
              linewidths=0.6, zorder=4)
  l0 = float(np.min(loc_abs_mean))
  l1 = float(np.max(loc_abs_mean))
  lpad = 0.08 * max(l1 - l0, 1e-6)
  ax2.set_ylim(l0 - lpad, l1 + lpad)
  ax2.set_ylabel(r'mean $|\mu|$', color='#e0c3ff', fontsize=8)
  ax2.tick_params(colors='#e0c3ff', labelsize=8)

  ax_txt = fig.add_axes([0.80, 0.22, 0.18, 0.62])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  ax_txt.text(0.05, 0.92, r'$\sigma$ mean', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=8, va='center')
  ax_txt.text(0.05, 0.78, f'{sm:.3f}', transform=ax_txt.transAxes,
              color='#5ec8ff', fontsize=13, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.60, r'$\sigma$ min', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=8, va='center')
  ax_txt.text(0.05, 0.46, f'{smin:.3f}', transform=ax_txt.transAxes,
              color='#E8834C', fontsize=13, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.28, r'mean $|\mu|$', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=8, va='center')
  ax_txt.text(0.05, 0.14, f'{lm:.3f}', transform=ax_txt.transAxes,
              color='#e0c3ff', fontsize=13, fontweight='bold', va='center')
  fig.suptitle(
      title or r'policy $\mu,\sigma$ (mode-cube xyz+yaw)',
      color='#e8eef4', fontsize=10, y=0.96)
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def _render_gae_strip(
    advantage: np.ndarray,
    value: np.ndarray,
    success: np.ndarray,
    t: int,
    width: int,
    height: int = 220,
    *,
    title: str = '',
) -> np.ndarray:
  """Raw GAE A_t (return-std reward, no minibatch-norm) and V(s_t)."""
  T = len(advantage)
  a_now = float(advantage[t])
  v_now = float(value[t])
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  dpi = 120
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi,
                   facecolor='#0f1419')
  ax = fig.add_axes([0.07, 0.22, 0.72, 0.62])
  ax.set_facecolor('#0f1419')
  xs = np.arange(T)
  ax.axhline(0.0, color='#c8d0d8', ls='-', lw=0.8, alpha=0.55, zorder=1)
  ax.plot(xs, advantage, color='#2a5a4a', lw=1.6, alpha=0.45, zorder=2)
  ax.plot(xs[: t + 1], advantage[: t + 1], color='#5ee8a8', lw=2.4, zorder=3)
  ax.scatter([t], [a_now], s=55, color='#ffe566', edgecolors='#1a1a1a',
             linewidths=0.8, zorder=5)
  ax.axvline(t, color='#ffe566', ls=':', lw=1.0, alpha=0.7, zorder=4)
  if first_succ >= 0:
    ax.axvline(first_succ, color='#ff8a4c', ls='--', lw=1.2, alpha=0.85,
               zorder=3)
  a0 = float(np.min(advantage))
  a1 = float(np.max(advantage))
  pad = 0.08 * max(a1 - a0, 1e-6)
  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(a0 - pad, a1 + pad)
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel(r'GAE $A_t$ (raw)', color='#5ee8a8', fontsize=8)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  ax.tick_params(axis='y', colors='#5ee8a8')
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6)

  ax2 = ax.twinx()
  ax2.plot(xs, value, color='#3a5a7a', lw=1.2, alpha=0.4, zorder=1)
  ax2.plot(xs[: t + 1], value[: t + 1], color='#4C9BE8', lw=2.0, zorder=2)
  ax2.scatter([t], [v_now], s=36, color='#4C9BE8', edgecolors='#1a1a1a',
              linewidths=0.6, zorder=4)
  v0 = float(np.min(value))
  v1 = float(np.max(value))
  vpad = 0.08 * max(v1 - v0, 1e-6)
  ax2.set_ylim(v0 - vpad, v1 + vpad)
  ax2.set_ylabel(r'$V(s_t)$', color='#4C9BE8', fontsize=8)
  ax2.tick_params(colors='#4C9BE8', labelsize=8)

  ax_txt = fig.add_axes([0.80, 0.22, 0.18, 0.62])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  a_col = '#5ee8a8' if a_now >= 0.0 else '#E8834C'
  ax_txt.text(0.05, 0.88, r'$A_t$', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.70, f'{a_now:+.3f}', transform=ax_txt.transAxes,
              color=a_col, fontsize=14, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.48, r'$V(s_t)$', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.30, f'{v_now:.3f}', transform=ax_txt.transAxes,
              color='#4C9BE8', fontsize=14, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.10, f't = {t}/{T - 1}', transform=ax_txt.transAxes,
              color='#c8d0d8', fontsize=10, va='center')
  fig.suptitle(
      title or r'GAE $A_t$  (return-std $r$, no minibatch-norm)',
      color='#e8eef4', fontsize=10, y=0.96)
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def select_action_to_cube(select: np.ndarray, num_cubes: int,
                          select_scale: float = np.pi) -> np.ndarray:
  """Map continuous PD ``select_action`` ∈ [-1, 1] → cube index (BuilderBench)."""
  s = np.asarray(select, dtype=np.float64)
  x = float(select_scale) * s + np.pi
  bins = np.arange(1, int(num_cubes) + 1) * (2.0 * np.pi / float(num_cubes))
  idx = np.digitize(x, bins)
  return np.clip(idx, 0, int(num_cubes) - 1).astype(np.int32)


def _render_select_strip(
    select: np.ndarray,
    cube: np.ndarray,
    t: int,
    width: int,
    num_cubes: int,
    height: int = 180,
    *,
    title: str = 'state select_action',
) -> np.ndarray:
  """Timeline of PD state select (continuous) + decoded cube id."""
  T = len(select)
  s_now = float(select[t])
  c_now = int(cube[t])
  dpi = 120
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi,
                   facecolor='#0f1419')
  ax = fig.add_axes([0.07, 0.24, 0.72, 0.58])
  ax.set_facecolor('#0f1419')
  xs = np.arange(T)
  ax.step(xs, cube, where='post', color='#7dffb0', lw=2.2, zorder=2)
  ax.scatter([t], [c_now], s=55, color='#ffe566', edgecolors='#1a1a1a',
             linewidths=0.8, zorder=4)
  ax.axvline(t, color='#ffe566', ls=':', lw=1.0, alpha=0.7, zorder=3)
  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(-0.5, float(num_cubes) - 0.5)
  ax.set_yticks(list(range(int(num_cubes))))
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel('selected cube', color='#7dffb0', fontsize=9)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6, axis='x')

  ax2 = ax.twinx()
  ax2.plot(xs, select, color='#c49bff', lw=1.4, alpha=0.85, zorder=1)
  ax2.plot(xs[: t + 1], select[: t + 1], color='#e0c3ff', lw=2.0, zorder=2)
  ax2.set_ylim(-1.15, 1.15)
  ax2.set_ylabel(r'select $\in[-1,1]$', color='#c49bff', fontsize=9)
  ax2.tick_params(colors='#c49bff', labelsize=8)

  ax_txt = fig.add_axes([0.80, 0.24, 0.18, 0.58])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  ax_txt.text(0.05, 0.82, 'select now', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.58, f'{s_now:+.3f}', transform=ax_txt.transAxes,
              color='#e0c3ff', fontsize=15, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.34, 'cube', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.12, f'{c_now}', transform=ax_txt.transAxes,
              color='#7dffb0', fontsize=18, fontweight='bold', va='center')
  fig.suptitle(title, color='#e8eef4', fontsize=10, y=0.96)
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def _render_pos_delta_strip(
    step_delta: np.ndarray,
    t: int,
    width: int,
    *,
    selected_cube: np.ndarray | None = None,
    height: int = 200,
    title: str = r'per-cube $\Vert\Delta\mathrm{xyz}\Vert$ (reward state)',
) -> np.ndarray:
  """Timeline of per-cube step-to-step position change (meters).

  ``step_delta`` shape ``(T, num_cubes)``.  Selected cube (if given) is drawn
  thicker so PD creep on the active cube is obvious.
  """
  deltas = np.asarray(step_delta, dtype=np.float64)
  T, nc = deltas.shape
  now = deltas[t]
  colors = ['#5ec8ff', '#ff8a4c', '#7dffb0', '#e0c3ff', '#ffe566',
            '#ff6b9d', '#9ad0ff', '#c8d0d8']
  dpi = 120
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi,
                   facecolor='#0f1419')
  ax = fig.add_axes([0.07, 0.24, 0.72, 0.58])
  ax.set_facecolor('#0f1419')
  xs = np.arange(T)
  sel = (None if selected_cube is None
         else np.asarray(selected_cube, dtype=np.int32))
  for c in range(nc):
    col = colors[c % len(colors)]
    lw = 1.5
    if sel is not None:
      # Emphasize segments where this cube is selected.
      ax.plot(xs, deltas[:, c], color=col, lw=1.2, alpha=0.35, zorder=1)
      mask = (sel == c)
      if np.any(mask):
        y = np.where(mask, deltas[:, c], np.nan)
        ax.plot(xs, y, color=col, lw=2.6, alpha=0.95, zorder=3,
                label=f'c{c}')
      else:
        ax.plot([], [], color=col, lw=2.0, label=f'c{c}')
    else:
      ax.plot(xs, deltas[:, c], color=col, lw=1.8, alpha=0.9, zorder=2,
              label=f'c{c}')
    ax.scatter([t], [now[c]], s=40, color=col, edgecolors='#1a1a1a',
               linewidths=0.6, zorder=5)
  ax.axvline(t, color='#ffe566', ls=':', lw=1.0, alpha=0.7, zorder=4)
  y_max = float(np.max(deltas)) if np.any(deltas > 0) else 1e-3
  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(-0.02 * y_max, y_max * 1.15 + 1e-6)
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel(r'$\Vert\Delta xyz\Vert$ (m)', color='#c8d0d8', fontsize=9)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6)
  ax.legend(loc='upper left', fontsize=7, framealpha=0.25,
            labelcolor='#c8d0d8', ncol=min(nc, 5))

  ax_txt = fig.add_axes([0.80, 0.24, 0.18, 0.58])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  tot_now = float(np.sum(now))
  max_c = int(np.argmax(now))
  ax_txt.text(0.05, 0.86, 'Σ|Δ| now', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.66, f'{tot_now:.4f}', transform=ax_txt.transAxes,
              color='#ffe566', fontsize=14, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.42, 'max cube', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.22, f'c{max_c}={now[max_c]:.4f}',
              transform=ax_txt.transAxes,
              color=colors[max_c % len(colors)], fontsize=12,
              fontweight='bold', va='center')
  fig.suptitle(title, color='#e8eef4', fontsize=10, y=0.96)
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def cube_step_deltas_from_pos(cube_pos: np.ndarray) -> np.ndarray:
  """``(T, num_cubes, 3)`` → ``(T, num_cubes)`` step L2 deltas (0 at t=0)."""
  pos = np.asarray(cube_pos, dtype=np.float64)
  d = np.zeros(pos.shape[:2], dtype=np.float64)
  if pos.shape[0] > 1:
    d[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
  return d


def _obs_to_goal_np(states: np.ndarray, start_index: int, end_index: int,
                    goal_state_indices) -> np.ndarray:
  """NumPy mirror of ``ppo_learner._obs_to_goal_jax``."""
  if goal_state_indices is not None:
    return states[:, np.asarray(goal_state_indices, dtype=np.int32)]
  if int(end_index) == -1:
    return states[:, int(start_index):]
  return states[:, int(start_index):int(end_index)]


def compute_hit_series(
    packed: np.ndarray,
    *,
    obs_dim: int,
    start_index: int,
    end_index: int,
    goal_state_indices,
    hit_mode: str,
    hit_tol: float,
    task_goal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Per-step hit indicator and distance for CRL hit-bonus.

  Returns ``(hit, dist)`` each shape ``(T,)`` where
  ``hit = 1{dist < tol}`` and
  * ``sf``:   ``dist = ‖obs_to_goal(s) − g‖`` (packed goal)
  * ``goal``: ``dist = ‖obs_to_goal(s) − g*‖`` (task goal)
  """
  packed = np.asarray(packed, dtype=np.float64)
  s = packed[:, :obs_dim]
  g = packed[:, obs_dim:]
  s_goal = _obs_to_goal_np(s, start_index, end_index, goal_state_indices)
  mode = (hit_mode or '').strip().lower()
  if mode == 'goal':
    if task_goal is None:
      dist = np.full((packed.shape[0],), np.nan, dtype=np.float64)
    else:
      tg = np.asarray(task_goal, dtype=np.float64).reshape(-1)
      dist = np.linalg.norm(s_goal - tg[None, :], axis=-1)
  else:
    # default / 'sf'
    dist = np.linalg.norm(s_goal - g, axis=-1)
  hit = (dist < float(hit_tol)).astype(np.float32)
  return hit, dist.astype(np.float32)


def _render_hit_strip(
    hit: np.ndarray,
    dist: np.ndarray,
    t: int,
    width: int,
    *,
    tol: float,
    hit_mode: str = 'sf',
    height: int = 180,
    title: str = '',
) -> np.ndarray:
  """Timeline of ``1{‖s_g−g‖<tol}`` with distance on a twin axis."""
  T = len(hit)
  hit_now = float(hit[t])
  dist_now = float(dist[t])
  qualifies = hit_now >= 0.5
  mode = (hit_mode or 'sf').strip().lower()
  ind_label = (r'$1\{\|s_g-g^\star\|<\mathrm{tol}\}$' if mode == 'goal'
               else r'$1\{\|s_g-g\|<\mathrm{tol}\}$')
  dist_label = (r'$\|s_g-g^\star\|$' if mode == 'goal'
                else r'$\|s_g-g\|$')
  dpi = 120
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi,
                   facecolor='#0f1419')
  ax = fig.add_axes([0.07, 0.24, 0.72, 0.58])
  ax.set_facecolor('#0f1419')
  xs = np.arange(T)
  ax.step(xs, hit, where='post', color='#5a6a7a', lw=1.4, alpha=0.45, zorder=1)
  ax.step(xs[: t + 1], hit[: t + 1], where='post',
          color='#7dffb0', lw=2.4, zorder=2)
  ax.fill_between(xs[: t + 1], hit[: t + 1], 0.0, step='post',
                   color='#7dffb0', alpha=0.15, zorder=1)
  ax.scatter([t], [hit_now], s=55, color='#ffe566', edgecolors='#1a1a1a',
             linewidths=0.8, zorder=4)
  ax.axvline(t, color='#ffe566', ls=':', lw=1.0, alpha=0.7, zorder=3)
  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(-0.08, 1.15)
  ax.set_yticks([0.0, 1.0])
  ax.set_yticklabels(['0', '1'])
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel(ind_label, color='#7dffb0', fontsize=9)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6)

  ax2 = ax.twinx()
  finite = dist[np.isfinite(dist)]
  d_max = float(np.max(finite)) if finite.size else float(tol) * 2.0
  y_hi = max(d_max * 1.15, float(tol) * 2.5, 1e-3)
  ax2.plot(xs, dist, color='#ff8a4c', lw=1.3, alpha=0.55, zorder=1)
  ax2.plot(xs[: t + 1], dist[: t + 1], color='#ffb07a', lw=2.0, zorder=2)
  ax2.axhline(float(tol), color='#ff8a4c', ls='--', lw=1.2, alpha=0.9,
              zorder=2)
  ax2.set_ylim(-0.02 * y_hi, y_hi)
  ax2.set_ylabel(dist_label, color='#ff8a4c', fontsize=9)
  ax2.tick_params(colors='#ff8a4c', labelsize=8)

  ax_txt = fig.add_axes([0.80, 0.24, 0.18, 0.58])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  ax_txt.text(0.05, 0.86, 'hit now', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(0.05, 0.66, '1' if qualifies else '0',
              transform=ax_txt.transAxes,
              color='#7dffb0' if qualifies else '#ff8a4c',
              fontsize=18, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.42, dist_label + ' now', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=8, va='center')
  ax_txt.text(0.05, 0.22, f'{dist_now:.4f}', transform=ax_txt.transAxes,
              color='#ffb07a', fontsize=13, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.06, f'tol={float(tol):g}', transform=ax_txt.transAxes,
              color='#9aa7b5', fontsize=8, va='center')
  fig.suptitle(
      title or rf'hit bonus indicator  ·  tol={float(tol):g}',
      color='#e8eef4', fontsize=10, y=0.96)
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def _compose_frame(strip: np.ndarray, render: np.ndarray,
                   reward: float, success: bool,
                   extra_strips=None,
                   select_badge: tuple[float, int] | None = None) -> np.ndarray:
  strips = [strip]
  if extra_strips:
    strips.extend(list(extra_strips))
  rw = strips[0].shape[1]
  rh = int(round(render.shape[0] * (rw / render.shape[1])))
  render_r = np.asarray(
      Image.fromarray(render).resize((rw, rh), Image.Resampling.LANCZOS))
  im = Image.fromarray(render_r).convert('RGBA')
  draw = ImageDraw.Draw(im, 'RGBA')
  if select_badge is not None:
    s_val, c_val = select_badge
    badge = f'r_online = {reward:+.3f}   select = {s_val:+.3f}   cube = {c_val}'
  else:
    badge = f'r_online = {reward:+.3f}'
  font = _font(26 if select_badge is not None else 28)
  bb = draw.textbbox((0, 0), badge, font=font)
  pad_x, pad_y = 14, 8
  bw, bh = bb[2] - bb[0] + 2 * pad_x, bb[3] - bb[1] + 2 * pad_y
  outline = (125, 255, 176, 255) if success else (255, 229, 102, 255)
  draw.rounded_rectangle(
      [12, 12, 12 + bw, 12 + bh], radius=10,
      fill=(16, 22, 28, 210), outline=outline, width=2)
  draw.text((12 + pad_x, 12 + pad_y - 2), badge, fill=outline[:3], font=font)
  if success:
    sfont = _font(22)
    st = 'SUCCESS'
    sbb = draw.textbbox((0, 0), st, font=sfont)
    sw, sh = sbb[2] - sbb[0] + 20, sbb[3] - sbb[1] + 10
    sx = rw - sw - 12
    draw.rounded_rectangle(
        [sx, 12, sx + sw, 12 + sh], radius=8,
        fill=(20, 60, 40, 220), outline=(125, 255, 176, 255), width=2)
    draw.text((sx + 10, 14), st, fill=(125, 255, 176), font=sfont)
  return np.concatenate(
      [*strips, np.asarray(im.convert('RGB'))], axis=0)


def _write_mp4(frames, path: str, fps: int) -> None:
  os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
  old_ld = os.environ.pop('LD_LIBRARY_PATH', None)
  try:
    import imageio.v2 as imageio
    imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)
  finally:
    if old_ld is not None:
      os.environ['LD_LIBRARY_PATH'] = old_ld
  if not os.path.isfile(path) or os.path.getsize(path) < 1000:
    raise RuntimeError(f'video write failed: {path}')


def _enumerate_ckpts(checkpoint: str, checkpoint_dir: str,
                     ckpt_stride: int = 1):
  if checkpoint_dir:
    files = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'ckpt_iter_*.pkl')),
        key=lambda p: int(re.search(r'ckpt_iter_(\d+)\.pkl$', p).group(1)))
    stride = max(1, int(ckpt_stride))
    if stride > 1:
      files = files[::stride]
    out = []
    for p in files:
      m = re.search(r'ckpt_iter_(\d+)\.pkl$', p)
      out.append((f'iter_{int(m.group(1)):07d}', p))
    return out
  base = os.path.splitext(os.path.basename(checkpoint))[0]
  label = base[len('ckpt_'):] if base.startswith('ckpt_') else base
  return [(label, checkpoint)]


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', default=DEFAULT_CKPT)
  p.add_argument('--checkpoint_dir', default='',
                 help='If set, render every ckpt_iter_*.pkl in this dir.')
  p.add_argument('--ckpt_stride', type=int, default=1,
                 help='Keep every Nth milestone when using --checkpoint_dir '
                      '(e.g. 2 = every other checkpoint).')
  p.add_argument('--env', default=DEFAULT_ENV)
  p.add_argument('--out_dir', default=OUT_DIR)
  p.add_argument('--reward_ckpt', default='',
                 help='Optional separate ckpt for φ/ψ (default: policy ckpt).')
  p.add_argument('--tag', default='',
                 help='Output stem (single-ckpt mode).')
  p.add_argument('--tag_prefix', default='c3t1_crl_online',
                 help='Prefix for batch mode tags: {prefix}_{label}.')
  p.add_argument('--title', default='')
  p.add_argument('--allow_no_success', action='store_true')
  p.add_argument('--max_tries', type=int, default=MAX_TRIES)
  p.add_argument('--seed', type=int, default=SEED)
  p.add_argument('--fps', type=int, default=FPS)
  p.add_argument('--skip_existing', action='store_true')
  p.add_argument('--show_select', action='store_true',
                 help='Add a timeline strip for state select_action / cube id')
  p.add_argument('--show_pos_delta', action='store_true',
                 help='Add per-cube ||Δxyz|| strip from the reward state s')
  p.add_argument('--normalize_reward', action='store_true',
                 help='Also show a 2nd reward strip normalised by episode std.')
  p.add_argument('--show_phi_psi_grad', action='store_true',
                 help='Add ||∇_s (φ·ψ)|| and ||∇_a (φ·ψ)|| strips over the '
                      'episode (g held fixed)')
  p.add_argument('--show_hit', action='store_true',
                 help='Add 1{||s_g−g||<tol} (+ distance) strip; also on by '
                      'default whenever run_config has a hit-bonus mode')
  p.add_argument('--no_show_hit', action='store_true',
                 help='Disable the hit-indicator strip even if hit-bonus is on')
  p.add_argument('--match_run_init', action='store_true',
                 help='Use permute/fixed_start_x from run_config (default: '
                      'force nopermute + fixed_start_x=0.1)')
  p.add_argument('--fixed_start_x', type=float, default=VIDEO_FIXED_START_X,
                 help='Start-box x when forcing norand')
  return p.parse_args()


def _render_one(args, *, label, ckpt_path, ctx, networks, env, mocap_targets,
                ep_len, num_cubes, reward_fn, mean0, var1, out_dir,
                hit_mode: str = '',
                hit_tol: float = 1e-2,
                task_goal: np.ndarray | None = None,
                grad_norm_fn=None):
  tag = args.tag if (args.tag and not args.checkpoint_dir) else (
      f'{args.tag_prefix}_{label}')
  out_mp4 = os.path.join(out_dir, f'{tag}.mp4')
  still_path = os.path.join(out_dir, f'{tag}_still.png')
  csv_path = os.path.join(out_dir, f'{tag}.csv')
  if args.skip_existing and os.path.isfile(out_mp4):
    print(f'[vid] skip existing {out_mp4}', flush=True)
    return

  print(f'[vid] === {label} ({ckpt_path}) ===', flush=True)
  ckpt = ppo_learner.load_checkpoint(ckpt_path)
  reward_ckpt_path = args.reward_ckpt.strip() or ckpt_path
  rew_ckpt = (ckpt if reward_ckpt_path == ckpt_path
              else ppo_learner.load_checkpoint(reward_ckpt_path))
  # Online only (not tau-weighted EMA).
  q_params = rew_ckpt['q_params']
  print(f'[vid] policy_iter={ckpt.get("iteration")} ep_len={ep_len} '
        f'reward_src=q_params(online) hit_bonus={hit_mode!r}', flush=True)

  policy = _make_policy_fn(
      networks, ckpt['policy_params'], stochastic=False,
      filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes,
      normalize_obs=False, obs_dim=ctx.obs_dim,
      start_index=ctx.start_index, end_index=ctx.end_index)
  run = _compile_rollout_and_states(
      policy, env, ep_len, ctx.fixed_target_goal, mocap_targets, num_cubes,
      ctx.filter_policy_obs)

  key = jax.random.PRNGKey(args.seed)
  key, warm_key = jax.random.split(key)
  print('[vid] warming compile...', flush=True)
  _ = jax.block_until_ready(run(warm_key))
  print('[vid] compile done', flush=True)

  best = None
  for attempt in range(int(args.max_tries)):
    key, roll_key = jax.random.split(key)
    traj, states = run(roll_key)
    packed = np.asarray(traj['packed'], dtype=np.float32)
    actions = np.asarray(traj['action'], dtype=np.float32)
    succ = np.asarray(traj['success'], dtype=np.float32)
    rewards = np.asarray(
        reward_fn(q_params, jnp.asarray(packed), jnp.asarray(actions),
                  mean0, var1), dtype=np.float32)
    reached = bool(np.any(succ >= 0.5))
    print(f'[vid] try={attempt} success={reached} '
          f'rew_sum={rewards.sum():.2f} first_succ='
          f'{int(np.argmax(succ >= 0.5)) if reached else -1}', flush=True)
    sel = np.asarray(states.info['select_action'], dtype=np.float32)
    if sel.ndim > 1:
      sel = sel.reshape(sel.shape[0], -1)[:, 0]
    cur = dict(rewards=rewards, success=succ, states=states, select=sel,
               packed=packed, actions=actions, attempt=attempt)
    if reached:
      best = cur
      break
    if best is None or rewards.sum() > best['rewards'].sum():
      best = cur

  if best is None:
    raise RuntimeError('no trajectory collected')
  if (not np.any(best['success'] >= 0.5)) and (not args.allow_no_success):
    raise RuntimeError('no successful trajectory found '
                       '(pass --allow_no_success to keep best return)')

  rewards = best['rewards']
  success = best['success']
  states = best['states']
  select = np.asarray(best['select'], dtype=np.float32)
  cube = select_action_to_cube(select, num_cubes)
  packed = np.asarray(best['packed'], dtype=np.float32)
  actions = np.asarray(best['actions'], dtype=np.float32)
  cube_pos = packed[:, :3 * num_cubes].reshape(len(packed), num_cubes, 3)
  step_delta = cube_step_deltas_from_pos(cube_pos)
  show_hit = (not args.no_show_hit) and (
      bool(args.show_hit) or bool(hit_mode))
  hit = np.zeros(len(packed), dtype=np.float32)
  dist = np.full(len(packed), np.nan, dtype=np.float32)
  if show_hit and hit_mode:
    hit, dist = compute_hit_series(
        packed,
        obs_dim=int(ctx.obs_dim),
        start_index=int(ctx.start_index),
        end_index=int(ctx.end_index),
        goal_state_indices=getattr(ctx, 'goal_state_indices', None),
        hit_mode=hit_mode,
        hit_tol=float(hit_tol),
        task_goal=task_goal,
    )
  show_grad = bool(args.show_phi_psi_grad) and grad_norm_fn is not None
  grad_s = np.zeros(len(packed), dtype=np.float32)
  grad_a = np.zeros(len(packed), dtype=np.float32)
  if show_grad:
    gs, ga = grad_norm_fn(
        q_params, jnp.asarray(packed), jnp.asarray(actions))
    grad_s = np.asarray(gs, dtype=np.float32)
    grad_a = np.asarray(ga, dtype=np.float32)
    print(f'[vid] ||∇_s φ·ψ|| range=[{grad_s.min():.4g},{grad_s.max():.4g}] '
          f'||∇_a φ·ψ|| range=[{grad_a.min():.4g},{grad_a.max():.4g}]',
          flush=True)
  T = len(rewards)
  first_succ = (int(np.argmax(success >= 0.5))
                if np.any(success >= 0.5) else -1)
  first_hit = (int(np.argmax(hit >= 0.5)) if np.any(hit >= 0.5) else -1)
  print(f'[vid] using attempt={best["attempt"]} first_success_t={first_succ} '
        f'rew_sum={rewards.sum():.3f}', flush=True)
  print(f'[vid] select range=[{select.min():+.3f},{select.max():+.3f}] '
        f'cubes_used={sorted(set(int(x) for x in cube))}', flush=True)
  if show_hit and hit_mode:
    print(f'[vid] hit_mode={hit_mode!r} tol={hit_tol:g} '
          f'hit_frac={float(hit.mean()):.3f} first_hit_t={first_hit} '
          f'dist[min,max]=[{float(np.nanmin(dist)):.4f},'
          f'{float(np.nanmax(dist)):.4f}]', flush=True)
  if first_succ >= 0:
    t0 = first_succ
    print(f'[vid] pos Δ after first_succ t={t0}: '
          f'sum={step_delta[t0:].sum():.5f} '
          f'max_step={step_delta[t0:].max():.5f} m', flush=True)

  csv_cols = [
      np.arange(T), rewards, success,
      dist.astype(np.float32),
      select, cube.astype(np.float32),
      step_delta.sum(axis=1).astype(np.float32),
      hit.astype(np.float32),
      grad_s.astype(np.float32),
      grad_a.astype(np.float32),
  ]
  header = ('t,reward_online,success,dist,select_action,select_cube,'
            'pos_delta_sum,hit,grad_s_norm,grad_a_norm')
  for c in range(num_cubes):
    csv_cols.append(step_delta[:, c].astype(np.float32))
    header += f',pos_delta_c{c}'
  np.savetxt(
      csv_path, np.stack(csv_cols, axis=1), delimiter=',',
      header=header, comments='')

  print('[vid] rendering frames...', flush=True)
  renders = []
  for i in range(T):
    renders.append(np.asarray(env.render_from_info(
        np.asarray(states.data.qpos[i][0]),
        np.asarray(states.data.qvel[i][0]),
        np.asarray(states.info['target_mocap_pos'][i][0]),
        np.asarray(states.info['target_mocap_quat'][i][0]),
    )))
  width = int(renders[0].shape[1])
  if width % 2:
    width -= 1

  title, ylabel = _reward_title_ylabel(
      hit_mode, bool(getattr(ctx, 'crl_state_only', False)), tag,
      title_override=args.title)

  rewards_norm = rewards / (float(np.std(rewards)) + 1e-8)
  print('[vid] composing reward overlay...', flush=True)
  frames = []
  for t in range(T):
    strip = _render_reward_strip(
        rewards, success, t, width=width, height=220,
        title=title, ylabel=ylabel, ylim=None)
    extra = []
    select_badge = None
    if args.normalize_reward:
      extra.append(_render_reward_strip(
          rewards_norm, success, t, width=width, height=180,
          title=rf'normalised reward  $r/\sigma_r$  ·  {tag}',
          ylabel=r'$r/\sigma_r$',
          line_color='#ffb347'))
    if show_grad:
      extra.append(_render_grad_norm_strip(
          grad_s, grad_a, success, t, width=width, height=220,
          title=(r'$\Vert\nabla_s(\varphi(s,a)\cdot\psi(g))\Vert$  /  '
                 r'$\Vert\nabla_a(\varphi(s,a)\cdot\psi(g))\Vert$'
                 rf'  ·  {tag}')))
    if show_hit and hit_mode:
      if hit_mode == 'goal':
        hit_title = (
            rf'hit $1{{\|s_g-g^\star\|<\mathrm{{tol}}}}$'
            rf'  ·  tol={float(hit_tol):g}  ·  {tag}')
      else:
        hit_title = (
            rf'hit $1{{\|s_g-g\|<\mathrm{{tol}}}}$'
            rf'  ·  tol={float(hit_tol):g}  ·  {tag}')
      extra.append(_render_hit_strip(
          hit, dist, t, width=width, tol=float(hit_tol),
          hit_mode=hit_mode, height=180, title=hit_title))
    if args.show_select:
      extra.append(_render_select_strip(
          select, cube, t, width=width, num_cubes=num_cubes, height=180,
          title=rf'state select_action → cube  ·  {tag}'))
      select_badge = (float(select[t]), int(cube[t]))
    if args.show_pos_delta:
      extra.append(_render_pos_delta_strip(
          step_delta, t, width=width, selected_cube=cube, height=200,
          title=rf'per-cube $\Vert\Delta xyz\Vert$ in $s$  ·  {tag}'))
    composed = _compose_frame(
        strip, renders[t], float(rewards[t]), bool(success[t] >= 0.5),
        extra_strips=extra or None, select_badge=select_badge)
    h, w = composed.shape[:2]
    if h % 2 or w % 2:
      composed = composed[: h - (h % 2), : w - (w % 2)]
    frames.append(composed)
    if (t + 1) % 10 == 0 or t == T - 1:
      print(f'[vid]   framed {t + 1}/{T}', flush=True)
  frames.extend([frames[-1]] * HOLD_LAST)
  _write_mp4(frames, out_mp4, args.fps)
  still_t = min(first_succ + 2, T - 1) if first_succ >= 0 else T // 2
  Image.fromarray(frames[still_t]).save(still_path)
  print(f'[vid] wrote {out_mp4} ({os.path.getsize(out_mp4) / 1e6:.2f} MB)',
        flush=True)


def main():
  _ensure_bbv()
  args = _parse_args()
  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)
  entries = _enumerate_ckpts(
      args.checkpoint, args.checkpoint_dir, ckpt_stride=int(args.ckpt_stride))
  if not entries:
    raise FileNotFoundError('no checkpoints found')
  print(f'[vid] jax={jax.default_backend()} devices={jax.devices()}')
  print(f'[vid] {len(entries)} checkpoint(s) '
        f'(stride={max(1, int(args.ckpt_stride))}) → {out_dir}')

  first_path = entries[0][1]
  env_name = args.env
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, _ = parse_bb_env_id(env_id)
  ctx = _load_train_ctx(env_name, first_path)
  if not args.match_run_init:
    force_video_nopermute_norand(ctx, fixed_start_x=float(args.fixed_start_x))
    print(f'[vid] forcing nopermute + fixed_start_x={ctx.fixed_start_x}',
          flush=True)
  networks = _build_networks(env_name, seed=args.seed, ctx=ctx)
  env, _base, mocap_targets, ep_len = _make_bb_env(env_id, ctx)

  cfg = ContrastiveConfig()
  cfg.obs_dim = int(ctx.obs_dim)
  cfg.start_index = int(ctx.start_index)
  cfg.end_index = int(ctx.end_index)
  cfg.ppo_norm_obs = False
  hit_mode = _apply_hit_bonus_from_run_config(cfg, first_path)
  hit_tol = float(getattr(cfg, 'ppo_crl_hit_bonus_tol', 1e-2))
  task_goal = getattr(cfg, 'ppo_crl_hit_bonus_goal', None)
  if task_goal is not None:
    task_goal = np.asarray(task_goal, dtype=np.float32).reshape(-1)
  reward_fn = ppo_learner.make_reward_fn(networks, cfg)
  grad_norm_fn = make_phi_psi_grad_norm_fn(networks, cfg)
  mean0 = jnp.zeros((ctx.obs_dim,), dtype=jnp.float32)
  var1 = jnp.ones((ctx.obs_dim,), dtype=jnp.float32)

  for label, path in entries:
    _render_one(
        args, label=label, ckpt_path=path, ctx=ctx, networks=networks,
        env=env, mocap_targets=mocap_targets, ep_len=ep_len,
        num_cubes=num_cubes, reward_fn=reward_fn, mean0=mean0, var1=var1,
        out_dir=out_dir, hit_mode=hit_mode, hit_tol=hit_tol,
        task_goal=task_goal, grad_norm_fn=grad_norm_fn)


if __name__ == '__main__':
  main()
