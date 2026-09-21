#!/usr/bin/env python3
"""Re-roll Sawyer bin episodes and overlay mocap tracking error + actions.

Tracks ‖mocap_target − TCP‖ (meters) after each control step, plus object
speed and the 4-D MetaWorld action (dx, dy, dz, grip) in [-1, 1]. Useful
for choosing a hybrid bounded-mocap threshold and for checking whether
overshoot frames coincide with saturated or near-zero actions.

Example:
  python scripts/render_sawyer_mocap_tracking_error_video.py \\
      --checkpoint=logs/.../checkpoints/ckpt_iter_0034000.pkl \\
      --output=videos/.../sawyer_bin_iter_0034000_err.mp4 \\
      --seed=7 --stochastic \\
      --sawyer_bin_safe_grasp_reset --bin_randomize_gripper_init
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'scripts'))

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import env_utils
from contrastive import ppo_learner
import ppo_rollout_video as roll

_ACT_LABELS = ('dx', 'dy', 'dz', 'grip')
_ACT_COLORS = (
    (255, 95, 95),
    (90, 210, 110),
    (80, 170, 255),
    (230, 140, 255),
)
_ERR_STRIP_H = 110
_ACT_STRIP_H = 150


def _font(size: int = 16):
  for path in (
      '/usr/share/fonts/dejavu/DejaVuSansMono.ttf',
      '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
      '/usr/share/fonts/liberation/LiberationMono-Regular.ttf',
  ):
    if os.path.isfile(path):
      try:
        return ImageFont.truetype(path, size=size)
      except OSError:
        pass
  return ImageFont.load_default()


def _tracking_error(gym_env) -> float:
  mocap = np.asarray(gym_env.data.mocap_pos[0], dtype=np.float64)
  tcp = np.asarray(gym_env.get_endeff_pos(), dtype=np.float64)
  return float(np.linalg.norm(mocap - tcp))


def _time_xs(n: int, pad_l: int, plot_w: int) -> list[int]:
  return [pad_l + int(i * (plot_w - 1) / max(n - 1, 1)) for i in range(max(n, 1))]


def _draw_polyline(draw, xs, ys, valid, *, upto: int, color, width: int):
  pts = []
  last = min(upto, len(xs) - 1)
  for i in range(last + 1):
    if not valid[i]:
      if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)
      pts = []
      continue
    pts.append((xs[i], ys[i]))
  if len(pts) >= 2:
    draw.line(pts, fill=color, width=width)


def _draw_error_strip(
    draw, errors, obj_speeds, *, t: int, ymax: float, w: int, y_off: int,
    strip_h: int, font, font_sm):
  pad_l, pad_r, pad_t, pad_b = 8, 8, 22, 8
  plot_w = w - pad_l - pad_r
  plot_h = strip_h - pad_t - pad_b
  y0 = y_off + pad_t
  draw.rectangle(
      [pad_l, y0, pad_l + plot_w, y0 + plot_h], outline=(70, 70, 70), width=1)
  for frac, label in ((0.0, '0'), (0.5, f'{0.5 * ymax * 100:.0f}'),
                      (1.0, f'{ymax * 100:.0f}cm')):
    yy = y0 + plot_h - int(frac * plot_h)
    draw.line([(pad_l, yy), (pad_l + plot_w, yy)], fill=(45, 45, 45), width=1)
    draw.text((pad_l + 2, yy - 12), label, fill=(160, 160, 160), font=font_sm)

  for thr_m, color in ((0.02, (80, 180, 80)), (0.03, (220, 180, 40)),
                       (0.05, (220, 90, 90)), (0.15, (255, 150, 40)),
                       (0.20, (255, 60, 60))):
    if thr_m > ymax:
      continue
    yy = y0 + plot_h - int((thr_m / ymax) * plot_h)
    draw.line([(pad_l, yy), (pad_l + plot_w, yy)], fill=color, width=1)

  n = max(len(errors), 1)
  xs = _time_xs(n, pad_l, plot_w)
  ys = [
      y0 + plot_h - int(np.clip(e / ymax, 0.0, 1.0) * plot_h) for e in errors
  ]
  valid = [True] * n
  cur_i = int(np.clip(t, 0, n - 1))
  _draw_polyline(draw, xs, ys, valid, upto=n - 1, color=(65, 85, 105), width=2)
  _draw_polyline(
      draw, xs, ys, valid, upto=cur_i, color=(90, 180, 255), width=3)
  if xs:
    cursor_x = xs[cur_i]
    draw.line(
        [(cursor_x, y0), (cursor_x, y0 + plot_h)],
        fill=(255, 220, 80), width=1)
    r = 3
    draw.ellipse(
        [xs[cur_i] - r, ys[cur_i] - r, xs[cur_i] + r, ys[cur_i] + r],
        fill=(255, 220, 80))

  cur_e = errors[cur_i] if errors else float('nan')
  cur_v = obj_speeds[cur_i] if obj_speeds else float('nan')
  title = (
      f'Mocap-TCP error (cm) | step {cur_i}/{n - 1} | '
      f'current {cur_e * 100:.1f} cm | object speed {cur_v * 100:.1f} cm/s')
  draw.text((pad_l, y_off + 2), title, fill=(235, 235, 235), font=font)


def _action_to_y(val: float, y0: int, plot_h: int) -> int:
  frac = (float(np.clip(val, -1.0, 1.0)) + 1.0) / 2.0
  return y0 + plot_h - int(frac * plot_h)


def _draw_action_strip(
    draw, actions, *, t: int, n_frames: int, w: int, y_off: int,
    strip_h: int, font, font_sm):
  pad_l, pad_r, pad_t, pad_b = 8, 8, 22, 8
  plot_w = w - pad_l - pad_r
  plot_h = strip_h - pad_t - pad_b
  y0 = y_off + pad_t

  y_lo = _action_to_y(-0.1, y0, plot_h)
  y_hi = _action_to_y(0.1, y0, plot_h)
  if y_lo < y_hi:
    y_lo, y_hi = y_hi, y_lo
  draw.rectangle(
      [pad_l, y_hi, pad_l + plot_w, y_lo], fill=(42, 42, 28))

  draw.rectangle(
      [pad_l, y0, pad_l + plot_w, y0 + plot_h], outline=(70, 70, 70), width=1)
  for val, label in ((-1.0, '-1'), (0.0, '0'), (1.0, '+1')):
    yy = _action_to_y(val, y0, plot_h)
    draw.line([(pad_l, yy), (pad_l + plot_w, yy)], fill=(45, 45, 45), width=1)
    draw.text((pad_l + 2, yy - 12), label, fill=(160, 160, 160), font=font_sm)
  for val, color in ((-1.0, (220, 90, 90)), (1.0, (220, 90, 90))):
    yy = _action_to_y(val, y0, plot_h)
    draw.line([(pad_l, yy), (pad_l + plot_w, yy)], fill=color, width=1)

  n = max(n_frames, 1)
  xs = _time_xs(n, pad_l, plot_w)
  act = np.asarray(actions, dtype=np.float64)
  if act.ndim != 2 or act.shape[0] != n:
    act = np.full((n, 4), np.nan, dtype=np.float64)
    act[:min(n, len(actions))] = np.asarray(actions, dtype=np.float64)[
        :min(n, len(actions))]
  cur_i = int(np.clip(t, 0, n - 1))
  for d, color in enumerate(_ACT_COLORS):
    ys = [_action_to_y(act[i, d] if np.isfinite(act[i, d]) else 0.0, y0, plot_h)
          for i in range(n)]
    valid = [bool(np.isfinite(act[i, d])) for i in range(n)]
    dim = tuple(int(c * 0.35) for c in color)
    _draw_polyline(draw, xs, ys, valid, upto=n - 1, color=dim, width=2)
    _draw_polyline(draw, xs, ys, valid, upto=cur_i, color=color, width=3)
    if valid[cur_i]:
      r = 3
      draw.ellipse(
          [xs[cur_i] - r, ys[cur_i] - r, xs[cur_i] + r, ys[cur_i] + r],
          fill=color)

  if xs:
    cursor_x = xs[cur_i]
    draw.line(
        [(cursor_x, y0), (cursor_x, y0 + plot_h)],
        fill=(255, 220, 80), width=1)

  def _tw(s):
    return int(draw.textlength(s, font=font) if hasattr(draw, 'textlength') else 8 * len(s))

  prefix = 'Action [-1,1] | '
  draw.text((pad_l, y_off + 2), prefix, fill=(235, 235, 235), font=font)
  x = pad_l + _tw(prefix)
  cur = act[cur_i]
  if np.all(np.isfinite(cur)):
    for d, lab in enumerate(_ACT_LABELS):
      chunk = f'{lab}={cur[d]:+.2f} '
      draw.text((x, y_off + 2), chunk, fill=_ACT_COLORS[d], font=font)
      x += _tw(chunk)
    a_inf = float(np.max(np.abs(cur)))
    a_xyz = float(np.linalg.norm(cur[:3]))
    rest = f'| |a|∞={a_inf:.2f} ||xyz||={a_xyz:.2f}'
    draw.text((x, y_off + 2), rest, fill=(235, 235, 235), font=font)
  else:
    draw.text((x, y_off + 2), 'no action (reset)', fill=(180, 180, 180), font=font)


def _overlay_strips(
    frame: np.ndarray,
    errors: list[float],
    obj_speeds: list[float],
    actions: list,
    *,
    t: int,
    ymax: float,
) -> np.ndarray:
  """Mocap-error strip above the frame, action strip below."""
  h, w = frame.shape[:2]
  canvas = Image.new('RGB', (w, h + _ERR_STRIP_H + _ACT_STRIP_H), (18, 18, 18))
  canvas.paste(Image.fromarray(frame), (0, _ERR_STRIP_H))
  draw = ImageDraw.Draw(canvas)
  font = _font(15)
  font_sm = _font(13)
  _draw_error_strip(
      draw, errors, obj_speeds, t=t, ymax=ymax, w=w, y_off=0,
      strip_h=_ERR_STRIP_H, font=font, font_sm=font_sm)
  _draw_action_strip(
      draw, actions, t=t, n_frames=len(errors), w=w,
      y_off=_ERR_STRIP_H + h, strip_h=_ACT_STRIP_H, font=font, font_sm=font_sm)
  return np.asarray(canvas, dtype=np.uint8)


def _rollout_with_error(
    policy_params, gym_env, networks, render, *, max_steps: int,
    stochastic: bool, seed: int):
  @jax.jit
  def policy_mode(params, obs):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  @jax.jit
  def policy_sample(params, obs, rng):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample(dist, rng)

  np.random.seed(seed)
  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  errors = [_tracking_error(gym_env)]
  obj_speeds = [0.0]
  # Frame t=0 is reset (no action yet). Later frames store the action that
  # produced that state, so overshoot lines up with the command that caused it.
  actions = [np.full(4, np.nan, dtype=np.float32)]
  obj_prev = np.asarray(gym_env._get_pos_objects(), dtype=np.float64)
  # dt per control step ≈ frame_skip * model.opt.timestep
  dt = float(getattr(gym_env, 'dt', gym_env.frame_skip * gym_env.model.opt.timestep))
  frames_raw = [render()]
  total_reward = 0.0
  success = False
  rng = jax.random.PRNGKey(seed)
  for t in range(max_steps):
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = policy_sample(policy_params, obs[None], k)
    else:
      action_j = policy_mode(policy_params, obs[None])
    action = np.asarray(action_j)[0].astype(np.float32)
    obs_next, r, done, _info = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    total_reward += float(r)
    if float(r) > 0.0:
      success = True
    obj = np.asarray(gym_env._get_pos_objects(), dtype=np.float64)
    obj_speeds.append(float(np.linalg.norm(obj - obj_prev) / max(dt, 1e-6)))
    obj_prev = obj
    errors.append(_tracking_error(gym_env))
    actions.append(action.copy())
    frames_raw.append(render())
    if done:
      break
  return frames_raw, errors, obj_speeds, actions, dict(
      total_reward=total_reward, success=success, length=len(frames_raw),
      dt=dt)


def main() -> None:
  from ppo_contrastive import fixed_goal_dict
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--env', default='sawyer_bin', choices=list(fixed_goal_dict))
  p.add_argument('--output', required=True)
  p.add_argument('--csv_out', default='')
  p.add_argument('--width', type=int, default=640)
  p.add_argument('--height', type=int, default=480)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--rotate', type=int, default=180)
  p.add_argument('--camera', default='corner')
  p.add_argument('--stochastic', action='store_true')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--max_steps', type=int, default=-1)
  p.add_argument('--ymax_cm', type=float, default=10.0,
                 help='Vertical axis of the overlay strip (cm).')
  p.add_argument('--sawyer_randomize_init', action='store_true', default=True)
  p.add_argument('--no_sawyer_randomize_init', dest='sawyer_randomize_init',
                 action='store_false')
  p.add_argument('--sawyer_bin_safe_grasp_reset', action='store_true')
  p.add_argument('--bin_randomize_gripper_init', action='store_true')
  p.add_argument('--bin_randomize_tcp_z', action='store_true')
  p.add_argument('--sawyer_bin_bounded_step_mocap', action='store_true')
  p.add_argument('--sawyer_bin_selective_antiwindup', action='store_true')
  p.add_argument('--sawyer_bin_trackerr_terminate_20cm', action='store_true')
  p.add_argument('--sawyer_max_episode_steps', type=int, default=-1)
  args = p.parse_args()

  entries = roll._enumerate_checkpoints(args.checkpoint)
  if not entries:
    raise SystemExit(f'no checkpoints at {args.checkpoint}')
  if len(entries) != 1:
    raise SystemExit('pass a single checkpoint .pkl for this probe')

  label, path = entries[0]
  networks, _ = roll._build_networks(args.env, seed=args.seed)
  env_kwargs = {}
  if args.env in ('sawyer_bin', 'sawyer_peg'):
    env_kwargs['randomize_init'] = bool(args.sawyer_randomize_init)
  if args.env == 'sawyer_bin' and args.sawyer_bin_safe_grasp_reset:
    env_kwargs['safe_grasp_reset'] = True
  if args.env == 'sawyer_bin' and args.bin_randomize_gripper_init:
    env_kwargs['randomize_gripper_init'] = True
  if args.env == 'sawyer_bin' and args.bin_randomize_tcp_z:
    env_kwargs['randomize_tcp_z'] = True
  if args.env == 'sawyer_bin' and args.sawyer_bin_bounded_step_mocap:
    env_kwargs['bounded_step_mocap'] = True
  if args.env == 'sawyer_bin' and args.sawyer_bin_selective_antiwindup:
    if args.sawyer_bin_bounded_step_mocap:
      raise SystemExit(
          'selective anti-windup and bounded-step mocap are separate modes')
    env_kwargs['selective_antiwindup_mocap'] = True
  if args.env == 'sawyer_bin' and args.sawyer_bin_trackerr_terminate_20cm:
    if args.sawyer_bin_selective_antiwindup:
      raise SystemExit(
          'trackerr termination and selective anti-windup are separate modes')
    env_kwargs['terminate_tracking_error'] = True
  if int(args.sawyer_max_episode_steps) > 0:
    env_kwargs['max_episode_steps'] = int(args.sawyer_max_episode_steps)

  gym_env, _, env_max_steps = env_utils.load(
      args.env, fixed_start_end=fixed_goal_dict[args.env], seed=args.seed,
      **env_kwargs)
  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  render = roll._get_render_fn(
      gym_env, args.width, args.height, args.camera, rotate_deg=args.rotate)

  ckpt = ppo_learner.load_checkpoint(path)
  frames_raw, errors, obj_speeds, actions, stats = _rollout_with_error(
      ckpt['policy_params'], gym_env, networks, render,
      max_steps=max_steps, stochastic=args.stochastic, seed=args.seed)

  # Always show the entire trajectory instead of clipping large errors at
  # the requested minimum axis limit. Round up to a readable 5cm boundary.
  max_error_cm = max(errors) * 100.0
  auto_ymax_cm = max(5.0, np.ceil(max_error_cm / 5.0) * 5.0)
  ymax = max(float(args.ymax_cm), auto_ymax_cm) / 100.0
  frames = [
      _overlay_strips(fr, errors, obj_speeds, actions, t=i, ymax=ymax)
      for i, fr in enumerate(frames_raw)
  ]

  out = args.output
  if out.endswith(os.sep) or os.path.isdir(out):
    out = os.path.join(out, f'{args.env}_{label}_trackerr.mp4')
  roll._write_video(frames, out, args.fps)

  err = np.asarray(errors, dtype=np.float64)
  spd = np.asarray(obj_speeds, dtype=np.float64)
  act = np.asarray(actions, dtype=np.float64)
  # Heuristic: "slam-ish" frames = object moved > 5cm in one control step.
  slam = spd * float(stats['dt']) > 0.05
  calm = ~slam
  valid = np.isfinite(act).all(axis=1)
  a_inf = np.full(len(act), np.nan, dtype=np.float64)
  a_xyz = np.full(len(act), np.nan, dtype=np.float64)
  if np.any(valid):
    a_inf[valid] = np.max(np.abs(act[valid]), axis=1)
    a_xyz[valid] = np.linalg.norm(act[valid, :3], axis=1)
  n_valid = int(np.sum(valid))
  summary = {
      'checkpoint': path,
      'seed': int(args.seed),
      'stochastic': bool(args.stochastic),
      'selective_antiwindup': bool(args.sawyer_bin_selective_antiwindup),
      'success': bool(stats['success']),
      'length': int(stats['length']),
      'error_cm_mean': float(np.mean(err) * 100),
      'error_cm_median': float(np.median(err) * 100),
      'error_cm_p90': float(np.quantile(err, 0.90) * 100),
      'error_cm_p95': float(np.quantile(err, 0.95) * 100),
      'error_cm_max': float(np.max(err) * 100),
      'error_cm_mean_calm': float(np.mean(err[calm]) * 100) if np.any(calm) else None,
      'error_cm_max_calm': float(np.max(err[calm]) * 100) if np.any(calm) else None,
      'error_cm_mean_slam': float(np.mean(err[slam]) * 100) if np.any(slam) else None,
      'error_cm_max_slam': float(np.max(err[slam]) * 100) if np.any(slam) else None,
      'n_slam_frames': int(np.sum(slam)),
      'obj_speed_cm_s_max': float(np.max(spd) * 100),
      'action_absmax': float(np.nanmax(a_inf)) if n_valid else None,
      'action_xyz_l2_max': float(np.nanmax(a_xyz)) if n_valid else None,
      'frac_action_sat_0p95': (
          float(np.mean(a_inf[valid] >= 0.95)) if n_valid else None),
      'frac_action_tiny_0p10': (
          float(np.mean(a_inf[valid] < 0.10)) if n_valid else None),
      'output': out,
  }
  print(json.dumps(summary, indent=2))

  csv_path = args.csv_out or (os.path.splitext(out)[0] + '_trackerr.csv')
  os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)
  with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=[
        't', 'error_m', 'error_cm', 'obj_speed_m_s', 'slam_jump',
        'a_dx', 'a_dy', 'a_dz', 'a_grip', 'a_inf', 'a_xyz_l2'])
    w.writeheader()
    for t, (e, v, a) in enumerate(zip(errors, obj_speeds, actions)):
      a = np.asarray(a, dtype=np.float64)
      finite = bool(np.all(np.isfinite(a)))
      w.writerow({
          't': t,
          'error_m': e,
          'error_cm': e * 100,
          'obj_speed_m_s': v,
          'slam_jump': int(v * float(stats['dt']) > 0.05),
          'a_dx': float(a[0]) if finite else '',
          'a_dy': float(a[1]) if finite else '',
          'a_dz': float(a[2]) if finite else '',
          'a_grip': float(a[3]) if finite else '',
          'a_inf': float(np.max(np.abs(a))) if finite else '',
          'a_xyz_l2': float(np.linalg.norm(a[:3])) if finite else '',
      })
  print(f'CSV {csv_path}')
  print(f'VIDEO {out}')


if __name__ == '__main__':
  main()
