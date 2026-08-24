#!/usr/bin/env python3
"""Sample p(s_goal | s_f) from sidecar backward-NF checkpoints.

``s_f`` is the run's fixed task goal (raw, unnormalized).  Draws N samples
per checkpoint and writes an ``.npz`` plus a MuJoCo montage of cube layouts.

  python scripts/sample_nf_backward.py \\
      --run_dir=logs/.../ppo_builderbench_creative_5_task2_0 \\
      --n_samples=10 --n_ckpts=5

Render an existing dump without resampling:

  python scripts/sample_nf_backward.py --from_npz=.../samples_sf_taskgoal.npz
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
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

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from contrastive.nf_density_backward import (
    make_nf_backward_networks,
    nf_backward_sample,
)
from envs.builderbench_utils import (
    apply_fixed_start_x,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

CUBE_COLORS = ['#d1495b', '#2d6a4f', '#1d4e89', '#b08900', '#6a4c93']
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')


def _load_run_config(run_dir: str) -> dict:
  path = os.path.join(run_dir, 'run_config.json')
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def _resolved(payload: dict) -> dict:
  resolved = payload.get('resolved_config') or {}
  flags = payload.get('flags') or {}
  return {**flags, **resolved}


def _task_goal(payload: dict) -> np.ndarray:
  raw = payload.get('fixed_start_end')
  if raw is None:
    raw = _resolved(payload).get('fixed_start_end')
  if raw is None:
    raise ValueError('run_config.json has no fixed_start_end')
  return np.asarray(raw, dtype=np.float32).reshape(-1)


def _list_nf_bwd_ckpts(run_dir: str) -> list[tuple[int, str]]:
  ckpt_dir = os.path.join(run_dir, 'checkpoints', 'nf_bwd')
  out = []
  for name in os.listdir(ckpt_dir):
    m = _CKPT_RE.match(name)
    if m is None:
      continue
    path = os.path.join(ckpt_dir, name)
    out.append((int(m.group(1)), path))
  out.sort()
  if not out:
    raise FileNotFoundError(f'no ckpt_iter_*.pkl in {ckpt_dir}')
  return out


def _pick_even(items: list, n: int) -> list:
  if n <= 0:
    raise ValueError('--n_ckpts must be > 0')
  if len(items) <= n:
    return list(items)
  idx = np.round(np.linspace(0, len(items) - 1, n)).astype(int)
  seen = set()
  picked = []
  for i in idx:
    i = int(i)
    if i in seen:
      continue
    seen.add(i)
    picked.append(items[i])
  return picked


def _build_nets(payload: dict, goal_dim: int):
  cfg = _resolved(payload)
  hidden = cfg.get('hidden_layer_sizes', (256,) * 6)
  if isinstance(hidden, str):
    hidden = tuple(int(x) for x in hidden.split(',') if x.strip())
  else:
    hidden = tuple(int(x) for x in hidden)
  return make_nf_backward_networks(
      goal_dim=int(goal_dim),
      hidden_layer_sizes=hidden,
      rep_size=int(cfg.get('nf_rep_size', 64)),
      num_blocks=int(cfg.get('nf_num_blocks', 8)),
      channels=int(cfg.get('nf_coupling_width', 256)),
      sa_hidden=int(cfg.get('nf_sa_hidden', 1024)),
      sa_num_layers=int(cfg.get('nf_sa_num_layers', 4)),
      scale_tanh=bool(cfg.get('nf_scale_tanh', False)),
      scale_tanh_c=float(cfg.get('nf_scale_tanh_c', 2.0)),
  )


def _load_bwd_ckpt(path: str) -> dict:
  with open(path, 'rb') as fh:
    return pickle.load(fh)


def _plot(samples: np.ndarray, task_goal: np.ndarray, iters: np.ndarray,
          out_png: str):
  """samples: (n_ckpt, n_samples, n_cubes, 3)."""
  n_ckpt, n_samples, n_cubes, _ = samples.shape
  g = task_goal.reshape(n_cubes, 3)
  fig, axes = plt.subplots(2, n_ckpt, figsize=(3.4 * n_ckpt, 6.6),
                           squeeze=False)
  for c, it in enumerate(iters):
    pts = samples[c]
    l2 = np.linalg.norm(pts.reshape(n_samples, -1) - task_goal[None], axis=-1)
    for row, (xi, yi, xlab, ylab, title_extra) in enumerate((
        (0, 1, 'x', 'y', 'top-down'),
        (0, 2, 'x', 'z', 'side'),
    )):
      ax = axes[row][c]
      for cube in range(n_cubes):
        color = CUBE_COLORS[cube % len(CUBE_COLORS)]
        ax.scatter(pts[:, cube, xi], pts[:, cube, yi],
                   s=22, c=color, alpha=0.55, linewidths=0,
                   label=f'c{cube}' if (c == 0 and row == 0) else None,
                   zorder=2)
        ax.scatter([g[cube, xi]], [g[cube, yi]],
                   marker='*', s=110, c=color, edgecolors='k',
                   linewidths=0.4, zorder=3)
      ax.set_aspect('equal', adjustable='box')
      ax.grid(True, alpha=0.28)
      ax.set_xlabel(xlab)
      if c == 0:
        ax.set_ylabel(ylab)
      if row == 0:
        ax.set_title(f'iter {int(it)}  ({title_extra})\n'
                     f'mean ‖s−g‖={l2.mean():.3f}')
      else:
        ax.set_title(title_extra)
  handles, labels = axes[0][0].get_legend_handles_labels()
  if handles:
    fig.legend(handles, labels, loc='upper center', ncol=n_cubes,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
  fig.suptitle(
      r'backward NF samples  $s_{\mathrm{goal}}\sim p(s_{\mathrm{goal}}\mid s_f)$'
      '\n'
      r'$s_f=$ task goal (stars); 10 samples / checkpoint',
      y=1.08, fontsize=11)
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
  fig.savefig(out_png, dpi=160, bbox_inches='tight')
  plt.close(fig)


def _font(size: int = 16):
  try:
    return ImageFont.truetype(
        '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', size)
  except Exception:
    return ImageFont.load_default()


def _make_render_env(env_name: str, fixed_start_x: float | None = 0.1):
  from builderbench.creative_cube import CreativeCube, default_config
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.permute_start_boxes = False
  cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  base = CreativeCube(config=cfg)
  apply_fixed_start_x(base, fixed_start_x)
  return base, num_cubes


def _free_camera(model, lookat, zoom: float):
  """Same azimuth/elevation as the default scene cam, closer and recentered."""
  import mujoco
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = np.asarray(lookat, dtype=np.float64).reshape(3)
  extent = float(model.stat.extent) if float(model.stat.extent) > 0 else 0.8
  cam.distance = (1.5 * extent) / max(float(zoom), 1e-3)
  cam.azimuth = float(model.vis.global_.azimuth)
  cam.elevation = float(model.vis.global_.elevation)
  return cam


def _render_cube_positions(env, cube_pos_vec, mocap_pos_vec=None,
                           height: int = 360, width: int = 480,
                           zoom: float = 2.5, lookat=None):
  """Set cube XYZs on init qpos (identity quats) and render. Mocap = task goal.

  ``zoom`` is relative to the default free camera (distance ← 1.5·extent / zoom)
  and ``lookat`` defaults to the mean cube xyz so the cluster fills the frame.
  """
  import mujoco
  num_task = int(env._num_task_cubes)
  qpos = np.array(env._init_q, copy=True)
  qpos[np.asarray(env._objs_pos_qpos_idxs)] = np.asarray(
      cube_pos_vec, dtype=np.float32).reshape(-1)
  qvel = np.zeros_like(np.array(env._init_v))
  if mocap_pos_vec is None:
    mocap_pos = np.tile(np.array([10.0, 10.0, 10.0]), num_task)
  else:
    mocap_pos = np.asarray(mocap_pos_vec, dtype=np.float32).reshape(-1)
  identity_quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), num_task)
  cubes = np.asarray(cube_pos_vec, dtype=np.float32).reshape(-1, 3)
  if lookat is None:
    lookat = cubes.mean(axis=0)
  d = mujoco.MjData(env._mj_model)
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  d.mocap_pos[env._task_mocap_targets] = mocap_pos.reshape(num_task, 3)
  d.mocap_quat[env._task_mocap_targets] = identity_quats.reshape(num_task, 4)
  mujoco.mj_forward(env._mj_model, d)
  cam = _free_camera(env._mj_model, lookat, zoom)
  renderer = mujoco.Renderer(env._mj_model, height=height, width=width)
  renderer.update_scene(d, camera=cam)
  img = np.asarray(renderer.render(), dtype=np.uint8)
  renderer.close()
  return img


def _caption(img: np.ndarray, text: str, bar_h: int = 28) -> np.ndarray:
  h, w = img.shape[:2]
  canvas = Image.new('RGB', (w, h + bar_h), color=(18, 18, 22))
  canvas.paste(Image.fromarray(img), (0, bar_h))
  draw = ImageDraw.Draw(canvas)
  draw.text((8, 6), text, fill=(235, 235, 240), font=_font(14))
  return np.asarray(canvas)


def _write_one_ckpt_plot(cells: list[np.ndarray], title: str, out_png: str):
  """One row: task goal + samples."""
  cell_h, cell_w = cells[0].shape[:2]
  title_h = 40
  canvas = Image.new(
      'RGB', (len(cells) * cell_w, title_h + cell_h), color=(16, 16, 20))
  draw = ImageDraw.Draw(canvas)
  draw.text((12, 10), title, fill=(230, 230, 235), font=_font(18))
  for c, img in enumerate(cells):
    canvas.paste(Image.fromarray(img), (c * cell_w, title_h))
  os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
  canvas.save(out_png)
  print(f'[render] wrote {out_png}', flush=True)


def _render_montage(samples: np.ndarray, task_goal: np.ndarray,
                    iters: np.ndarray, env_name: str, out_png: str,
                    frames_dir: str | None = None, zoom: float = 2.5):
  """One plot per checkpoint: task goal + samples."""
  n_ckpt, n_samples, n_cubes, _ = samples.shape
  env, env_cubes = _make_render_env(env_name)
  if env_cubes != n_cubes and int(env._num_task_cubes) != n_cubes:
    raise ValueError(
        f'env has {env_cubes} cubes / {env._num_task_cubes} task cubes, '
        f'samples have {n_cubes}')
  lookat = np.asarray(task_goal, dtype=np.float32).reshape(-1, 3).mean(axis=0)
  print(f'[render] CreativeCube {env_name}  {n_ckpt} plots × {n_samples} samples  '
        f'zoom={zoom:g}  lookat={lookat}',
        flush=True)

  goal_img = _render_cube_positions(
      env, task_goal, mocap_pos_vec=task_goal, height=480, width=640,
      zoom=zoom, lookat=lookat)
  stem, ext = os.path.splitext(out_png)
  if not ext:
    ext = '.png'
  for c, it in enumerate(iters):
    pts = samples[c]
    l2 = np.linalg.norm(pts.reshape(n_samples, -1) - task_goal[None], axis=-1)
    cells = [_caption(goal_img, 'task goal')]
    if frames_dir:
      it_dir = os.path.join(frames_dir, f'iter_{int(it):07d}')
      os.makedirs(it_dir, exist_ok=True)
      Image.fromarray(goal_img).save(os.path.join(it_dir, 'task_goal.png'))
    for s in range(n_samples):
      img = _render_cube_positions(
          env, pts[s], mocap_pos_vec=task_goal, height=480, width=640,
          zoom=zoom, lookat=lookat)
      cells.append(_caption(img, f's{s}  ‖s−g‖={l2[s]:.3f}'))
      if frames_dir:
        Image.fromarray(img).save(os.path.join(it_dir, f's{s:02d}.png'))
    ckpt_png = f'{stem}_iter_{int(it):07d}{ext}'
    _write_one_ckpt_plot(
        cells,
        f'iter {int(it)}   mean ‖s−g‖={l2.mean():.3f}   '
        'cubes = sample, ghost = task goal',
        ckpt_png)
  # Keep the original path as a pointer to the last plot name pattern.
  print(f'[render] wrote {n_ckpt} plots under {stem}_iter_*{ext}', flush=True)


def _env_name_from_payload(payload: dict) -> str:
  return str(_resolved(payload).get('env') or payload.get('env')
             or 'builderbench_creative_5_task2')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', default='')
  p.add_argument('--from_npz', default='',
                 help='skip sampling; render this dump')
  p.add_argument('--n_samples', type=int, default=10)
  p.add_argument('--n_render', type=int, default=5,
                 help='how many samples to render per checkpoint')
  p.add_argument('--n_ckpts', type=int, default=5)
  p.add_argument('--iters', default='',
                 help='comma-separated checkpoint iters; default = even pick')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out_dir', default='')
  p.add_argument('--render', action='store_true', default=True)
  p.add_argument('--no_render', action='store_false', dest='render')
  p.add_argument('--scatter', action='store_true', default=False)
  p.add_argument('--cam_zoom', type=float, default=2.5,
                 help='free-camera zoom vs default scene (2.5 = 2.5× closer)')
  args = p.parse_args()

  if args.from_npz:
    npz_path = os.path.abspath(args.from_npz)
    data = np.load(npz_path)
    samples = np.asarray(data['samples'])
    task_goal = np.asarray(data['task_goal'], dtype=np.float32).reshape(-1)
    iters = np.asarray(data['iteration'])
    out_dir = args.out_dir or os.path.dirname(npz_path)
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else os.path.dirname(out_dir)
    payload = _load_run_config(run_dir)
    env_name = _env_name_from_payload(payload)
    print(f'[sample] loaded {npz_path}  shape={samples.shape}', flush=True)
  else:
    if not args.run_dir:
      raise SystemExit('--run_dir is required unless --from_npz is set')
    run_dir = os.path.abspath(args.run_dir)
    payload = _load_run_config(run_dir)
    env_name = _env_name_from_payload(payload)
    task_goal = _task_goal(payload)
    goal_dim = int(task_goal.size)
    if goal_dim % 3 != 0:
      raise ValueError(f'task goal dim {goal_dim} is not 3×cubes')
    n_cubes = goal_dim // 3

    available = _list_nf_bwd_ckpts(run_dir)
    if args.iters.strip():
      want = {int(x) for x in args.iters.split(',') if x.strip()}
      by_iter = dict(available)
      missing = sorted(want - set(by_iter))
      if missing:
        raise FileNotFoundError(f'nf_bwd ckpts missing: {missing}')
      chosen = [(it, by_iter[it]) for it in sorted(want)]
    else:
      chosen = _pick_even(available, int(args.n_ckpts))

    nets = _build_nets(payload, goal_dim)
    n_samples = int(args.n_samples)
    key = jax.random.PRNGKey(int(args.seed))

    samples = np.zeros((len(chosen), n_samples, n_cubes, 3), dtype=np.float32)
    means = np.zeros((len(chosen), goal_dim), dtype=np.float32)
    stds = np.zeros((len(chosen), goal_dim), dtype=np.float32)
    iters = np.zeros(len(chosen), dtype=np.int32)
    steps = np.zeros(len(chosen), dtype=np.int64)

    sf = jnp.broadcast_to(jnp.asarray(task_goal)[None, :], (n_samples, goal_dim))

    for i, (it, path) in enumerate(chosen):
      ckpt = _load_bwd_ckpt(path)
      params = ckpt['nf_backward_params']
      s_mean = np.asarray(ckpt['nf_bwd_s_mean'], dtype=np.float32).reshape(-1)
      s_std = np.asarray(ckpt['nf_bwd_s_std'], dtype=np.float32).reshape(-1)
      if s_mean.size != goal_dim or s_std.size != goal_dim:
        raise ValueError(
            f'ckpt {it}: normalizer dim {s_mean.size}/{s_std.size} '
            f'!= goal_dim {goal_dim}')
      key, sub = jax.random.split(key)
      s_goal = nf_backward_sample(
          nets, params, sf, sub,
          jnp.asarray(s_mean), jnp.asarray(s_std))
      s_goal = np.asarray(s_goal)
      if not np.isfinite(s_goal).all():
        raise RuntimeError(f'non-finite samples at iter {it}')
      samples[i] = s_goal.reshape(n_samples, n_cubes, 3)
      means[i] = s_mean
      stds[i] = s_std
      iters[i] = int(ckpt.get('iteration', it))
      steps[i] = int(ckpt.get('global_step', -1))
      l2 = np.linalg.norm(s_goal - task_goal[None], axis=-1)
      print(f'[sample] iter={it}  global_step={steps[i]}  '
            f'mean‖s-g‖={l2.mean():.4f}  min={l2.min():.4f}  max={l2.max():.4f}',
            flush=True)

    out_dir = args.out_dir or os.path.join(run_dir, 'nf_bwd_samples')
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, 'samples_sf_taskgoal.npz')
    np.savez_compressed(
        npz_path,
        samples=samples,
        task_goal=task_goal,
        s_mean=means,
        s_std=stds,
        iteration=iters,
        global_step=steps,
        n_samples=np.int32(n_samples),
    )
    print(f'[sample] wrote {npz_path}')

  os.makedirs(out_dir, exist_ok=True)
  if args.scatter:
    scatter_path = os.path.join(out_dir, 'samples_sf_taskgoal_scatter.png')
    _plot(samples, task_goal, iters, scatter_path)
    print(f'[sample] wrote {scatter_path}')
  if args.render:
    n_show = int(args.n_render)
    if n_show <= 0:
      raise ValueError('--n_render must be > 0')
    if n_show > samples.shape[1]:
      raise ValueError(
          f'--n_render={n_show} but dump only has {samples.shape[1]} samples')
    render_path = os.path.join(out_dir, 'samples_sf_taskgoal_render.png')
    frames_dir = os.path.join(out_dir, 'renders')
    _render_montage(samples[:, :n_show], task_goal, iters, env_name,
                    render_path, frames_dir=frames_dir,
                    zoom=float(args.cam_zoom))


if __name__ == '__main__':
  main()
