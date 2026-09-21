"""Composite CRL/NF goal-preimage top-K frames into weighted blends + GIFs.

For each checkpoint:
  1. Re-sample the same stochastic trajectories (fixed seed) as the gallery,
     score under the preimage weight w ∝ score(s,a,g*), take top-K.
  2. Save clean (unannotated) rank frames.
  3. Form a probability-weighted blend:
         I_blend = Σ_i  (w_i / Σ w) · I_i
     so brighter/more opaque contributions come from higher-mass modes.
  4. Also write a 'ghost' panel (same blend, plus a soft max-hold of the
     frames) and GIFs that animate the discrete support / stage progression.

Example::

  python scripts/builderbench_goal_preimage_composite.py \\
      --repr_mode=crl \\
      --run_dir=logs/ppo_builderbench_creative3_task1_e1024_pd_tau0p5/ppo_builderbench_creative_3_task1_0 \\
      --iters=0,150,300,1950 \\
      --output=figs/builderbench/creative3_task1_crl_goal_preimage_composite/
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from contrastive import ppo_learner
from envs.builderbench_utils import sgcrl_env_name_to_bb_env_id

# Import sibling viz helpers (scripts/ is not a package).
import importlib.util as _ilu
_viz_path = os.path.join(_REPO, 'scripts', 'builderbench_nf_goal_preimage_viz.py')
_spec = _ilu.spec_from_file_location('bb_preimage_viz', _viz_path)
viz = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules['bb_preimage_viz'] = viz  # required for @dataclass during exec
_spec.loader.exec_module(viz)


def _font(size: int = 18):
  try:
    return ImageFont.truetype(
        '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', size)
  except Exception:
    return ImageFont.load_default()


def _label_bar(img: np.ndarray, text: str, bar_h: int = 36) -> np.ndarray:
  h, w = img.shape[:2]
  canvas = Image.new('RGB', (w, h + bar_h), color=(18, 18, 22))
  canvas.paste(Image.fromarray(img), (0, bar_h))
  draw = ImageDraw.Draw(canvas)
  draw.text((10, 8), text, fill=(235, 235, 240), font=_font(16))
  return np.asarray(canvas)


def _iteration_title_card(iteration: int, width: int, height: int) -> np.ndarray:
  """Full-frame cut card announcing a new checkpoint."""
  canvas = Image.new('RGB', (width, height), color=(12, 12, 16))
  draw = ImageDraw.Draw(canvas)
  # Accent bar
  draw.rectangle([0, height // 2 - 70, width, height // 2 + 70],
                 fill=(28, 32, 48))
  title = f'iteration = {iteration}'
  font_big = _font(54)
  # Approximate center via textbbox
  bbox = draw.textbbox((0, 0), title, font=font_big)
  tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
  x = (width - tw) // 2
  y = (height - th) // 2 - 8
  draw.text((x + 2, y + 2), title, fill=(0, 0, 0), font=font_big)
  draw.text((x, y), title, fill=(255, 220, 90), font=font_big)
  sub = 'goal-preimage  ·  top-k support'
  font_sm = _font(20)
  sb = draw.textbbox((0, 0), sub, font=font_sm)
  sw = sb[2] - sb[0]
  draw.text(((width - sw) // 2, y + th + 18), sub,
            fill=(180, 185, 200), font=font_sm)
  return np.asarray(canvas)


def _rank_salient_frame(
    img: np.ndarray,
    *,
    iteration: int,
    rank: int,
    weight: float | None = None,
    score: float | None = None,
    score_name: str = 'φ·ψ',
) -> np.ndarray:
  """Overlay a large RANK badge + iteration strip on a clean render."""
  im = Image.fromarray(img).convert('RGB')
  draw = ImageDraw.Draw(im, 'RGBA')
  W, H = im.size

  # Top strip: iteration (always visible)
  strip_h = 44
  draw.rectangle([0, 0, W, strip_h], fill=(18, 18, 24, 230))
  draw.text((14, 8), f'iteration = {iteration}',
            fill=(255, 220, 90), font=_font(26))

  # Large RANK badge, top-left under strip
  badge = f'RANK {rank}'
  font_rank = _font(48)
  bb = draw.textbbox((0, 0), badge, font=font_rank)
  bw, bh = bb[2] - bb[0] + 28, bb[3] - bb[1] + 18
  bx, by = 12, strip_h + 12
  # Rank-colored accents (1=gold … 5=cool)
  palette = [
      (255, 196, 40),
      (220, 220, 230),
      (205, 140, 70),
      (120, 180, 255),
      (160, 160, 180),
  ]
  color = palette[min(rank, 5) - 1]
  draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=10,
                         fill=(0, 0, 0, 200), outline=color + (255,), width=3)
  draw.text((bx + 14, by + 6), badge, fill=color, font=font_rank)

  # Optional weight / score chip
  bits = []
  if weight is not None:
    bits.append(f'w = {weight:.3f}')
  if score is not None:
    bits.append(f'{score_name} = {score:.2f}')
  if bits:
    wtxt = '   '.join(bits)
    font_w = _font(22)
    wb = draw.textbbox((0, 0), wtxt, font=font_w)
    ww, wh = wb[2] - wb[0] + 20, wb[3] - wb[1] + 12
    wx, wy = 12, H - wh - 12
    draw.rounded_rectangle([wx, wy, wx + ww, wy + wh], radius=8,
                           fill=(0, 0, 0, 190))
    draw.text((wx + 10, wy + 4), wtxt, fill=(230, 230, 235), font=font_w)

  return np.asarray(im.convert('RGB'))


def _write_gif(frames: List[np.ndarray], path: str, duration_ms: int = 700,
               durations: List[int] | None = None):
  imgs = [Image.fromarray(f) for f in frames]
  # Match sizes
  w = max(im.size[0] for im in imgs)
  h = max(im.size[1] for im in imgs)
  normed = []
  for im in imgs:
    canvas = Image.new('RGB', (w, h), color=(16, 16, 20))
    canvas.paste(im, ((w - im.size[0]) // 2, (h - im.size[1]) // 2))
    normed.append(canvas)
  os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
  if durations is None:
    durations = [duration_ms] * len(normed)
  assert len(durations) == len(normed)
  normed[0].save(
      path, save_all=True, append_images=normed[1:],
      duration=durations, loop=0, optimize=False)
  print(f'[composite] wrote {path}  ({len(normed)} frames)')


def _build_topk_support_gif(
    stages: List[Dict],
    out_path: str,
    *,
    title_ms: int = 1400,
    rank_ms: int = 700,
    score_name: str = 'φ·ψ',
):
  """GIF with iteration title-card cuts + salient RANK badges."""
  frames: List[np.ndarray] = []
  durs: List[int] = []
  # Infer canvas size from first frame
  ref = stages[0]['frames'][0]
  H, W = ref.shape[:2]

  for stage in stages:
    it = int(stage['iteration'])
    frames.append(_iteration_title_card(it, W, H))
    durs.append(title_ms)

    w = np.asarray(stage['weights'], dtype=np.float64)
    w_n = w / np.sum(w)
    scores = stage.get('raw_scores')
    show_w = bool(stage.get('show_weights', True))
    for r, (fr, wi) in enumerate(zip(stage['frames'], w_n), start=1):
      sc = None if scores is None else float(scores[r - 1])
      frames.append(_rank_salient_frame(
          fr, iteration=it, rank=r,
          weight=(float(wi) if show_w else None),
          score=sc, score_name=score_name))
      durs.append(rank_ms)

  _write_gif(frames, out_path, durations=durs)


def _weighted_blend(frames: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
  w = np.asarray(weights, dtype=np.float64)
  w = w / np.sum(w)
  acc = np.zeros(frames[0].shape, dtype=np.float64)
  for f, wi in zip(frames, w):
    acc += wi * f.astype(np.float64)
  return np.clip(np.round(acc), 0, 255).astype(np.uint8)


def _ghost_max(frames: List[np.ndarray], weights: np.ndarray,
               floor: float = 0.15) -> np.ndarray:
  """Soft multi-exposure: each frame contributes with alpha ∝ weight.

  Implemented as a normalized weighted average in linear space (same as
  blend) composited 50/50 with a max-projection so low-mass modes still
  leave a faint ghost.
  """
  blend = _weighted_blend(frames, weights).astype(np.float64)
  stack = np.stack([f.astype(np.float64) for f in frames], axis=0)
  mx = np.max(stack, axis=0)
  # Emphasize high-mass modes in the ghost: weight the max by relative w.
  w = np.asarray(weights, dtype=np.float64)
  w = w / np.sum(w)
  ghost = np.zeros_like(blend)
  for f, wi in zip(frames, w):
    a = floor + (1.0 - floor) * wi
    ghost = np.maximum(ghost, a * f.astype(np.float64))
  out = 0.55 * blend + 0.45 * ghost
  return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _stage_strip(frames: List[np.ndarray], weights: np.ndarray,
                 title: str, cell_w: int = 200, cell_h: int = 150) -> np.ndarray:
  """Horizontal strip: top-K thumbnails with weight bars + blend on the right."""
  k = len(frames)
  gap = 6
  bar_h = 14
  label_h = 28
  blend = _weighted_blend(frames, weights)
  thumbs = [Image.fromarray(f).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            for f in frames]
  blend_big = Image.fromarray(blend).resize(
      (cell_w * 2 + gap, cell_h * 2 + gap + bar_h), Image.Resampling.BILINEAR)

  # 1 row of K thumbs + weight bars, then blend to the right spanning 2 rows.
  # Simpler layout: [thumb1..thumbK] on top row; blend centered below.
  row1_w = k * cell_w + (k - 1) * gap
  total_w = max(row1_w, blend_big.size[0])
  total_h = label_h + cell_h + bar_h + gap + blend_big.size[1] + 8
  canvas = Image.new('RGB', (total_w, total_h), color=(16, 16, 20))
  draw = ImageDraw.Draw(canvas)
  draw.text((8, 6), title, fill=(235, 235, 240), font=_font(16))

  w = np.asarray(weights, dtype=np.float64)
  w_n = w / np.sum(w)
  y0 = label_h
  for i, (th, wi) in enumerate(zip(thumbs, w_n)):
    x = i * (cell_w + gap)
    canvas.paste(th, (x, y0))
    # weight bar under thumb
    bw = int(round(wi * cell_w))
    draw.rectangle([x, y0 + cell_h, x + cell_w, y0 + cell_h + bar_h],
                   fill=(40, 40, 48))
    draw.rectangle([x, y0 + cell_h, x + bw, y0 + cell_h + bar_h],
                   fill=(90, 180, 255))
    draw.text((x + 4, y0 + cell_h + 1), f'r{i+1} w={wi:.2f}',
              fill=(240, 240, 245), font=_font(11))

  bx = (total_w - blend_big.size[0]) // 2
  by = y0 + cell_h + bar_h + gap
  canvas.paste(blend_big, (bx, by))
  draw.text((bx + 8, by + 6), 'weighted blend  Σ w_i I_i',
            fill=(255, 255, 255), font=_font(14))
  return np.asarray(canvas)


def _montage_row(images: List[np.ndarray], labels: List[str],
                 cell_w: int = 480, cell_h: int = 360) -> np.ndarray:
  n = len(images)
  label_h = 32
  gap = 8
  W = n * cell_w + (n - 1) * gap
  H = label_h + cell_h
  canvas = Image.new('RGB', (W, H), color=(16, 16, 20))
  draw = ImageDraw.Draw(canvas)
  for i, (im, lab) in enumerate(zip(images, labels)):
    x = i * (cell_w + gap)
    draw.text((x + 8, 6), lab, fill=(235, 235, 240), font=_font(15))
    tile = Image.fromarray(im).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
    canvas.paste(tile, (x, label_h))
  return np.asarray(canvas)


def _montage_grid(images: List[np.ndarray], labels: List[str],
                 ncols: int = 4, cell_w: int = 560, cell_h: int = 420
                 ) -> np.ndarray:
  n = len(images)
  ncols = max(1, min(int(ncols), n))
  nrows = (n + ncols - 1) // ncols
  label_h = 32
  gap = 8
  W = ncols * cell_w + (ncols - 1) * gap
  H = nrows * (label_h + cell_h) + (nrows - 1) * gap
  canvas = Image.new('RGB', (W, H), color=(16, 16, 20))
  draw = ImageDraw.Draw(canvas)
  for i, (im, lab) in enumerate(zip(images, labels)):
    r, c = divmod(i, ncols)
    x = c * (cell_w + gap)
    y = r * (label_h + cell_h + gap)
    draw.text((x + 8, y + 6), lab, fill=(235, 235, 240), font=_font(15))
    tile = Image.fromarray(im).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
    canvas.paste(tile, (x, y + label_h))
  return np.asarray(canvas)


def _parked_mocap(template: np.ndarray) -> np.ndarray:
  """Move mocap markers far off-table so they do not appear in renders."""
  out = np.asarray(template, dtype=np.float64).copy()
  flat = out.reshape(-1, 3)
  flat[:] = (10.0, 10.0, 10.0)
  return out.reshape(out.shape)


def _render_goal_solid(
    base,
    video_env,
    hard_goal: np.ndarray,
    *,
    height: int,
    width: int,
    camera=None,
    slot_to_cube: np.ndarray | None = None,
) -> np.ndarray:
  """Render the hard-goal pyramid as solid cubes (no mocap markers).

  By default cube ``i`` (among masked cubes, in index order) goes to goal
  slot ``i``. Pass ``slot_to_cube`` (length = #goal slots) to put cube
  ``slot_to_cube[s]`` at goal slot ``s`` — useful when success is
  permutation-invariant and the policy uses a different color layout.
  """
  qpos = np.array(base._init_q, copy=True)
  qvel = np.zeros_like(np.asarray(base._init_v))
  goal = np.asarray(hard_goal, dtype=np.float32).reshape(-1, 3)
  mask = np.asarray(base._target_cube_masks_data).reshape(-1).astype(bool)
  num_cubes = int(len(mask))
  pos = np.asarray(qpos[np.asarray(base._objs_pos_qpos_idxs)], dtype=np.float32)
  pos = pos.reshape(num_cubes, 3)
  if int(goal.shape[0]) != int(mask.sum()):
    raise ValueError(
        f'hard_goal has {goal.shape[0]} cubes but mask selects {int(mask.sum())}')
  if slot_to_cube is None:
    pos[mask] = goal
  else:
    order = np.asarray(slot_to_cube, dtype=np.int32).reshape(-1)
    if order.shape[0] != goal.shape[0]:
      raise ValueError(
          f'slot_to_cube length {order.shape[0]} != goal slots {goal.shape[0]}')
    if np.unique(order).size != order.size:
      raise ValueError(f'slot_to_cube has duplicates: {order.tolist()}')
    for s, cube in enumerate(order.tolist()):
      if cube < 0 or cube >= num_cubes or not bool(mask[cube]):
        raise ValueError(
            f'slot_to_cube[{s}]={cube} is not a masked-in cube '
            f'(num_cubes={num_cubes}, mask={mask.tolist()})')
      pos[cube] = goal[s]
  qpos[np.asarray(base._objs_pos_qpos_idxs)] = pos.reshape(-1)
  qpos[np.asarray(base._objs_quat_qpos_idxs)] = np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), num_cubes)
  n_task = int(base._num_task_cubes)
  mocap_pos = np.tile(np.array([10.0, 10.0, 10.0], dtype=np.float32), n_task)
  mocap_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), n_task)
  return viz._render_state(
      video_env, qpos, qvel, mocap_pos, mocap_quat,
      height=height, width=width, camera=camera,
  )


def _collect_stage(
    *,
    stage_iter: int,
    run_dir: str,
    env_name: str,
    repr_mode: str,
    ctx,
    networks,
    nf_nets,
    video_env,
    mocap_targets,
    num_cubes: int,
    episode_length: int,
    hard_goal: np.ndarray,
    num_trajs: int,
    top_k: int,
    seed: int,
    height: int = 480,
    width: int = 640,
    camera=None,
    post_success_steps: int = -1,
    hide_mocap: bool = False,
) -> Dict:
  ckpt_path = os.path.join(
      run_dir, 'checkpoints', f'ckpt_iter_{stage_iter:07d}.pkl')
  ckpt = ppo_learner.load_checkpoint(ckpt_path)
  policy_params = ckpt['policy_params']
  if repr_mode == 'crl':
    q_params = (ckpt['q_params_ema']
                if ckpt.get('q_params_ema') is not None
                else ckpt['q_params'])
    goal_mean = goal_std = None
  else:
    q_params = ckpt['q_params']
    goal_mean, goal_std = viz._load_goal_stats(
        run_dir, stage_iter, ctx.goal_dim)
    goal_std = np.maximum(goal_std, ctx.nf_goal_std_min).astype(np.float32)

  policy = viz._make_policy_fn(
      networks, policy_params,
      filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes)

  # Deterministic key schedule matching gallery: PRNGKey(seed) then one
  # split per stage in order is NOT the same — use a stage-specific seed
  # so composites are reproducible regardless of stage subset.
  key = jax.random.PRNGKey(seed + 10007 * int(stage_iter))
  data = viz._collect_trajectories(
      policy, video_env, key,
      num_trajs=num_trajs,
      episode_length=episode_length,
      fixed_target_goal=ctx.fixed_target_goal,
      mocap_targets=mocap_targets,
      num_cubes=num_cubes,
      filter_policy_obs=ctx.filter_policy_obs,
      obs_dim=ctx.obs_dim,
      post_success_steps=post_success_steps,
  )
  if repr_mode == 'nf':
    raw, score, weights = viz._score_preimage_nf(
        nf_nets, q_params, data['state'], data['action'],
        hard_goal, goal_mean, goal_std)
  else:
    raw, score, weights = viz._score_preimage_crl(
        networks, q_params, data['state'], data['action'], hard_goal)

  top_idx = np.argsort(-weights)[:top_k]
  frames = []
  for idx in top_idx:
    mocap_pos = data['mocap_pos'][idx]
    mocap_quat = data['mocap_quat'][idx]
    if hide_mocap:
      mocap_pos = _parked_mocap(mocap_pos)
    frames.append(viz._render_state(
        video_env,
        data['qpos'][idx], data['qvel'][idx],
        mocap_pos, mocap_quat,
        height=height, width=width, camera=camera,
    ))
  top_w = weights[top_idx]
  selected_traj = data['traj_index'][top_idx]
  selected_step = data['traj_step'][top_idx]
  print(
      '[composite]   selected '
      + ', '.join(
          f'rank{rank}=traj{int(traj_i)}:t{int(step_i)}'
          for rank, (traj_i, step_i) in enumerate(
              zip(selected_traj, selected_step), start=1)),
      flush=True)
  return {
      'iteration': stage_iter,
      'frames': frames,
      'weights': top_w,
      'raw_scores': raw[top_idx],
      'all_weights': weights,
      'top_idx': top_idx,
      'selected_traj': selected_traj,
      'selected_step': selected_step,
      'first_success_steps': data['first_success_steps'],
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      '--run_dir',
      default=('logs/ppo_builderbench_creative3_task1_e1024_pd_tau0p5/'
               'ppo_builderbench_creative_3_task1_0'))
  ap.add_argument('--env', default='builderbench_creative_3_task1')
  ap.add_argument('--repr_mode', choices=['crl', 'nf'], default='crl')
  ap.add_argument('--iters', default='0,150,300,1950')
  ap.add_argument(
      '--output',
      default='figs/builderbench/creative3_task1_crl_goal_preimage_composite/')
  ap.add_argument('--num_trajs', type=int, default=10)
  ap.add_argument('--top_k', type=int, default=5)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--gif_ms', type=int, default=800)
  ap.add_argument('--width', type=int, default=960)
  ap.add_argument('--height', type=int, default=720)
  ap.add_argument('--cam_zoom', type=float, default=5.0,
                  help='Free-camera zoom vs default distance (larger = closer).')
  ap.add_argument('--cam_lookat', default='',
                  help='x,y,z look-at. Default: mean of the hard-goal cubes.')
  ap.add_argument('--cam_azimuth', default='',
                  help='Override azimuth; default is the scene camera.')
  ap.add_argument('--cam_elevation', default='',
                  help='Override elevation; default is the scene camera.')
  ap.add_argument('--montage_ncols', type=int, default=0,
                  help='Blend/ghost montage columns. 0 = auto (4, or 8 if many).')
  ap.add_argument('--skip_topk_gif', action='store_true',
                  help='Skip the long rank-cycling GIF (useful for dense iters).')
  ap.add_argument(
      '--post_success_steps', type=int, default=-1,
      help=('After first hard success, keep this many extra macro-steps '
            '(includes the successful state). -1 = keep full episode.'))
  ap.add_argument(
      '--hide_mocap', action='store_true',
      help='Park mocap goal markers off-table in rendered frames.')
  ap.add_argument(
      '--write_goal_panel', action='store_true',
      help='Also write goal_solid.png (hard-goal cubes, no mocap markers).')
  ap.add_argument(
      '--goal_cube_order', default='',
      help=('Comma-separated cube ids for goal slots 0..N-1 when writing the '
            'goal panel (permutation-invariant color layout). Empty = cube '
            'index order. Example for C5T2 policy colors: 1,2,3,0,4'))
  ap.add_argument(
      '--goal_panel_name', default='goal_solid.png',
      help='Filename for --write_goal_panel / --hide_mocap goal render.')
  ap.add_argument(
      '--goal_only', action='store_true',
      help='Write the goal panel then exit (skip trajectory composites).')
  args = ap.parse_args()

  iters = [int(x) for x in args.iters.split(',') if x.strip()]
  run_dir = os.path.abspath(args.run_dir)
  out_dir = os.path.abspath(args.output)
  os.makedirs(out_dir, exist_ok=True)

  print(f'[composite] repr_mode={args.repr_mode} iters={iters} '
        f'post_success_steps={args.post_success_steps} '
        f'hide_mocap={bool(args.hide_mocap)}')
  ctx, _ = viz._load_train_ctx(args.env, run_dir)
  hard_goal = np.asarray(ctx.fixed_target_goal, dtype=np.float32).reshape(-1)
  env_id = sgcrl_env_name_to_bb_env_id(args.env)

  print('[composite] building networks / env...', flush=True)
  networks, nf_nets, act_dim, _ = viz._build_networks(
      args.env, args.seed, ctx, args.repr_mode)
  ctx.act_dim = act_dim
  video_env, base, mocap_targets, episode_length, num_cubes = viz._make_bb_env(
      env_id, ctx)
  base._mj_model.vis.global_.offwidth = max(
      int(args.width), int(base._mj_model.vis.global_.offwidth))
  base._mj_model.vis.global_.offheight = max(
      int(args.height), int(base._mj_model.vis.global_.offheight))

  if args.cam_lookat.strip():
    lookat = np.asarray(
        [float(x) for x in args.cam_lookat.split(',')], dtype=np.float64)
  else:
    lookat = hard_goal.reshape(-1, 3).mean(axis=0)
  azimuth = None if not str(args.cam_azimuth).strip() else float(args.cam_azimuth)
  elevation = (None if not str(args.cam_elevation).strip()
               else float(args.cam_elevation))
  camera = viz._free_camera(
      base._mj_model, lookat, float(args.cam_zoom),
      azimuth=azimuth, elevation=elevation)
  print(f'[composite] ep_len={episode_length} zoom={args.cam_zoom} '
        f'lookat={np.asarray(lookat).round(3).tolist()}', flush=True)
  score_name = 'logp' if args.repr_mode == 'nf' else 'φ·ψ'

  slot_to_cube = None
  if str(args.goal_cube_order).strip():
    slot_to_cube = np.asarray(
        [int(x) for x in str(args.goal_cube_order).split(',') if x.strip()],
        dtype=np.int32)
    print(f'[composite] goal slot_to_cube={slot_to_cube.tolist()}', flush=True)

  if args.write_goal_panel or args.hide_mocap or args.goal_only:
    goal_img = _render_goal_solid(
        base, video_env, hard_goal,
        height=int(args.height), width=int(args.width), camera=camera,
        slot_to_cube=slot_to_cube)
    goal_name = str(args.goal_panel_name).strip() or 'goal_solid.png'
    goal_path = os.path.join(out_dir, goal_name)
    Image.fromarray(goal_img).save(goal_path)
    print(f'[composite] wrote {goal_path}', flush=True)
    if args.goal_only:
      print('[composite] goal_only: done', flush=True)
      return

  stages = []
  blend_frames = []
  ghost_frames = []
  strip_panels = []

  for it in iters:
    print(f'\n[composite] === iter {it} ===', flush=True)
    stage = _collect_stage(
        stage_iter=it,
        run_dir=run_dir,
        env_name=args.env,
        repr_mode=args.repr_mode,
        ctx=ctx,
        networks=networks,
        nf_nets=nf_nets,
        video_env=video_env,
        mocap_targets=mocap_targets,
        num_cubes=num_cubes,
        episode_length=episode_length,
        hard_goal=hard_goal,
        num_trajs=args.num_trajs,
        top_k=args.top_k,
        seed=args.seed,
        height=int(args.height),
        width=int(args.width),
        camera=camera,
        post_success_steps=int(args.post_success_steps),
        hide_mocap=bool(args.hide_mocap),
    )
    stages.append(stage)
    stage_dir = os.path.join(out_dir, f'iter_{it:07d}')
    os.makedirs(stage_dir, exist_ok=True)

    w = stage['weights']
    w_n = w / np.sum(w)
    for r, (fr, wi, rs) in enumerate(
        zip(stage['frames'], w_n, stage['raw_scores']), start=1):
      path = os.path.join(stage_dir, f'rank{r:02d}_clean.png')
      Image.fromarray(fr).save(path)
    np.savez(
        os.path.join(stage_dir, 'selected_samples.npz'),
        top_idx=stage['top_idx'],
        traj_index=stage['selected_traj'],
        traj_step=stage['selected_step'],
        weights=stage['weights'],
        normalized_top_weights=w_n,
        raw_scores=stage['raw_scores'],
        first_success_steps=stage['first_success_steps'],
        post_success_steps=np.asarray(int(args.post_success_steps)),
    )
    blend = _weighted_blend(stage['frames'], stage['weights'])
    ghost = _ghost_max(stage['frames'], stage['weights'])
    blend_l = _label_bar(
        blend,
        f'iter={it}  weighted blend  mass_in_top{args.top_k}='
        f'{float(np.sum(stage["weights"])):.3f}')
    ghost_l = _label_bar(ghost, f'iter={it}  ghost (blend + soft max)')
    Image.fromarray(blend).save(os.path.join(stage_dir, 'weighted_blend.png'))
    Image.fromarray(ghost).save(os.path.join(stage_dir, 'ghost_blend.png'))
    Image.fromarray(blend_l).save(os.path.join(stage_dir, 'weighted_blend_labeled.png'))
    blend_frames.append(blend_l)
    ghost_frames.append(ghost_l)

    strip = _stage_strip(
        stage['frames'], stage['weights'],
        title=(f'{args.repr_mode.upper()} preimage  iter={it}  '
               f'top-{args.top_k} mass={float(np.sum(w)):.3f}'))
    Image.fromarray(strip).save(os.path.join(stage_dir, 'strip.png'))
    strip_panels.append(strip)
    print(f'[composite]   top-{args.top_k} weights={w_n.round(3).tolist()} '
          f'mass={float(np.sum(w)):.4f}')

  # Grid montage of blends (wrap so 7 early ckpts stay readable).
  labels = [f'iter {it}' for it in iters]
  blends_only = [_weighted_blend(s['frames'], s['weights']) for s in stages]
  ghosts_only = [_ghost_max(s['frames'], s['weights']) for s in stages]
  ncols = int(args.montage_ncols)
  if ncols <= 0:
    ncols = 8 if len(iters) > 12 else (4 if len(iters) > 4 else len(iters))
  montage_blend = _montage_grid(blends_only, labels, ncols=ncols)
  montage_ghost = _montage_grid(
      ghosts_only, [f'{l} ghost' for l in labels], ncols=ncols)
  Image.fromarray(montage_blend).save(
      os.path.join(out_dir, 'blends_montage.png'))
  Image.fromarray(montage_ghost).save(
      os.path.join(out_dir, 'ghosts_montage.png'))
  print(f'[composite] wrote {out_dir}/blends_montage.png')

  # Stack strips vertically (skip when dense — it becomes a 20k-px image).
  if len(iters) <= 16:
    max_w = max(p.shape[1] for p in strip_panels)
    total_h = sum(p.shape[0] for p in strip_panels) + 8 * (len(strip_panels) - 1)
    big = Image.new('RGB', (max_w, total_h), color=(16, 16, 20))
    y = 0
    for p in strip_panels:
      big.paste(Image.fromarray(p), (0, y))
      y += p.shape[0] + 8
    big.save(os.path.join(out_dir, 'strips_vertical.png'))

  # GIFs
  _write_gif(blend_frames, os.path.join(out_dir, 'blends_over_training.gif'),
             duration_ms=args.gif_ms)
  _write_gif(ghost_frames, os.path.join(out_dir, 'ghosts_over_training.gif'),
             duration_ms=args.gif_ms)
  if not args.skip_topk_gif:
    _build_topk_support_gif(
        stages,
        os.path.join(out_dir, 'topk_support_over_training.gif'),
        title_ms=max(1200, int(args.gif_ms * 1.5)),
        rank_ms=max(550, args.gif_ms // 2),
        score_name=score_name,
    )
  _write_gif(blend_frames, os.path.join(out_dir, 'blends_slow.gif'),
             duration_ms=max(args.gif_ms, 1200))

  print(f'[composite] done → {out_dir}')


if __name__ == '__main__':
  main()
