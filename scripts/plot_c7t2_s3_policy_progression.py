"""c7t2 deterministic-policy progression + goal panel.

Seed 3 (exact deterministic states re-rendered without target mocaps):
  1200: last frame, indigo cube lowered in z so it sits below the title
  1600: frame 25 (before the stack slumps)
  3600: frame 30, with only rose/green xyz exchanged

Seed 2 (exact deterministic states re-rendered without target mocaps):
  1400 label: iter-800 frame 20

The first panel is the fixed c7t2 target (solid cubes at
``default_fixed_target_goal(7, 1)``), same default camera as the videos.

  python scripts/plot_c7t2_s3_policy_progression.py

Writes ``paper material/paper plots/bb_c7t2_s3_policy_progression.pdf``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(
    REPO, 'paper material', 'paper plots', 'final final plots')
OUT_STEM = os.path.join(OUT_DIR, 'bb_c7t2_s3_policy_progression')

# Tight shared crop on the 640x480 default camera. It still includes the
# scattered purple/orange cubes at 1200 while removing most empty background.
CROP = (270, 215, 415, 380)  # left, top, right, bottom

# cube 0 coral, 1 teal, 2 mustard, 3 indigo, 4 rose, 5 green, 6 orange
GOAL_XYZ = np.array(
    [
        [0.27, 0.000, 0.02],
        [0.27, 0.010, 0.06],
        [0.27, -0.010, 0.10],
        [0.27, 0.000, 0.14],
        [0.27, -0.045, 0.02],  # rose (light pink)
        [0.27, -0.045, 0.06],  # green
        [0.27, 0.060, 0.02],
    ],
    dtype=np.float32,
)
# title, seed, iteration, frame, static xyz, clean frame, status label
PANELS = (
    ('Goal', None, None, None, GOAL_XYZ, None, None),
    ('1200', 3, 1200, 45, None, 's3_iter_0001200_f045_clean_purplelow.png', None),
    ('1400', 2, 800, 20, None, 's2_iter_0000800_f020_clean.png',
     'Unstable'),
    ('1600', 3, 1600, 25, None, 's3_iter_0001600_f025_clean.png',
     'Falling'),
    ('3600', 3, 3600, 30, None, 's3_iter_0003600_f030_clean_swap45.png',
     None),
)


def _render_cubes(out_png: str, xyz: np.ndarray) -> None:
  os.environ.setdefault('MUJOCO_GL', 'egl')
  os.environ.setdefault('BUILDERBENCH_MJX_IMPL', 'jax')
  sys.path.insert(0, REPO)
  from builderbench.creative_cube import CreativeCube, default_config
  from envs.builderbench_utils import parse_bb_env_id, sgcrl_env_name_to_bb_env_id

  env_id = sgcrl_env_name_to_bb_env_id('builderbench_creative_7_task2')
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.permute_start_boxes = False
  cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  env = CreativeCube(config=cfg)

  qpos = np.array(env._init_q, copy=True)
  qpos[np.asarray(env._objs_pos_qpos_idxs)] = np.asarray(xyz, np.float32).reshape(-1)
  qpos[np.asarray(env._objs_quat_qpos_idxs)] = np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), int(num_cubes))
  # Park mocap ghosts off-table so the panel shows the solid target cubes.
  n_task = int(env._num_task_cubes)
  mocap_pos = np.tile(np.array([10.0, 10.0, 10.0], dtype=np.float32), (n_task, 1))
  mocap_quat = np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_task, 1))
  qvel = np.zeros(int(env._mj_model.nv), dtype=np.float32)
  img = np.asarray(env.render_from_info(
      qpos, qvel, mocap_pos, mocap_quat,
      height=480, width=640, camera=-1,
  ))
  Image.fromarray(img).save(out_png)
  print(f'wrote cubes {out_png}\n{np.asarray(xyz).reshape(-1, 3)}')


def _crop(path: str) -> np.ndarray:
  img = np.asarray(Image.open(path).convert('RGB'))
  return img[CROP[1]:CROP[3], CROP[0]:CROP[2]]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  try:
    from matplotlib import font_manager

    paper_font = font_manager.findfont(font_manager.FontProperties(
        family=['Times New Roman', 'Times', 'DejaVu Serif']))
    return ImageFont.truetype(paper_font, size)
  except (ImportError, OSError, ValueError):
    return ImageFont.load_default()


def _zoom_goal(image: np.ndarray) -> np.ndarray:
  """Enlarge the target structure while retaining the panel dimensions."""
  height, width = image.shape[:2]
  # Keep the crop aspect ratio equal to the 145x165 output so cube geometry
  # is enlarged uniformly rather than stretched horizontally.
  zoomed = Image.fromarray(image).crop((0, 28, 120, height))
  return np.asarray(
      zoomed.resize((width, height), Image.Resampling.LANCZOS))


def main() -> None:
  os.makedirs(OUT_DIR, exist_ok=True)
  work = os.path.join(REPO, 'figs', 'builderbench', 'c7t2', '_c7t2_s3_frames')
  os.makedirs(work, exist_ok=True)

  images = []
  labels = []
  status_labels = []
  for label, seed, iteration, frame, xyz, clean_frame, status in PANELS:
    if xyz is not None:
      tag = 'goal' if iteration is None else f's{int(seed)}_iter_{int(iteration):07d}_swap45'
      path = os.path.join(work, f'{tag}.png')
      if not os.path.isfile(path):
        _render_cubes(path, xyz)
    else:
      path = os.path.join(work, str(clean_frame))
      if not os.path.isfile(path):
        raise FileNotFoundError(
            f'clean checkpoint render missing: {path}')
    image = _crop(path)
    images.append(_zoom_goal(image) if label == 'Goal' else image)
    labels.append(label)
    status_labels.append(status)

  scale = 4
  n = len(images)
  panels = []
  for img in images:
    panel = Image.fromarray(img).resize(
        (img.shape[1] * scale, img.shape[0] * scale), Image.Resampling.LANCZOS)
    panels.append(panel)
  w, h = panels[0].size
  gap = 18
  progress_h = 214
  pad = 24
  canvas_w = pad * 2 + n * w + (n - 1) * gap
  canvas_h = pad + progress_h + h + pad
  canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
  draw = ImageDraw.Draw(canvas)
  panel_font = _font(92)
  goal_font = _font(148)
  unstable_font = _font(80)
  progress_font = _font(124)
  gold = (245, 176, 24)
  progress_color = (0, 0, 0)
  for i, (panel, label, status) in enumerate(
      zip(panels, labels, status_labels)):
    x = pad + i * (w + gap)
    y = pad + progress_h
    canvas.paste(panel, (x, y))
    if i == 0:
      draw.rounded_rectangle(
          (x + 10, y + 10, x + w - 11, y + h - 11),
          radius=26, outline=gold, width=20)
    panel_title = 'Goal' if i == 0 else f'Iter {label}'
    title_font = goal_font if i == 0 else panel_font
    bbox = draw.textbbox((0, 0), panel_title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(
        (x + (w - tw) / 2, y + (32 if i == 0 else 12)),
        panel_title, fill=(255, 255, 255), font=title_font,
        stroke_width=(6 if i == 0 else 3), stroke_fill=(25, 38, 52))
    if status is not None:
      warning = status
      bbox = draw.textbbox((0, 0), warning, font=unstable_font)
      warning_x = x + (w - (bbox[2] - bbox[0])) / 2
      draw.text(
          (warning_x, y + 118),
          warning, fill=(240, 65, 55), font=unstable_font,
          stroke_width=4, stroke_fill=(75, 12, 12))
  # Goal is a reference; the training arrow starts at the first checkpoint.
  # Single filled polygon so the shaft cannot run through the head.
  arrow_start = pad + (w + gap) + 24
  arrow_end = pad + (n - 1) * (w + gap) + w - 24
  head_len = 52
  head_half = 28
  shaft_half = 6
  arrow_y = pad + progress_h - head_half - 10
  shaft_end = arrow_end - head_len
  draw.polygon(
      [
          (arrow_start, arrow_y - shaft_half),
          (shaft_end, arrow_y - shaft_half),
          (shaft_end, arrow_y - head_half),
          (arrow_end, arrow_y),
          (shaft_end, arrow_y + head_half),
          (shaft_end, arrow_y + shaft_half),
          (arrow_start, arrow_y + shaft_half),
      ],
      fill=progress_color,
  )
  progress = 'training progress'
  bbox = draw.textbbox((0, 0), progress, font=progress_font)
  progress_w = bbox[2] - bbox[0]
  draw.text(
      ((arrow_start + arrow_end - progress_w) / 2, pad + 4),
      progress, fill=progress_color, font=progress_font)
  os.makedirs(OUT_DIR, exist_ok=True)
  pdf = f'{OUT_STEM}.pdf'
  canvas.save(pdf)
  preview = os.path.join('/tmp', 'bb_c7t2_s3_policy_progression_preview.png')
  canvas.save(preview)
  print(f'wrote {pdf}')
  print(f'wrote preview {preview}')
  print('labels:', labels)


if __name__ == '__main__':
  main()
