#!/usr/bin/env python3
"""Paper collage: the eight BuilderBench tasks used in the env-row plots.

Titles match plot_baseline_env_row.py / plot_density_estimator_env_row.py.
Each panel is the hard-goal cube configuration, rendered with the same
front camera as the c5t2 paper stills (lookat 0.27,0,0.06; az=0; elev=-25;
zoom=5). The 7-cube zig-zag uses az=180 so it matches existing videos.
Mocap ghosts are parked off-table. Cubes masked out of the goal
(underspecified towers, overhang) are drawn translucent, marked with ?,
and a white “? not specified” legend is painted at the bottom of the still.

  python "paper material/paper plot pythons/plot_bb_task_overview.py"
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BUILDERBENCH_MJX_IMPL", "jax")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)
_BB = os.environ.get("BUILDERBENCH_ROOT", "/n/fs/mislresearch/builderbench")
if _BB not in sys.path:
  sys.path.insert(0, _BB)

import paper_style as ps  # noqa: E402

# Midpoint of BuilderBench target-origin sampling (envs/builderbench_utils.py).
_TARGET_MID = np.array([0.27, 0.0, 0.02], dtype=np.float32)

ps.apply()

ASSET_DIR = os.path.join(_REPO, "figs", "paper_task_collage")
OUT_STEM = os.path.join(
    _PAPER, "paper plots", "final final plots", "bb_task_overview")
PREVIEW = "/tmp/bb_task_overview_preview.png"

# Drawn large so a textwidth include leaves titles around 11–12pt.
FS_PANEL = 28
# Same camera as the c5t2 paper stills
# (job_bb_c5t2_nf_goal_preimage_ckpt10_front_ghost_warp.slurm).
# c7t2 is the exception: az=180 matches every existing zig-zag video /
# policy-progression still (orange singleton on the right).
PANEL_W = 960
PANEL_H = 720
# Slightly wider than the pyramid-only stills so the 4-cube stack is not clipped.
CAM_ZOOM = 3.8
CAM_AZIMUTH = 0.0
CAM_AZIMUTH_C7T2 = 180.0
CAM_ELEVATION = -25.0
CAM_LOOKAT = np.array([0.27, 0.0, 0.06], dtype=np.float64)
CAM_EXTENT = 0.8

# Order / names match the paper env-row figures. Titles wrap so a
# 2×4 include does not collide.
TASKS = (
    dict(key="c3t1", cubes=3, task=0, title="3-cube\nstacking"),
    dict(key="c4t1", cubes=4, task=0, title="4-cube\nstacking"),
    dict(key="c4t2", cubes=4, task=1, title="4-cube\nparallel towers"),
    dict(key="c5t2", cubes=5, task=1, title="5-cube\npyramid"),
    dict(key="c5t3", cubes=5, task=2, title="Underspecified\ntowers"),
    dict(key="c5t4", cubes=5, task=3, title="5-cube\nmaximum overhang"),
    dict(key="c7t2", cubes=7, task=1, title="7-cube\nzig-zag tower"),
    dict(key="c8t2", cubes=8, task=1, title="8-cube\nJenga tower"),
)


def _compress_pdf(src: str, dst: str, *, resolution: int = 150,
                  quality: int = 80) -> None:
  """JPEG-downsample rasters; titles stay vector."""
  subprocess.check_call([
      "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
      "-dNOPAUSE", "-dQUIET", "-dBATCH",
      "-dAutoFilterColorImages=false",
      "-dColorImageFilter=/DCTEncode",
      "-dDownsampleColorImages=true",
      "-dColorImageDownsampleType=/Bicubic",
      f"-dColorImageResolution={resolution}",
      f"-dJPEGQ={quality}",
      "-dAutoFilterGrayImages=false",
      "-dGrayImageFilter=/DCTEncode",
      f"-sOutputFile={dst}",
      src,
  ])


def _load_task_npz(num_cubes: int):
  path = os.path.join(
      _BB, "builderbench", "tasks", f"creative-{int(num_cubes)}.npz")
  return np.load(path)


def _task_goal_xyz(num_cubes: int, task_index: int) -> np.ndarray:
  """All cube centers at the designed goal offsets (including masked cubes)."""
  data = _load_task_npz(num_cubes)
  offsets = np.asarray(data["goals"][task_index], dtype=np.float32).reshape(-1, 3)
  if offsets.shape[0] != int(num_cubes):
    raise ValueError(
        f"creative-{num_cubes} task{task_index + 1} has {offsets.shape[0]} "
        f"goal cubes, expected {num_cubes}")
  return (_TARGET_MID.reshape(1, 3) + offsets).astype(np.float32)


def _task_mask(num_cubes: int, task_index: int) -> np.ndarray:
  """True = cube xyz is in the goal vector."""
  data = _load_task_npz(num_cubes)
  if "masks" not in data.files:
    return np.ones((int(num_cubes),), dtype=bool)
  return np.asarray(data["masks"][task_index], dtype=bool).reshape(-1)


def _make_env(num_cubes: int):
  from builderbench.creative_cube import CreativeCube, default_config

  cfg = default_config()
  cfg.num_cubes = int(num_cubes)
  cfg.task_id = 0
  cfg.permute_start_boxes = False
  cfg.impl = os.environ.get("BUILDERBENCH_MJX_IMPL", "jax")
  cfg.nconmax = 256
  cfg.njmax = 64
  return CreativeCube(config=cfg)


def _yaw_quat(yaw: float) -> np.ndarray:
  half = 0.5 * float(yaw)
  return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _task_quats(key: str, num_cubes: int) -> np.ndarray:
  """Identity yaw, except the 8-cube Jenga middle layer (z=0.04) at 45°."""
  quats = np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_cubes, 1))
  if key == "c8t2":
    # Cubes 2 and 3 are the middle pair in creative-8 task2.
    q = _yaw_quat(np.pi / 4)
    quats[2] = q
    quats[3] = q
  return quats.reshape(-1)


def _task_azimuth(key: str) -> float:
  return CAM_AZIMUTH_C7T2 if key == "c7t2" else CAM_AZIMUTH


def _front_camera(lookat, zoom: float, *, azimuth: float | None = None,
                  elevation: float | None = None):
  """Front cam (az=0, elev=-25), lookat/zoom set per structure."""
  import mujoco

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = np.asarray(lookat, dtype=np.float64).reshape(3)
  cam.distance = (1.5 * CAM_EXTENT) / max(float(zoom), 1e-3)
  cam.azimuth = CAM_AZIMUTH if azimuth is None else float(azimuth)
  cam.elevation = CAM_ELEVATION if elevation is None else float(elevation)
  return cam


def _lookat_and_zoom(pts: np.ndarray) -> tuple[np.ndarray, float]:
  """Frame the full goal so wide tasks (overhang, Jenga) are not clipped."""
  pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
  lo, hi = pts.min(axis=0), pts.max(axis=0)
  center = 0.5 * (lo + hi)
  span = hi - lo
  size = float(max(span[1], span[2], 0.08))
  lookat = np.array([0.27, float(center[1]), max(float(center[2]), 0.05)])
  dist = max(0.30, 0.16 + 2.2 * size)
  return lookat, (1.5 * CAM_EXTENT) / dist


def _cam_basis(lookat, distance: float, azimuth: float, elevation: float):
  """MuJoCo free-camera pose. az=0 is from +x, looking toward −x."""
  az = np.deg2rad(float(azimuth))
  el = np.deg2rad(float(elevation))
  lookat = np.asarray(lookat, dtype=np.float64).reshape(3)
  offset = distance * np.array(
      [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
  pos = lookat + offset
  forward = lookat - pos
  forward /= max(np.linalg.norm(forward), 1e-8)
  right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
  right /= max(np.linalg.norm(right), 1e-8)
  up = np.cross(forward, right)
  return pos, right, up, forward


def _project_xyz(xyz, lookat, distance: float, *, width: int, height: int,
                 fovy: float) -> tuple[float, float] | None:
  pos, right, up, forward = _cam_basis(
      lookat, distance, CAM_AZIMUTH, CAM_ELEVATION)
  rel = np.asarray(xyz, dtype=np.float64).reshape(3) - pos
  z = float(np.dot(rel, forward))
  if z <= 1e-5:
    return None
  x = float(np.dot(rel, right))
  y = float(np.dot(rel, up))
  f = 0.5 * height / np.tan(np.deg2rad(float(fovy)) / 2.0)
  u = 0.5 * width + f * x / z
  v = 0.5 * height - f * y / z
  return u, v


def _font(size: int):
  from PIL import ImageFont

  try:
    return ImageFont.truetype(
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", size)
  except Exception:
    return ImageFont.load_default()


def _draw_qmark(draw, u: float, v: float, *, r: int = 22, q_size: int = 36) -> None:
  font = _font(q_size)
  draw.ellipse(
      (u - r, v - r, u + r, v + r),
      fill=(255, 255, 255), outline=(20, 20, 20), width=3)
  tb = draw.textbbox((0, 0), "?", font=font)
  tw, th = tb[2] - tb[0], tb[3] - tb[1]
  draw.text((u - 0.5 * tw, v - 0.55 * th), "?", font=font, fill=(20, 20, 20))


def _draw_unspec_marks(img: np.ndarray, pts: np.ndarray, mask: np.ndarray,
                       lookat, zoom: float, *, fovy: float) -> np.ndarray:
  """Paint a circled ? on hidden cubes and a matching in-frame caption."""
  from PIL import ImageDraw

  dist = (1.5 * CAM_EXTENT) / max(float(zoom), 1e-3)
  im = Image.fromarray(img)
  draw = ImageDraw.Draw(im)
  h, w = img.shape[:2]
  r = 22
  for p, keep in zip(np.asarray(pts).reshape(-1, 3), mask.reshape(-1)):
    if keep:
      continue
    uv = _project_xyz(p, lookat, dist, width=w, height=h, fovy=fovy)
    if uv is None:
      continue
    u, v = uv
    if not (0 <= u < w and 0 <= v < h):
      continue
    _draw_qmark(draw, u, v, r=r)

  label = "not specified"
  font = _font(64)
  lr, lq = 32, 52
  tb = draw.textbbox((0, 0), label, font=font)
  tw, th = tb[2] - tb[0], tb[3] - tb[1]
  gap = 18
  total = 2 * lr + gap + tw
  x0 = 0.5 * (w - total)
  y = h - 72
  _draw_qmark(draw, x0 + lr, y, r=lr, q_size=lq)
  tx, ty = x0 + 2 * lr + gap, y - 0.45 * th
  for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
    draw.text((tx + dx, ty + dy), label, font=font, fill=(20, 20, 20))
  draw.text((tx, ty), label, font=font, fill=(255, 255, 255))
  return np.asarray(im)


def _fade_masked_cubes(model, num_cubes: int, mask: np.ndarray,
                       *, alpha: float = 0.32) -> np.ndarray:
  """Ghost cubes that are not in the goal. Returns rgba backup to restore."""
  import mujoco

  backup = np.array(model.geom_rgba, copy=True)
  for i, keep in enumerate(np.asarray(mask).reshape(-1).tolist()):
    if keep:
      continue
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{i}")
    if gid >= 0:
      model.geom_rgba[gid, 3] = float(alpha)
  return backup


def _render_goal(env, task: dict) -> np.ndarray:
  import mujoco

  num_cubes = int(task["cubes"])
  task_index = int(task["task"])
  goal = _task_goal_xyz(num_cubes, task_index)
  mask = _task_mask(num_cubes, task_index)
  qpos = np.array(env._init_q, copy=True)
  qpos[np.asarray(env._objs_pos_qpos_idxs)] = goal.reshape(-1)
  qpos[np.asarray(env._objs_quat_qpos_idxs)] = _task_quats(task["key"], num_cubes)

  env._mj_model.vis.global_.offwidth = max(
      int(env._mj_model.vis.global_.offwidth), PANEL_W)
  env._mj_model.vis.global_.offheight = max(
      int(env._mj_model.vis.global_.offheight), PANEL_H)

  d = mujoco.MjData(env._mj_model)
  d.qpos[:] = qpos
  d.qvel[:] = 0
  d.mocap_pos[:] = np.array([10.0, 10.0, 10.0])
  d.mocap_quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
  mujoco.mj_forward(env._mj_model, d)

  lookat, zoom = _lookat_and_zoom(goal)
  cam = _front_camera(lookat, zoom, azimuth=_task_azimuth(task["key"]))
  rgba_bak = _fade_masked_cubes(env._mj_model, num_cubes, mask)
  try:
    renderer = mujoco.Renderer(env._mj_model, height=PANEL_H, width=PANEL_W)
    renderer.update_scene(d, camera=cam)
    img = np.asarray(renderer.render(), dtype=np.uint8)
    renderer.close()
  finally:
    env._mj_model.geom_rgba[:] = rgba_bak
  if not np.all(mask):
    fovy = float(env._mj_model.vis.global_.fovy)
    img = _draw_unspec_marks(img, goal, mask, lookat, zoom, fovy=fovy)
  return img


def ensure_stills() -> None:
  os.makedirs(ASSET_DIR, exist_ok=True)
  envs: dict[int, object] = {}
  for task in TASKS:
    out = os.path.join(ASSET_DIR, f"bb_{task['key']}.png")
    n = int(task["cubes"])
    if n not in envs:
      print(f"building creative-{n} scene…", flush=True)
      envs[n] = _make_env(n)
    img = _render_goal(envs[n], task)
    Image.fromarray(img).save(out)
    print(f"wrote {out}", flush=True)


def _load(path: str) -> np.ndarray:
  return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
  ensure_stills()
  fig, axes = plt.subplots(2, 4, figsize=(17.2, 9.0), layout="constrained")
  fig.set_constrained_layout_pads(
      w_pad=0.08, h_pad=0.08, hspace=0.04, wspace=0.10)

  for ax, task in zip(axes.ravel(), TASKS):
    path = os.path.join(ASSET_DIR, f"bb_{task['key']}.png")
    ax.imshow(_load(path))
    ax.set_title(task["title"], fontsize=FS_PANEL, pad=6, color="#222222")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_linewidth(0.95)
      spine.set_color("#B0B0B0")

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  with tempfile.TemporaryDirectory() as td:
    raw = os.path.join(td, "raw")
    ps.savefig(fig, raw, dpi=220)
    _compress_pdf(raw + ".pdf", OUT_STEM + ".pdf")
  fig.savefig(PREVIEW, dpi=120, bbox_inches="tight", pad_inches=0.05)
  plt.close(fig)
  print(f"Saved: {OUT_STEM}.pdf ({os.path.getsize(OUT_STEM + '.pdf')} bytes)")
  print(f"preview: {PREVIEW}")


if __name__ == "__main__":
  main()
