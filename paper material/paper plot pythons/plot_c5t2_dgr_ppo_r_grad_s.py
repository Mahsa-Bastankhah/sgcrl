#!/usr/bin/env python3
"""c5t2 final dual-gradreg: log p(g|s,a) with ‖∇_s r‖ under each still.

Plots seed-2 iters 400 and 500 (shortly after first train success).

  python "paper material/paper plot pythons/plot_c5t2_dgr_ppo_r_grad_s.py"
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.offsetbox import (
    AnnotationBbox, DrawingArea, OffsetImage, TextArea, VPacker)
from PIL import Image, ImageDraw, ImageFont

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
  sys.path.insert(0, _here)

import paper_style as ps
ps.apply()

_paper = os.path.dirname(_here)
_repo = os.path.dirname(_paper)
_PROBE = os.path.join(
    _repo, "figs", "builderbench", "nf_logp_reward_probe",
    "c5t2_dgr_valuedgr_s2_online_all")
_OUT_DIR = os.path.join(_paper, "paper plots", "final final plots")

ITERS = (400, 500)

# Reward overlay strips in the probe video: 220 + 180 + 220.
_STRIP_H = 620
# Cube row only (no goal-marker pyramid). xa, ya, w, h in the env pane.
# Wider / left of the nfbwd crop so all five start cubes stay in frame.
_INIT_CROP = (220, 290, 170, 38)
_THUMB_ZOOM_INIT = 0.92
_THUMB_ZOOM_STACK = 1.00
_SUCCESS_FS = 40
_GRAD_FS = 28
_FS_YLABEL = 36
_FS_XLABEL = 40
_FS_TITLE = 32
_FS_TICK = 32


def _ffmpeg() -> str:
  try:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()
  except Exception:
    return "ffmpeg"


def _extract_frame(mp4: str, index: int) -> np.ndarray:
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, f"t{index:03d}.png")
    subprocess.check_call([
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", mp4,
        "-vf", f"select=eq(n\\,{int(index)})",
        "-vsync", "vfr", "-frames:v", "1",
        out,
    ])
    return np.asarray(Image.open(out).convert("RGB"))


def _env_row0(frame: np.ndarray) -> int:
  mean = frame.mean(axis=1)
  r, g = mean[:, 0], mean[:, 1]
  hits = np.where((r > 100.0) & (r > g + 15.0))[0]
  if len(hits):
    return int(hits[0])
  return min(_STRIP_H, int(frame.shape[0]) - 1)


def _sat_bbox(env: np.ndarray, y0f: float, y1f: float, x0f: float, x1f: float,
              thresh: float = 80.0) -> tuple[int, int, int, int]:
  sat = env.astype(np.float32).max(axis=2) - env.astype(np.float32).min(axis=2)
  h, w = env.shape[:2]
  y0b, y1b = int(round(y0f * h)), int(round(y1f * h))
  x0b, x1b = int(round(x0f * w)), int(round(x1f * w))
  ys, xs = np.where(sat[y0b:y1b, x0b:x1b] > thresh)
  if len(ys) < 10:
    ys, xs = np.where(sat[y0b:y1b, x0b:x1b] > 0.7 * thresh)
  if len(ys) < 10:
    raise RuntimeError("could not locate cubes in probe frame")
  return (int(xs.min() + x0b), int(ys.min() + y0b),
          int(xs.max() + x0b), int(ys.max() + y0b))


def _stack_crop(env: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
  """Tight crop on the stack; little floor under the cubes."""
  x0, y0, x1, y1 = bbox
  pad_x, pad_top, pad_bot = 16, 18, 12
  xa = max(0, x0 - pad_x)
  ya = max(0, y0 - pad_top)
  xb = min(env.shape[1], x1 + pad_x)
  yb = min(env.shape[0], y1 + pad_bot)
  return env[ya:yb, xa:xb]


def _boost_still(img: np.ndarray) -> np.ndarray:
  """Mild contrast lift so the pale pink cube and goal dots read in print."""
  x = img.astype(np.float32) / 255.0
  x = np.clip((x - 0.07) * 1.28 + 0.07, 0.0, 1.0)
  return (x * 255.0).astype(np.uint8)


def _hide_goal_markers(crop: np.ndarray) -> np.ndarray:
  """Paint out the mocap goal pyramid under the cube row."""
  h, w = crop.shape[:2]
  floor = np.median(crop[h // 2 :, :8].reshape(-1, 3), axis=0)
  rgb = crop.astype(np.float32)
  sat = rgb.max(axis=2) - rgb.min(axis=2)
  yy = np.arange(h)[:, None]
  xx = np.arange(w)[None, :]
  lower_center = (
      (yy > 0.55 * h) & (xx > 0.22 * w) & (xx < 0.78 * w)
      & (sat > 12.0) & (rgb.max(axis=2) > 50.0))
  out = crop.copy()
  out[lower_center] = floor.astype(np.uint8)
  return out


def _crop_scene(frame: np.ndarray, *, mode: str) -> np.ndarray:
  env = frame[_env_row0(frame):]
  if mode == "init":
    xa, ya, w, h = _INIT_CROP
    crop = _hide_goal_markers(env[ya:ya + h, xa:xa + w])
  else:
    bbox = _sat_bbox(env, 0.58, 0.90, 0.30, 0.70)
    crop = _stack_crop(env, bbox)
  return _boost_still(crop)


def _label_thumb(img: np.ndarray, t: int, *, emphasize: bool = False) -> np.ndarray:
  """White t=… on the still; no plate. emphasize → slightly larger/bold."""
  import matplotlib.font_manager as fm
  im = Image.fromarray(img).convert("RGB")
  draw = ImageDraw.Draw(im)
  text = f"t={int(t)}"
  size = 20
  if emphasize:
    size = 24
  font = ImageFont.load_default()
  names = (("DejaVu Sans", "bold"), ("DejaVuSans-Bold", "normal"),
           ("Liberation Sans", "bold")) if emphasize else (
               ("DejaVu Sans", "normal"), ("DejaVuSans", "normal"),
               ("Liberation Sans", "normal"))
  for name, weight in names:
    try:
      path = fm.findfont(fm.FontProperties(family=name, weight=weight))
      font = ImageFont.truetype(path, size=size)
      break
    except (OSError, ValueError):
      continue
  bbox = draw.textbbox((0, 0), text, font=font)
  th = bbox[3] - bbox[1]
  x = 4
  y = max(2, im.size[1] - th - 14)
  draw.text((x, y), text, font=font, fill=(255, 255, 255))
  return np.asarray(im)


def _border_still(img: np.ndarray) -> np.ndarray:
  im = Image.fromarray(img).convert("RGB")
  draw = ImageDraw.Draw(im)
  w, h = im.size
  draw.rectangle([0, 0, w - 1, h - 1], outline=(136, 136, 136), width=2)
  return np.asarray(im)


def _disp_h_pts(img: np.ndarray, zoom: float, dpi: float) -> float:
  return float(img.shape[0]) * float(zoom) * 72.0 / float(dpi)


def _add_thumb(ax, img: np.ndarray, t: float, *, zoom: float, grad: float,
               align_x: float = 1.0, dx: float = 0.0,
               spacer_pts: float = 0.0) -> None:
  oi = OffsetImage(_border_still(img), zoom=zoom, interpolation="sinc")
  oi.image.set_clip_on(False)
  props = dict(fontsize=_GRAD_FS, color=ps.C["vermillion"], ha="center")
  lab = TextArea(
      rf"$\Vert\nabla_s r\Vert={float(grad):.0f}$", textprops=props)
  children = [oi]
  if spacer_pts > 0.5:
    children.append(DrawingArea(1, spacer_pts, 0, 0))
  children.append(lab)
  pack = VPacker(children=children, align="center", pad=0, sep=6)
  ab = AnnotationBbox(
      pack,
      (t, 0.0),
      xycoords=("data", "axes fraction"),
      xybox=(dx, -40.0),
      boxcoords="offset points",
      box_alignment=(align_x, 1.0),
      pad=0.0,
      frameon=False,
      annotation_clip=False,
  )
  ab.set_clip_on(False)
  ax.add_artist(ab)


def _paths(it: int) -> tuple[str, str, str]:
  stem = f"c5t2_dgr_valuedgr_s2_online_iter_{int(it):07d}"
  csv_path = os.path.join(_PROBE, f"{stem}.csv")
  mp4_path = os.path.join(_PROBE, f"{stem}.mp4")
  out_stem = os.path.join(_OUT_DIR, f"c5t2_dgr_iter{int(it):04d}_ppo_r_grad_s")
  return csv_path, mp4_path, out_stem


def _plot_one(it: int) -> None:
  csv_path, mp4_path, out_stem = _paths(it)
  if not os.path.isfile(csv_path) or not os.path.isfile(mp4_path):
    raise FileNotFoundError(f"missing probe for iter {it}: {csv_path}")

  d = np.genfromtxt(csv_path, delimiter=",", names=True)
  t = d["t"]
  logp = d["reward_logp_online"]
  success = d["success"] >= 0.5
  grad_s = d["grad_s_norm"]
  if not np.any(success):
    raise RuntimeError(f"iter {it}: probe episode has no success")

  t_succ = int(np.argmax(success))
  t_end = int(np.where(success)[0][-1])
  print(f"iter {it}: first success t = {t_succ} / {int(t[-1])}")
  print(f"iter {it}: last success t  = {t_end}")
  print(
      f"iter {it}: logp start={logp[0]:.2f} succ={logp[t_succ]:.2f} "
      f"end={logp[t_end]:.2f} range={logp.max() - logp.min():.2f}")
  print(
      f"iter {it}: grad_s start={grad_s[0]:.1f} succ={grad_s[t_succ]:.1f} "
      f"end={grad_s[t_end]:.1f} max={grad_s.max():.1f}")

  thumb_ts = [0, t_succ]
  if t_end != t_succ:
    thumb_ts.append(t_end)
  thumbs = {}
  for ti in thumb_ts:
    mode = "init" if ti == 0 else "stack"
    thumbs[ti] = _label_thumb(
        _crop_scene(_extract_frame(mp4_path, ti), mode=mode), ti,
        emphasize=True)

  fig, ax = ps.figure("single")
  fig.set_size_inches(8.6, 7.0)
  zoom = {k: (_THUMB_ZOOM_INIT if k == 0 else _THUMB_ZOOM_STACK) for k in thumbs}
  h_pts = {k: _disp_h_pts(thumbs[k], zoom[k], fig.dpi) for k in thumbs}
  h_max = max(h_pts.values())

  ax.plot(t, logp, color=ps.C["blue"])
  ax.axvline(t_succ, color=ps.C["green"], linestyle=ps.DASH, zorder=2)
  ax.set_xlabel(r"$t$", fontsize=_FS_XLABEL, labelpad=4)
  ax.set_ylabel(
      r"$r(s,a)=\log p(g\mid s,a)$",
      fontsize=_FS_YLABEL, labelpad=10)
  ax.set_title(
      f"PIM reward + grad regularizer  (iter {int(it)})",
      fontsize=_FS_TITLE, pad=ps.TITLE_PAD)
  ax.set_xlim(float(t[0]), float(t[-1]))
  lo = float(np.min(logp))
  hi = float(np.max(logp))
  pad = max(0.35, 0.08 * (hi - lo if hi > lo else 1.0))
  ax.set_ylim(lo - pad, hi + pad)
  ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
  ax.yaxis.set_minor_locator(mticker.NullLocator())
  ps.style_axes(ax, which="major")
  ax.tick_params(axis="both", which="major", labelsize=_FS_TICK)
  ax.tick_params(axis="y", which="minor", left=False, length=0)
  ax.tick_params(axis="y", which="major", pad=6)

  ax.annotate(
      "success", xy=(t_succ, logp[t_succ]),
      xytext=(8, -24), textcoords="offset points",
      color=ps.C["green"], va="top", ha="left",
      fontsize=_SUCCESS_FS, zorder=10)

  dx = {0: -18, t_succ: 46, t_end: 6}
  align = {0: 0.52}
  _add_thumb(
      ax, thumbs[0], 0.0, zoom=_THUMB_ZOOM_INIT, align_x=align[0],
      dx=dx[0], grad=grad_s[0], spacer_pts=h_max - h_pts[0])
  _add_thumb(
      ax, thumbs[t_succ], float(t_succ), zoom=_THUMB_ZOOM_STACK,
      dx=dx[t_succ], grad=grad_s[t_succ],
      spacer_pts=h_max - h_pts[t_succ])
  if t_end in thumbs:
    _add_thumb(
        ax, thumbs[t_end], float(t_end), zoom=_THUMB_ZOOM_STACK,
        dx=dx[t_end], grad=grad_s[t_end],
        spacer_pts=h_max - h_pts[t_end])

  os.makedirs(os.path.dirname(out_stem), exist_ok=True)
  ps.savefig(fig, out_stem, pad_inches=0.40)
  plt.close(fig)
  print(f"→ {out_stem}.pdf")


def main() -> None:
  for it in ITERS:
    _plot_one(it)


if __name__ == "__main__":
  main()
