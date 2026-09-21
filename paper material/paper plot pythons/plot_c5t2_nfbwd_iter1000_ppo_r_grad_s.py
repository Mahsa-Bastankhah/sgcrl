#!/usr/bin/env python3
"""c5t2 nfbwd iter 1000 vs dual-gradreg iter 400: log p(g|s,a).

Stills and ‖∇_s r‖ stay on the unregularized episode (same blue as that
line). The regularized reward is overlaid in orange.

  python "paper material/paper plot pythons/plot_c5t2_nfbwd_iter1000_ppo_r_grad_s.py"
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
OUT_STEM = os.path.join(
    _paper, "paper plots", "final final plots",
    "c5t2_nfbwd_iter1000_ppo_r_grad_s")

_PROBE = os.path.join(
    _repo, "figs", "builderbench", "nf_logp_reward_probe",
    "c5t2_nfbwd_succ_s0_online_all")
CSV_PATH = os.path.join(
    _PROBE, "c5t2_nfbwd_succ_s0_online_iter_0001000.csv")
MP4_PATH = os.path.join(
    _PROBE, "c5t2_nfbwd_succ_s0_online_iter_0001000.mp4")
_PROBE_REG = os.path.join(
    _repo, "figs", "builderbench", "nf_logp_reward_probe",
    "c5t2_dgr_valuedgr_s2_online_all")
CSV_REG_PATH = os.path.join(
    _PROBE_REG, "c5t2_dgr_valuedgr_s2_online_iter_0000400.csv")
C_UNREG = ps.C["blue"]
C_REG = ps.C["vermillion"]

# Reward overlay strips in the probe video: 220 + 180 + 220.
_STRIP_H = 620
# Cube row only (no goal-marker pyramid). xa, ya, w, h in the env pane.
# Shifted right so the pink cube is fully inside the frame.
_INIT_CROP = (248, 290, 142, 38)
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


def _hex_rgb(color: str) -> tuple[int, int, int]:
  c = color.lstrip("#")
  return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _border_still(img: np.ndarray, color: str = C_UNREG) -> np.ndarray:
  im = Image.fromarray(img).convert("RGB")
  draw = ImageDraw.Draw(im)
  w, h = im.size
  draw.rectangle([0, 0, w - 1, h - 1], outline=_hex_rgb(color), width=3)
  return np.asarray(im)


def _disp_h_pts(img: np.ndarray, zoom: float, dpi: float) -> float:
  return float(img.shape[0]) * float(zoom) * 72.0 / float(dpi)


def _add_thumb(ax, img: np.ndarray, t: float, *, zoom: float, grad: float,
               align_x: float = 1.0, dx: float = 0.0,
               spacer_pts: float = 0.0, color: str = C_UNREG) -> None:
  # align_x=1: right edge on t (just before that time). 0: left edge on t.
  oi = OffsetImage(_border_still(img, color), zoom=zoom, interpolation="sinc")
  oi.image.set_clip_on(False)
  props = dict(fontsize=_GRAD_FS, color=color, ha="center")
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


def main() -> None:
  d = np.genfromtxt(CSV_PATH, delimiter=",", names=True)
  d_reg = np.genfromtxt(CSV_REG_PATH, delimiter=",", names=True)
  t = d["t"]
  logp = d["reward_logp_online"]
  t_reg = d_reg["t"]
  logp_reg = d_reg["reward_logp_online"]
  success = d["success"] >= 0.5
  grad_s = d["grad_s_norm"]
  pos_sum = d["pos_delta_sum"]
  cube_cols = [n for n in d.dtype.names if n.startswith("pos_delta_c")]

  t_succ = int(np.argmax(success))
  t_end = int(np.where(success)[0][-1])
  move_incl = float(pos_sum[t_succ:].sum())
  move_after = float(pos_sum[t_succ + 1:].sum())
  print(f"first success t = {t_succ} / {int(t[-1])}")
  print(f"last success t  = {t_end}")
  print(f"sum_i ||Δxyz_i||  success→end (incl success step) = {move_incl:.4g} m")
  print(f"sum_i ||Δxyz_i||  after success (excl success step) = {move_after:.4g} m")
  for name in cube_cols:
    print(f"  {name} success→end = {float(d[name][t_succ:].sum()):.4g} m")

  thumbs = {
      0: _label_thumb(
          _crop_scene(_extract_frame(MP4_PATH, 0), mode="init"), 0,
          emphasize=True),
      t_succ: _label_thumb(
          _crop_scene(_extract_frame(MP4_PATH, t_succ), mode="stack"), t_succ,
          emphasize=True),
      t_end: _label_thumb(
          _crop_scene(_extract_frame(MP4_PATH, t_end), mode="stack"), t_end,
          emphasize=True),
  }

  fig, ax = ps.figure("single")
  fig.set_size_inches(8.6, 7.0)
  zoom = {0: _THUMB_ZOOM_INIT, t_succ: _THUMB_ZOOM_STACK, t_end: _THUMB_ZOOM_STACK}
  h_pts = {k: _disp_h_pts(thumbs[k], zoom[k], fig.dpi) for k in thumbs}
  h_max = max(h_pts.values())

  ax.plot(t, logp, color=C_UNREG, label="without regularization")
  ax.plot(t_reg, logp_reg, color=C_REG, label="with regularization")
  ax.axvline(t_succ, color=ps.C["green"], linestyle=ps.DASH, zorder=2)
  ax.set_xlabel(r"$t$", fontsize=_FS_XLABEL, labelpad=4)
  ax.set_ylabel(
      r"$r(s,a)=\log p(g\mid s,a)$",
      fontsize=_FS_YLABEL, labelpad=10)
  ax.set_title("PIM reward", fontsize=_FS_TITLE, pad=ps.TITLE_PAD)
  ax.set_xlim(float(t[0]), float(t[-1]))
  y_lo = min(float(np.min(logp)), float(np.min(logp_reg)))
  y_hi = max(float(np.max(logp)), float(np.max(logp_reg)))
  ax.set_ylim(y_lo - 0.8, y_hi + 0.8)
  ax.yaxis.set_major_locator(mticker.FixedLocator([-26.0, -17.0, -6.0]))
  ax.yaxis.set_minor_locator(mticker.NullLocator())
  ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
  ps.style_axes(ax, which="major")
  ax.tick_params(axis="both", which="major", labelsize=_FS_TICK)
  ax.tick_params(axis="y", which="minor", left=False, length=0)
  ax.tick_params(axis="y", which="major", pad=6)
  ps.nice_legend(ax, loc="center right", overlay=True, fontsize=26)

  ax.annotate(
      "success", xy=(t_succ, logp[t_succ]),
      xytext=(8, -24), textcoords="offset points",
      color=ps.C["green"], va="top", ha="left",
      fontsize=_SUCCESS_FS, zorder=10)

  _add_thumb(
      ax, thumbs[0], 0.0, zoom=_THUMB_ZOOM_INIT, align_x=0.52, dx=-18,
      grad=grad_s[0], spacer_pts=h_max - h_pts[0])
  _add_thumb(
      ax, thumbs[t_succ], float(t_succ), zoom=_THUMB_ZOOM_STACK, dx=46,
      grad=grad_s[t_succ], spacer_pts=h_max - h_pts[t_succ])
  _add_thumb(
      ax, thumbs[t_end], float(t_end), zoom=_THUMB_ZOOM_STACK, dx=6,
      grad=grad_s[t_end], spacer_pts=h_max - h_pts[t_end])

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM, pad_inches=0.40)
  plt.close(fig)


if __name__ == "__main__":
  main()
