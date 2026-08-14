#!/usr/bin/env python3
"""Dummy clip of the Sawyer NF dual-strip overlay (raw + frozen r/σ_G).

No env / JAX. Uses the same matplotlib strip + compose helpers as training.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'videos', 'sawyer', 'dummy_nf_raw_and_norm.mp4')


def _render_reward_strip(rewards, success, t, width, height=220, *,
                         title='', ylabel='r', line_color='#5ec8ff'):
  T = len(rewards)
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  r_now = float(rewards[t])
  r_min = float(np.min(rewards)) - 0.35
  r_max = float(np.max(rewards)) + 0.35
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
    ax.axvline(first_succ, color='#ff8a4c', ls='--', lw=1.2, alpha=0.85)
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
              color='#7dffb0' if succ_now else '#ffe566',
              fontsize=16, fontweight='bold', va='center')
  ax_txt.text(0.05, 0.28, f't = {t}/{T - 1}', transform=ax_txt.transAxes,
              color='#c8d0d8', fontsize=10, va='center')
  ax_txt.text(0.05, 0.10, 'SUCCESS' if succ_now else 'no success',
              transform=ax_txt.transAxes,
              color='#7dffb0' if succ_now else '#ff8a4c',
              fontsize=11, fontweight='bold', va='center')
  fig.suptitle(title, color='#e8eef4', fontsize=10, y=0.96)
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def _compose(strip, render, extra_strips):
  strips = [strip, *extra_strips]
  rw = strips[0].shape[1]
  rh = int(round(render.shape[0] * (rw / render.shape[1])))
  render_r = np.asarray(
      Image.fromarray(render).resize((rw, rh), Image.Resampling.LANCZOS))
  return np.concatenate([*strips, render_r], axis=0)


def _stamp(frame, r_raw, r_norm, std):
  im = Image.fromarray(frame).convert('RGBA')
  draw = ImageDraw.Draw(im, 'RGBA')
  try:
    font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 22)
  except OSError:
    font = ImageFont.load_default()
  for y, text, color in (
      (12, f'r_online = {r_raw:+.3f}', (125, 255, 176)),
      (58, f'r_PPO = {r_norm:+.3f}   σ_G = {std:g}', (255, 229, 102)),
  ):
    bb = draw.textbbox((0, 0), text, font=font)
    bw, bh = bb[2] - bb[0] + 24, bb[3] - bb[1] + 12
    draw.rounded_rectangle(
        [12, y, 12 + bw, y + bh], radius=8,
        fill=(16, 22, 28, 210), outline=color + (255,), width=2)
    draw.text((24, y + 4), text, fill=color, font=font)
  return np.asarray(im.convert('RGB'))


def main():
  T, H, W = 40, 360, 640
  frames_rgb = []
  for t in range(T):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (28, 36, 44)
    x = int(40 + (W - 80) * t / max(T - 1, 1))
    y = int(H * 0.45 + 40 * np.sin(2 * np.pi * t / T))
    img[max(0, y - 12):y + 12, max(0, x - 12):x + 12] = (80, 140, 200)
    frames_rgb.append(img)

  rng = np.random.default_rng(0)
  rewards = (-8.0 + 0.15 * np.arange(T) + rng.normal(0, 0.4, T)).astype(np.float32)
  rewards[28:] += 3.0
  # Dummy EMA/target: lagged online logp (what PPO actually uses).
  rewards_tgt = rewards.copy()
  rewards_tgt[1:] = 0.85 * rewards_tgt[:-1] + 0.15 * rewards[1:]
  success = np.zeros(T, dtype=np.float32)
  success[32:] = 1.0
  std = 12.5
  rewards_norm = (rewards_tgt / std).astype(np.float32)
  rewards_norm = rewards_norm + success  # extrew after norm, scale=1

  frames = []
  for t in range(T):
    strip = _render_reward_strip(
        rewards, success, t, width=W, height=220,
        title=r'dummy  online NF logp (raw)',
        ylabel=r'$r_{\mathrm{online}}=\log p_{\mathrm{NF}}$',
        line_color='#7dffb0')
    extra = _render_reward_strip(
        rewards_norm, success, t, width=W, height=180,
        title=rf'PPO / value target  $r_{{\mathrm{{tgt}}}}/\sigma_G$  ($\sigma_G$={std:g}, frozen)',
        ylabel=r'$r_{\mathrm{PPO}}$', line_color='#ffe566')
    composed = _compose(strip, frames_rgb[t], [extra])
    composed = _stamp(composed, float(rewards[t]), float(rewards_norm[t]), std)
    h, w = composed.shape[:2]
    composed = composed[: h - (h % 2), : w - (w % 2)]
    frames.append(composed)
  frames.extend([frames[-1]] * 8)

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  old_ld = os.environ.pop('LD_LIBRARY_PATH', None)
  try:
    import imageio.v2 as imageio
    imageio.mimwrite(OUT, frames, fps=12, codec='libx264', quality=8)
  finally:
    if old_ld is not None:
      os.environ['LD_LIBRARY_PATH'] = old_ld
  still = os.path.splitext(OUT)[0] + '.png'
  Image.fromarray(frames[len(frames) // 2]).save(still)
  print(f'wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)')
  print(f'wrote {still}')
  print(f'raw [{rewards.min():.2f}, {rewards.max():.2f}]  '
        f'norm [{rewards_norm.min():.3f}, {rewards_norm.max():.3f}]  σ_G={std}')


if __name__ == '__main__':
  main()
