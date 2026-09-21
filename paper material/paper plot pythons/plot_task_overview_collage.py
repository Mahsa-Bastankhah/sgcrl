#!/usr/bin/env python3
"""Paper collage: four task stills in a 2x2.

  Sawyer peg              |  BuilderBench
  Allegro grasp and throw |  ManiSkill

Sawyer uses a clean successful scripted insertion frame. Allegro is a
zoomed NVIDIA-init oblique still (hand + cube + table + bucket). Jenga
has a close physically rendered 45-degree middle layer. ManiSkill is
restyled with sparse room partitions.

  python "paper material/paper plot pythons/plot_task_overview_collage.py"
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)

import paper_style as ps  # noqa: E402

ps.apply()

ASSET_DIR = os.path.join(_REPO, 'figs', 'paper_task_collage')
THROW_SRC = os.path.join(
    _REPO, 'figs', 'allegro_kuka_throw', 'throw_default_init',
    'throw_default_oblique.png')
OUT_STEM = os.path.join(
    _PAPER, 'paper plots', 'final final plots', 'task_overview_collage')

# Drawn near paper_style single width so a 0.9\\textwidth include
# leaves panel titles around 11–12pt.
FS_PANEL = 28
PANEL_PX = 720
# Zoom on hand / cube / table / bucket; drops the Isaac Gym debug banner.
THROW_CROP = (185, 90, 815, 720)

PANELS = (
    ('Sawyer peg', 'panel_peg.png'),
    ('BuilderBench', 'panel_jenga.png'),
    ('Allegro grasp and throw', 'panel_allegro_throw.png'),
    ('ManiSkill', 'panel_fetch_room.png'),
)


def _load(path: str) -> np.ndarray:
  return np.asarray(Image.open(path).convert('RGB'))


def _compress_pdf(src: str, dst: str, *, resolution: int = 150,
                  quality: int = 80) -> None:
  """JPEG-downsample rasters; titles stay vector. ~80–90KB vs ~2MB raw."""
  subprocess.check_call([
      'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
      '-dNOPAUSE', '-dQUIET', '-dBATCH',
      '-dAutoFilterColorImages=false',
      '-dColorImageFilter=/DCTEncode',
      '-dDownsampleColorImages=true',
      '-dColorImageDownsampleType=/Bicubic',
      f'-dColorImageResolution={resolution}',
      f'-dJPEGQ={quality}',
      '-dAutoFilterGrayImages=false',
      '-dGrayImageFilter=/DCTEncode',
      f'-sOutputFile={dst}',
      src,
  ])


def _write_throw_panel(src: str, dst: str) -> None:
  im = Image.open(src).convert('RGB')
  crop = im.crop(THROW_CROP)
  side = min(crop.size)
  left = (crop.width - side) // 2
  top = (crop.height - side) // 2
  square = crop.crop((left, top, left + side, top + side))
  square.resize((PANEL_PX, PANEL_PX), Image.Resampling.LANCZOS).save(dst)


def ensure_assets() -> None:
  """Rebuild stills if missing (idempotent)."""
  os.makedirs(ASSET_DIR, exist_ok=True)
  peg_out = os.path.join(ASSET_DIR, 'panel_peg.png')
  jenga_out = os.path.join(ASSET_DIR, 'panel_jenga.png')
  throw_out = os.path.join(ASSET_DIR, 'panel_allegro_throw.png')
  fetch_out = os.path.join(ASSET_DIR, 'panel_fetch_room.png')

  if not os.path.isfile(peg_out):
    raise FileNotFoundError(peg_out)
  if not os.path.isfile(jenga_out):
    raise FileNotFoundError(jenga_out)
  if not os.path.isfile(fetch_out):
    raise FileNotFoundError(fetch_out)
  if not os.path.isfile(THROW_SRC):
    raise FileNotFoundError(THROW_SRC)
  _write_throw_panel(THROW_SRC, throw_out)


def main() -> None:
  ensure_assets()
  fig, axes = plt.subplots(2, 2, figsize=(8.6, 9.4), layout='constrained')
  fig.set_constrained_layout_pads(
      w_pad=0.06, h_pad=0.12, hspace=0.10, wspace=0.06)

  for ax, (title, name) in zip(axes.ravel(), PANELS):
    path = os.path.join(ASSET_DIR, name)
    ax.imshow(_load(path))
    ax.set_title(title, fontsize=FS_PANEL, pad=10, color='#222222')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_linewidth(0.95)
      spine.set_color('#B0B0B0')

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  with tempfile.TemporaryDirectory() as td:
    raw = os.path.join(td, 'raw')
    ps.savefig(fig, raw, dpi=220)
    _compress_pdf(raw + '.pdf', OUT_STEM + '.pdf')
  preview = '/tmp/task_overview_collage_preview.png'
  fig.savefig(preview, dpi=120, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'Saved: {OUT_STEM}.pdf ({os.path.getsize(OUT_STEM + ".pdf")} bytes)')
  print(f'preview: {preview}')


if __name__ == '__main__':
  main()
