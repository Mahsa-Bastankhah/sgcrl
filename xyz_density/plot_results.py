#!/usr/bin/env python3
"""Plot xyz density-probe results across CRL / NF / TD3 / FM / TDInfoNCE.

Produces:
  1. Five categorical-accuracy curves
     (train / in-dist / az-OOD / state-OOD / axy-OOD),
     each with all models + chance baseline.
  2. Three OOD density histogram families:
       - az_ood:   s=0, a_z=0.5  → p̂(z'|s,a)
       - state_ood: s=(ood_high_xy,ood_high_xy,0), a_z=0 → p̂(x'|s,a)
         (current default / reeval target: ood_high_xy=15)
       - axy_ood:  s=0, a_x,a_y ~ U[2,4], a_z=0 → p̂(x'|s,a)

Example::

  python -u xyz_density/plot_results.py \\
      --log_root=logs/xyz_density_az0_g05_100m \\
      --out_dir=figs/xyz_density_az0_g05_100m
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODES = ('crl', 'nf', 'nf_compact', 'nf_tiny', 'td3', 'fm', 'tdinfonce')
MODE_COLORS = {
    'crl': '#1f77b4',
    'nf': '#ff7f0e',
    'nf_compact': '#8c564b',
    'nf_tiny': '#e377c2',
    'td3': '#2ca02c',
    'fm': '#d62728',
    'tdinfonce': '#9467bd',
}
MODE_LABELS = {
    'crl': 'CRL',
    'nf': 'NF',
    'nf_compact': 'NF compact',
    'nf_tiny': 'NF tiny',
    'td3': 'TD3',
    'fm': 'FM',
    'tdinfonce': 'TDInfoNCE',
}
# Directory name → metrics/probe/ckpt file prefix (train.py uses repr_mode).
MODE_FILE_PREFIX = {
    'nf_compact': 'nf',
    'nf_tiny': 'nf',
}
# (metric_col, title, filename, log_y)
ACC_SPECS = (
    ('train/cat_acc', 'Train categorical accuracy', 'cat_acc_train.png',
     False),
    ('val/cat_acc_az0', 'In-dist val categorical accuracy (a_z=0)',
     'cat_acc_val_az0.png', False),
    ('val/cat_acc_az_ood',
     r'OOD val categorical accuracy ($a_z\sim U[-1,1]$)',
     'cat_acc_val_az_ood.png', True),
    ('val/cat_acc_state_ood',
     'OOD val categorical accuracy (start x,y=ood_high_xy, a_z=0)',
     'cat_acc_val_state_ood.png', True),
    ('val/cat_acc_axy_ood',
     r'OOD val categorical accuracy ($a_x,a_y\sim U[2,4]$, $a_z=0$)',
     'cat_acc_val_axy_ood.png', True),
)
PROBE_SPECS = (
    ('az_ood', "z'", 'ood_az_density'),
    ('state_ood', "x'", 'ood_state_density'),
    ('axy_ood', "x'", 'ood_axy_density'),
)


def _file_prefix(mode: str) -> str:
  return MODE_FILE_PREFIX.get(mode, mode)


def _find_metrics_csv(mode_dir: str, mode: str) -> Optional[str]:
  prefix = _file_prefix(mode)
  matches = sorted(glob.glob(
      os.path.join(mode_dir, f'metrics_{prefix}_seed*.csv')))
  return matches[-1] if matches else None


def _load_csv(path: str) -> Dict[str, np.ndarray]:
  with open(path, 'r', newline='') as f:
    rows = list(csv.DictReader(f))
  if not rows:
    return {}
  cols = rows[0].keys()
  out: Dict[str, np.ndarray] = {}
  for c in cols:
    vals = []
    for r in rows:
      v = r.get(c, '')
      if v is None or v == '':
        vals.append(np.nan)
      else:
        try:
          vals.append(float(v))
        except ValueError:
          vals.append(np.nan)
    out[c] = np.asarray(vals, dtype=np.float64)
  return out


def load_mode_csvs(log_root: str) -> Dict[str, Dict[str, np.ndarray]]:
  out = {}
  for mode in MODES:
    mode_dir = os.path.join(log_root, mode)
    path = _find_metrics_csv(mode_dir, mode)
    if path is None:
      path = _find_metrics_csv(log_root, mode)
    if path is None:
      print(f'[plot] missing metrics for {mode} under {log_root}')
      continue
    data = _load_csv(path)
    out[mode] = data
    n = len(next(iter(data.values()))) if data else 0
    print(f'[plot] loaded {mode}: {path}  ({n} rows)')
  return out


def _smooth_1d(y: np.ndarray, window: int = 11) -> np.ndarray:
  """Light centered rolling mean; no-op if the series is short."""
  y = np.asarray(y, dtype=np.float64)
  n = int(y.shape[0])
  if n < 5:
    return y
  w = int(window)
  if w % 2 == 0:
    w += 1
  w = min(w, n if n % 2 == 1 else n - 1)
  if w < 3:
    return y
  kernel = np.ones(w, dtype=np.float64) / float(w)
  # Reflect-pad so endpoints are not shrunk toward zero.
  pad = w // 2
  yp = np.pad(y, (pad, pad), mode='edge')
  return np.convolve(yp, kernel, mode='valid')


def plot_accuracy_curves(
    dfs: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
    chance: float,
) -> None:
  os.makedirs(out_dir, exist_ok=True)
  for col, title, fname, log_y in ACC_SPECS:
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    y_floor = max(chance / 10.0, 1e-4) if log_y else None
    for mode, data in dfs.items():
      if col not in data or 'global_step' not in data:
        continue
      x = data['global_step']
      y = data[col]
      mask = np.isfinite(x) & np.isfinite(y)
      if not np.any(mask):
        continue
      xx = x[mask]
      yy = y[mask].astype(np.float64)
      yy = _smooth_1d(yy, window=11)
      if log_y:
        yy = np.maximum(yy, y_floor)
      ax.plot(
          xx, yy,
          label=MODE_LABELS.get(mode, mode.upper()),
          color=MODE_COLORS.get(mode, None),
          linewidth=2.0,
      )
    ax.axhline(
        chance, color='0.45', linestyle='--', linewidth=1.5,
        label=f'chance 1/B={chance:.4f}')
    ax.set_xlabel('env steps')
    ax.set_ylabel('categorical accuracy')
    ax.set_title(title)
    if log_y:
      ax.set_yscale('log')
      ax.set_ylim(y_floor, 1.0)
      ax.grid(True, alpha=0.3, which='both')
    else:
      ax.set_ylim(-0.02, 1.02)
      ax.grid(True, alpha=0.3)
    ax.legend(
        frameon=False, loc='upper left',
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] wrote {out_path}')


def _load_probes(
    mode_dir: str, mode: str, probe_name: str,
) -> List[Tuple[int, dict]]:
  prefix = _file_prefix(mode)
  paths = sorted(glob.glob(os.path.join(
      mode_dir, 'ood_density_probe',
      f'probe_{probe_name}_{prefix}_iter*.npz')))
  # Backward compat: old az-only naming probe_{mode}_iter*.npz
  if not paths and probe_name == 'az_ood':
    paths = sorted(glob.glob(os.path.join(
        mode_dir, 'ood_density_probe', f'probe_{prefix}_iter*.npz')))
  out = []
  for p in paths:
    data = dict(np.load(p, allow_pickle=True))
    out.append((int(data['iteration']), data))
  return out


def plot_ood_density_histograms(
    log_root: str,
    out_dir: str,
    max_panels: int = 6,
) -> None:
  """Histogram-style normalized scores for each fixed OOD probe."""
  os.makedirs(out_dir, exist_ok=True)
  for probe_name, axis_label, file_stem in PROBE_SPECS:
    probes_by_mode: Dict[str, List[Tuple[int, dict]]] = {}
    for mode in MODES:
      mode_dir = os.path.join(log_root, mode)
      probes = _load_probes(mode_dir, mode, probe_name)
      if probes:
        probes_by_mode[mode] = probes
        print(f'[plot] {mode}/{probe_name}: {len(probes)} probes')
    if not probes_by_mode:
      print(f'[plot] no probes for {probe_name}; skipping')
      continue

    all_iters = sorted({it for lst in probes_by_mode.values() for it, _ in lst})
    if len(all_iters) > max_panels:
      idx = np.linspace(0, len(all_iters) - 1, max_panels).round().astype(int)
      chosen = [all_iters[i] for i in idx]
    else:
      chosen = all_iters

    any_data = next(iter(probes_by_mode.values()))[0][1]
    action = np.asarray(any_data['action'])
    mu = np.asarray(any_data['mu'])
    grid = np.asarray(any_data['grid'] if 'grid' in any_data else any_data['z_grid'])
    gt = np.asarray(
        any_data['gt_prob'] if 'gt_prob' in any_data else any_data['gt_prob_z'])

    # Softmax probs can hit exact 0; floor before log-y (matches reprobe_wide).
    y_floor = 1e-6

    def _ylog(y):
      return np.maximum(np.asarray(y, dtype=np.float64), y_floor)

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    if n == 1:
      axes = [axes]
    for ax, it in zip(axes, chosen):
      ax.plot(grid, _ylog(gt), color='k', linestyle='--', linewidth=1.5,
              label='true 1-step N' if it == chosen[0] else None)
      for mode, lst in probes_by_mode.items():
        nearest = min(lst, key=lambda t: abs(t[0] - it))
        data = nearest[1]
        g = np.asarray(data['grid'] if 'grid' in data else data['z_grid'])
        ax.plot(
            g, _ylog(data['prob']),
            color=MODE_COLORS.get(mode),
            linewidth=2.0,
            label=(MODE_LABELS.get(mode, mode.upper())
                   if it == chosen[0] else None),
        )
      ax.set_xlabel(axis_label)
      ax.set_title(f'iter {it}')
      ax.set_yscale('log')
      ax.set_ylim(y_floor, 1.0)
      ax.grid(True, alpha=0.3, which='both')
    axes[0].set_ylabel(rf'normalized $\hat p({axis_label}\mid s,a)$')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f'{probe_name}  s={np.asarray(any_data["state"]).tolist()}  '
        f'a=[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f}]  '
        f'μ=[{mu[0]:.3f},{mu[1]:.3f},{mu[2]:.3f}]',
        fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'{file_stem}_over_time.png')
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f'[plot] wrote {out_path}')

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(grid, _ylog(gt), 'k--', linewidth=1.8, label='true 1-step N')
    for mode, lst in probes_by_mode.items():
      data = lst[-1][1]
      g = np.asarray(data['grid'] if 'grid' in data else data['z_grid'])
      ax.plot(
          g, _ylog(data['prob']),
          color=MODE_COLORS.get(mode),
          linewidth=2.2,
          label=(f'{MODE_LABELS.get(mode, mode.upper())} '
                 f'(iter {int(data["iteration"])})'),
      )
    ax.set_xlabel(axis_label)
    ax.set_ylabel(rf'normalized $\hat p({axis_label}\mid s,a)$')
    ax.set_title(f'Latest {probe_name} density probe')
    ax.set_yscale('log')
    ax.set_ylim(y_floor, 1.0)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'{file_stem}_latest.png')
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f'[plot] wrote {out_path}')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--log_root', type=str,
                 default='logs/xyz_density_az0_g05_100m')
  p.add_argument('--out_dir', type=str,
                 default='figs/xyz_density_az0_g05_100m')
  p.add_argument('--batch_size', type=int, default=256,
                 help='Val batch size B (for chance line 1/B).')
  p.add_argument('--max_probe_panels', type=int, default=6)
  args = p.parse_args()

  dfs = load_mode_csvs(args.log_root)
  chance = 1.0 / float(args.batch_size)
  if dfs:
    plot_accuracy_curves(dfs, args.out_dir, chance=chance)
  plot_ood_density_histograms(
      args.log_root, args.out_dir, max_panels=int(args.max_probe_panels))
  print(f'[plot] done → {args.out_dir}')


if __name__ == '__main__':
  main()
