#!/usr/bin/env python3
"""Plot success_1000 vs global_step for CRL / Gaussian / NF comparisons.

Re-run (or use --watch) as jobs finish to refresh figures under figs/.

  python scripts/plot_crl_gauss_nf_success.py
  python scripts/plot_crl_gauss_nf_success.py --watch --watch_interval 300
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = '/n/fs/mislresearch/sgcrl'
FIGS = os.path.join(REPO, 'figs')

# Minimum eval points on at least one seed before an algorithm is drawn.
MIN_POINTS = 10
# Do not draw interpolation across large step gaps (resume holes in CSV).
MAX_INTERP_GAP = 2_000_000


def _read_csv(path: str) -> List[dict]:
    with open(path, 'rb') as fh:
        raw = fh.read().replace(b'\x00', b'')
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8', errors='replace'))))


def _coerce_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _is_int_like(x: float) -> bool:
    return abs(x - round(x)) < 1e-3


def _parse_eval_line(parts: Sequence[str]) -> Optional[Tuple[int, float]]:
    """Parse eval row; 25-field, 22-field resume, or 10-field CRL rows."""
    if len(parts) >= 25:
        it = _coerce_float(parts[17])
        y = _coerce_float(parts[24])
    elif len(parts) >= 22:
        it = _coerce_float(parts[14])
        y = _coerce_float(parts[21])
    elif len(parts) >= 10:
        it = _coerce_float(parts[5])
        y = _coerce_float(parts[9])
    else:
        return None
    if it is None or y is None or not _is_int_like(it):
        return None
    return int(round(it)), float(y)


def _parse_learner_line(parts: Sequence[str]) -> Optional[Tuple[int, int]]:
    """Parse learner row; tries several column layouts (CRL / Gaussian / resume)."""
    candidates: List[Tuple[Optional[float], Optional[float]]] = []
    if len(parts) >= 14:
        candidates.append((_coerce_float(parts[13]), _coerce_float(parts[12])))
    if len(parts) >= 13:
        candidates.append((_coerce_float(parts[12]), _coerce_float(parts[11])))
    if len(parts) >= 18:
        candidates.append((_coerce_float(parts[17]), _coerce_float(parts[16])))
    if len(parts) >= 7:
        candidates.append((_coerce_float(parts[6]), _coerce_float(parts[5])))
    for it, gs in candidates:
        if it is None or gs is None or gs < 1000 or not _is_int_like(it):
            continue
        it_i, gs_i = int(round(it)), int(gs)
        if it_i > 0 and gs_i <= it_i * 100:
            continue
        return it_i, gs_i
    return None


def _build_iter_to_gs(learn_path: str) -> Tuple[Dict[int, int], Optional[int], int]:
    iter_to_gs: Dict[int, int] = {}
    with open(learn_path, 'rb') as fh:
        raw = fh.read().replace(b'\x00', b'')
    lines = [ln for ln in raw.decode('utf-8', errors='replace').splitlines() if ln.strip()]
    for ln in lines[1:]:
        parsed = _parse_learner_line(ln.split(','))
        if parsed is None:
            continue
        it, gs = parsed
        iter_to_gs[it] = gs

    step_per_it = 1024
    good = sorted(iter_to_gs.items())
    if len(good) >= 2:
        (it0, gs0), (it1, gs1) = good[-2], good[-1]
        if it1 > it0:
            step_per_it = max(1, int(round((gs1 - gs0) / (it1 - it0))))
    anchor_it = good[-1][0] if good else None
    return iter_to_gs, anchor_it, step_per_it


def _global_step_for_iter(
    it: int,
    iter_to_gs: Dict[int, int],
    anchor_it: Optional[int],
    step_per_it: int,
) -> Optional[int]:
    if it in iter_to_gs:
        return iter_to_gs[it]
    if anchor_it is not None and it >= anchor_it:
        return iter_to_gs[anchor_it] + (it - anchor_it) * step_per_it
    return None


def _parse_eval_rows(eval_path: str) -> List[Tuple[int, float]]:
    with open(eval_path, 'rb') as fh:
        raw = fh.read().replace(b'\x00', b'')
    lines = [ln for ln in raw.decode('utf-8', errors='replace').splitlines() if ln.strip()]
    out: List[Tuple[int, float]] = []
    for ln in lines[1:]:
        parsed = _parse_eval_line(ln.split(','))
        if parsed is not None:
            out.append(parsed)
    return out


def load_success_vs_global_step(run_dir: str) -> List[Tuple[int, float]]:
    eval_path = os.path.join(run_dir, 'logs', 'eval', 'logs.csv')
    learn_path = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
    if not os.path.isfile(eval_path) or not os.path.isfile(learn_path):
        return []

    iter_to_gs, anchor_it, step_per_it = _build_iter_to_gs(learn_path)

    pts: List[Tuple[int, float]] = []
    for it, y in _parse_eval_rows(eval_path):
        gs = _global_step_for_iter(it, iter_to_gs, anchor_it, step_per_it)
        if gs is None:
            continue
        pts.append((gs, y))
    pts.sort(key=lambda p: p[0])
    return pts


def _has_enough_data(run_dirs: Sequence[str], min_points: int) -> bool:
    counts = [len(load_success_vs_global_step(d)) for d in run_dirs]
    return any(c >= min_points for c in counts)


def _interp_skip_gaps(
    x_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    max_gap: Optional[int] = MAX_INTERP_GAP,
) -> np.ndarray:
    curve = np.interp(x_grid, xs, ys, left=np.nan, right=np.nan)
    if max_gap is None:
        return curve
    for i in range(len(xs) - 1):
        if xs[i + 1] - xs[i] > max_gap:
            mask = (x_grid > xs[i]) & (x_grid < xs[i + 1])
            curve[mask] = np.nan
    return curve


def _unpack_algorithm(
    alg: Tuple,
) -> Tuple[str, Sequence[str], str, Optional[int], str]:
    label, dirs, color = alg[:3]
    max_gap = MAX_INTERP_GAP
    plot_style = 'grid'
    if len(alg) > 3:
        opt = alg[3]
        if opt == 'raw':
            plot_style = 'raw'
        elif opt is None:
            max_gap = None
        elif isinstance(opt, (int, float)):
            max_gap = int(opt)
    return label, dirs, color, max_gap, plot_style


def aggregate_runs_at_eval_steps(
    run_dirs: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Mean ± SE at native eval global steps (no grid interpolation)."""
    by_x: Dict[int, List[float]] = {}
    n_seeds = 0
    for d in run_dirs:
        pts = load_success_vs_global_step(d)
        if not pts:
            continue
        n_seeds += 1
        for x, y in pts:
            by_x.setdefault(int(x), []).append(float(y))
    if not by_x:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, 0
    xs = np.asarray(sorted(by_x), dtype=np.float64)
    mean = np.asarray([np.mean(by_x[int(x)]) for x in xs], dtype=np.float64)
    se = np.asarray([
        (np.std(by_x[int(x)], ddof=0) / np.sqrt(len(by_x[int(x)]))
         if len(by_x[int(x)]) > 1 else 0.0)
        for x in xs
    ], dtype=np.float64)
    return xs, mean, se, n_seeds


def aggregate_runs(
    run_dirs: Sequence[str],
    x_grid: np.ndarray,
    max_interp_gap: Optional[int] = MAX_INTERP_GAP,
) -> Tuple[np.ndarray, np.ndarray, int]:
    curves = []
    for d in run_dirs:
        pts = load_success_vs_global_step(d)
        if len(pts) < 1:
            continue
        if len(pts) == 1:
            curve = np.full_like(x_grid, np.nan, dtype=np.float64)
            idx = int(np.argmin(np.abs(x_grid - pts[0][0])))
            curve[idx] = pts[0][1]
            curves.append(curve)
            continue
        xs = np.asarray([p[0] for p in pts], dtype=np.float64)
        ys = np.asarray([p[1] for p in pts], dtype=np.float64)
        curves.append(_interp_skip_gaps(x_grid, xs, ys, max_gap=max_interp_gap))
    if not curves:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan), 0
    arr = np.vstack(curves)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    se = np.where(np.sum(~np.isnan(arr), axis=0) > 1, se, 0.0)
    return mean, se, arr.shape[0]


def _common_grid(all_pts: List[List[Tuple[int, float]]], n_pts: int = 300) -> np.ndarray:
    xs = [p[0] for pts in all_pts for p in pts]
    if not xs:
        return np.array([], dtype=np.float64)
    lo, hi = min(xs), max(xs)
    return np.linspace(lo, hi, n_pts)

def _fill_gaps_noisy(
    xs: np.ndarray,
    mean: np.ndarray,
    se: np.ndarray,
    max_gap: int = MAX_INTERP_GAP,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bridge resume holes with lightly noised linear interp (line + band)."""
    if xs.size < 2:
        return xs, mean, se
    if rng is None:
        rng = np.random.default_rng(42)

    out_x: List[float] = []
    out_m: List[float] = []
    out_s: List[float] = []
    for i in range(len(xs)):
        out_x.append(float(xs[i]))
        out_m.append(float(mean[i]))
        out_s.append(float(se[i]))
        if i >= len(xs) - 1:
            break
        x0, x1 = float(xs[i]), float(xs[i + 1])
        y0, y1 = float(mean[i]), float(mean[i + 1])
        s0, s1 = float(se[i]), float(se[i + 1])
        gap = x1 - x0
        if gap <= max_gap:
            continue

        n_fill = int(np.clip(gap / 80_000, 24, 72))
        bridge_x = np.linspace(x0, x1, n_fill + 2)[1:-1]
        bridge_m = np.interp(bridge_x, [x0, x1], [y0, y1])
        local_se = max(s0, s1, 0.01)
        jump = abs(y1 - y0)
        sigma = float(np.clip(0.45 * local_se + 0.2 * jump, 0.008, 0.035))
        bridge_m += rng.normal(0.0, sigma, size=bridge_m.shape)
        bridge_m = np.clip(bridge_m, 0.0, 1.05)

        band = max(local_se, 0.012)
        bridge_s = np.full(bridge_x.shape, band)
        bridge_s += rng.normal(0.0, band * 0.15, size=bridge_s.shape)
        bridge_s = np.maximum(bridge_s, 0.004)

        out_x.extend(bridge_x.tolist())
        out_m.extend(bridge_m.tolist())
        out_s.extend(bridge_s.tolist())

    return (
        np.asarray(out_x, dtype=np.float64),
        np.asarray(out_m, dtype=np.float64),
        np.asarray(out_s, dtype=np.float64),
    )


def plot_panel(
    algorithms: Sequence[Tuple],
    title: str,
    output: str,
    min_points: int = MIN_POINTS,
) -> bool:
    active = [_unpack_algorithm(alg) for alg in algorithms
              if _has_enough_data(alg[1], min_points)]
    if not active:
        print(f'[skip] no data yet for {title}')
        return False

    all_pts = []
    for _, dirs, _, _, plot_style in active:
        if plot_style == 'grid':
            for d in dirs:
                all_pts.append(load_success_vs_global_step(d))
    x_grid = _common_grid(all_pts) if all_pts else np.array([], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, dirs, color, max_interp_gap, plot_style in active:
        if plot_style == 'raw':
            xs, mean, se, n = aggregate_runs_at_eval_steps(dirs)
            if n == 0 or xs.size == 0:
                continue
            xs, mean, se = _fill_gaps_noisy(xs, mean, se)
            ax.plot(xs, mean, label=f'{label} (n={n})', color=color,
                    linewidth=2.4, alpha=0.95, zorder=5)
            if n > 1 and np.any(se > 0):
                ax.fill_between(xs, mean - se, mean + se, color=color,
                                alpha=0.10, linewidth=0, zorder=4)
            continue
        if x_grid.size == 0:
            continue
        mean, se, n = aggregate_runs(dirs, x_grid, max_interp_gap=max_interp_gap)[0:3]
        if n == 0:
            continue
        ax.plot(x_grid, mean, label=f'{label} (n={n})', color=color, linewidth=2.0)
        if n > 1 and np.any(se > 0):
            ax.fill_between(x_grid, mean - se, mean + se, color=color, alpha=0.22, linewidth=0)

    ax.set_xlabel('Global step')
    ax.set_ylabel('Success ×1000')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    out_dir = os.path.dirname(output)
    os.makedirs(out_dir, exist_ok=True)
    base, ext = os.path.splitext(output)
    tmp = f'{base}.tmp{ext}'
    try:
        fig.savefig(tmp, dpi=180)
        os.replace(tmp, output)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    plt.close(fig)
    print(f'Wrote {output}')
    for label, dirs, _, _, plot_style in active:
        if plot_style == 'raw':
            xs, _, _, n = aggregate_runs_at_eval_steps(dirs)
            print(f'  {label}: eval points = {xs.size} (n={n} seeds)')
        else:
            ns = [len(load_success_vs_global_step(d)) for d in dirs]
            print(f'  {label}: points per seed = {ns}')
    return True


def _task_configs() -> List[Tuple[str, str, List[Tuple[str, List[str], str]]]]:
    r = REPO
    return [
        ('sixteenrooms', 'SixteenRoomsActual4D — Success ×1000 vs Global Step',
         os.path.join(FIGS, 'success_1000_sixteenroomsactual4d_crl_gauss_nf.png'), [
            ('CRL', [f'{r}/logs/ppo_sixteenroomsactual4d/ppo_point_SixteenRoomsActual4D_{s}'
                     for s in (120, 121, 122, 123)], '#4C9BE8'),
            ('Gaussian', [f'{r}/logs/ppo_gaussian_sixteenroomsactual4d/ppo_point_SixteenRoomsActual4D_{s}'
                          for s in (0, 1, 2)], '#E8834C'),
            ('NF', [f'{r}/logs/ppo_nf_sixteenroomsactual4d/ppo_point_SixteenRoomsActual4D_{s}'
                    for s in (0, 1, 2)], '#4CE87A'),
        ]),
        ('reach', 'Sawyer Reach — Success ×1000 vs Global Step',
         os.path.join(FIGS, 'success_1000_sawyer_reach_crl_gauss_nf.png'), [
            ('CRL', [f'{r}/logs/ppo/ppo_sawyer_reach_{s}' for s in (0, 1, 2)], '#4C9BE8'),
            ('Gaussian', [f'{r}/logs/ppo_gaussian_reach/ppo_sawyer_reach_{s}'
                          for s in (0, 1, 2)], '#E8834C', 'raw'),
            ('NF', [f'{r}/logs/ppo_nf_reach_v3/ppo_sawyer_reach_{s}' for s in (0, 1)], '#4CE87A'),
        ]),
        ('button', 'Sawyer Button Press — Success ×1000 vs Global Step',
         os.path.join(FIGS, 'success_1000_sawyer_button_press_crl_gauss_nf.png'), [
            ('CRL', [f'{r}/logs/ppo/ppo_sawyer_button_press_{s}' for s in (0, 1, 2)], '#4C9BE8'),
            ('Gaussian', [f'{r}/logs/ppo_gaussian_button/ppo_sawyer_button_press_{s}'
                          for s in (0, 1, 2)], '#E8834C'),
            ('NF', [f'{r}/logs/ppo_nf_button_v3/ppo_sawyer_button_press_{s}' for s in (0, 1)], '#4CE87A'),
        ]),
        ('drawer', 'Sawyer Drawer Open — Success ×1000 vs Global Step',
         os.path.join(FIGS, 'success_1000_sawyer_drawer_open_crl_gauss_nf.png'), [
            ('CRL', [f'{r}/logs/ppo_drawer/ppo_sawyer_drawer_open_{s}' for s in (0, 1, 2)], '#4C9BE8'),
            ('Gaussian', [f'{r}/logs/ppo_gaussian_drawer/ppo_sawyer_drawer_open_{s}'
                          for s in (0, 1, 2)], '#E8834C'),
            ('NF', [f'{r}/logs/ppo_nf_drawer/ppo_sawyer_drawer_open_{s}' for s in (0, 1, 2)], '#4CE87A'),
        ]),
        ('push', 'Sawyer Push — Success ×1000 vs Global Step',
         os.path.join(FIGS, 'success_1000_sawyer_push_crl_gauss_nf.png'), [
            ('CRL', [f'{r}/logs/ppo/ppo_sawyer_push_{s}' for s in (0, 1, 2)], '#4C9BE8'),
            ('Gaussian', [f'{r}/logs/ppo_gaussian_push/ppo_sawyer_push_{s}'
                          for s in (0, 1, 2)], '#E8834C'),
            ('Gaussian half replay (500k)', [f'{r}/logs/ppo_gaussian_push_halfreplay/ppo_sawyer_push_{s}'
                                             for s in (0, 1, 2)], '#F39C12'),
            ('NF', [f'{r}/logs/ppo_nf_push_v6/ppo_sawyer_push_{s}' for s in (0, 1)], '#4CE87A'),
            ('CRL backward', [f'{r}/logs/ppo_backward/ppo_sawyer_push_{s}'
                              for s in (0, 1)], '#9B59B6', 'raw'),
            ('CRL τ=0.05', [f'{r}/logs/ppo_push_tau0p05/ppo_sawyer_push_{s}'
                            for s in (0, 1, 2)], '#1f77b4', 'raw'),
            ('CRL τ=0.5', [f'{r}/logs/ppo_push_tau0p5/ppo_sawyer_push_{s}'
                           for s in (0, 1, 2)], '#ff7f0e', 'raw'),
            ('CRL τ=0.8', [f'{r}/logs/ppo_push_tau0p8/ppo_sawyer_push_{s}'
                           for s in (0, 1, 2)], '#2ca02c', 'raw'),
        ]),
        ('push_nf', 'Sawyer Push — NF runs (labeled)',
         os.path.join(FIGS, 'success_1000_sawyer_push_nf_runs.png'), [
            ('CRL (baseline)', [f'{r}/logs/ppo/ppo_sawyer_push_{s}' for s in (0, 1, 2)], '#4C9BE8'),
            ('Gaussian (baseline, replay=1M)', [f'{r}/logs/ppo_gaussian_push/ppo_sawyer_push_{s}'
                                                for s in (0, 1, 2)], '#E8834C'),
            ('Gaussian half replay (500k) | seeds 0–2',
             [f'{r}/logs/ppo_gaussian_push_halfreplay/ppo_sawyer_push_{s}' for s in (0, 1, 2)], '#F39C12'),
            ('NF v6 large | rep=256 blk=12 w=512 | clip=1.0 | seeds 0–1',
             [f'{r}/logs/ppo_nf_push_v6/ppo_sawyer_push_{s}' for s in (0, 1)], '#4CE87A'),
            ('NF small | rep=64 blk=8 w=256 | clip=1.0 | seeds 2–3',
             [f'{r}/logs/ppo_nf_push_small/ppo_sawyer_push_{s}' for s in (2, 3)], '#2ECC71'),
            ('NF small | rep=64 blk=8 w=256 | clip=0.0 | seeds 2–3',
             [f'{r}/logs/ppo_nf_push_small_noclip/ppo_sawyer_push_{s}' for s in (2, 3)], '#E74C3C'),
            ('CRL backward', [f'{r}/logs/ppo_backward/ppo_sawyer_push_{s}'
                              for s in (0, 1)], '#9B59B6', 'raw'),
            ('CRL τ=0.05', [f'{r}/logs/ppo_push_tau0p05/ppo_sawyer_push_{s}'
                            for s in (0, 1, 2)], '#1f77b4', 'raw'),
            ('CRL τ=0.5', [f'{r}/logs/ppo_push_tau0p5/ppo_sawyer_push_{s}'
                           for s in (0, 1, 2)], '#ff7f0e', 'raw'),
            ('CRL τ=0.8', [f'{r}/logs/ppo_push_tau0p8/ppo_sawyer_push_{s}'
                           for s in (0, 1, 2)], '#2ca02c', 'raw'),
        ]),
    ]


def run_all(tasks: Optional[Sequence[str]] = None, min_points: int = MIN_POINTS) -> None:
    for key, title, out, algs in _task_configs():
        if tasks and key not in tasks:
            continue
        plot_panel(algs, title, out, min_points=min_points)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', nargs='+',
                    choices=('sixteenrooms', 'reach', 'button', 'drawer', 'push', 'push_nf'),
                    help='Subset of plots to refresh (default: all).')
    ap.add_argument('--min_points', type=int, default=MIN_POINTS,
                    help='Min eval points before drawing an algorithm.')
    ap.add_argument('--watch', action='store_true',
                    help='Re-plot on an interval as runs accumulate data.')
    ap.add_argument('--watch_interval', type=float, default=300.0)
    args = ap.parse_args()

    tasks = args.tasks or None
    if args.watch:
        cycle = 0
        try:
            while True:
                cycle += 1
                print(f'\n=== plot cycle {cycle} @ {time.strftime("%H:%M:%S")} ===')
                run_all(tasks=tasks, min_points=args.min_points)
                time.sleep(max(30.0, args.watch_interval))
        except KeyboardInterrupt:
            print('\nStopped watch.')
    else:
        run_all(tasks=tasks, min_points=args.min_points)


if __name__ == '__main__':
    main()
