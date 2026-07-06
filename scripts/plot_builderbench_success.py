#!/usr/bin/env python3
"""Plot BuilderBench training success curves, grouped by task.

For each creative/task pair (e.g. creative2 task1), overlays all active e1024
run variants (vanilla, PD, tau, …) on one figure with a legend.

Reads ``train_success_mean`` from learner CSV logs under ``logs/ppo_builderbench_*``.
Writes PNGs to ``figs/builderbench/``.

Usage:
    python scripts/plot_builderbench_success.py
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

LOG_ROOT = "/n/fs/mislresearch/sgcrl/logs"
FIGS_DIR = "/n/fs/mislresearch/sgcrl/figs/builderbench"
METRIC = "train_success_mean"
X_COL = "global_step"

ACCENT_COLORS = [
    "#4C9BE8", "#E8834C", "#4CE87A", "#E84C6F",
    "#A84CE8", "#E8D44C", "#4CE8D4", "#E84CA8",
]

DIR_RE = re.compile(
    r"^ppo_builderbench_(creative(?P<creative>\d+)_task(?P<task>\d+))"
    r"(?P<suffix>.*)$"
)


def _fmt_steps(v, _):
    if v == 0:
        return "0"
    if v >= 1e6:
        s = f"{v/1e6:.1f}M"
        return s.replace(".0M", "M")
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return str(int(v))


def _variant_label(suffix: str) -> str:
    suffix = suffix or ""
    if suffix == "_e1024":
        return "vanilla"
    if suffix == "_e1024_pd":
        return "PD"
    if suffix == "_e1024_tau0p5":
        return "tau=0.5"
    if suffix == "_e1024_tau0p9":
        return "tau=0.9"
    if suffix == "_e1024_pd_tau0p5":
        return "PD + tau=0.5"
    if suffix == "_e1024_pd_tau0p9":
        return "PD + tau=0.9"
    return suffix.lstrip("_") or "default"


def _discover_runs():
    """Return {group_key: [(log_dir_name, label), ...]}."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if not os.path.isdir(LOG_ROOT):
        return groups

    for name in sorted(os.listdir(LOG_ROOT)):
        if not name.startswith("ppo_builderbench_creative"):
            continue
        m = DIR_RE.match(name)
        if not m:
            continue
        suffix = m.group("suffix")
        if "e256" in suffix or "e128" in suffix:
            continue
        if "e1024" not in suffix and suffix != "":
            continue
        if suffix and not suffix.startswith("_e1024"):
            continue

        log_dir = os.path.join(LOG_ROOT, name)
        has_data = False
        for run_name in os.listdir(log_dir):
            csv_path = os.path.join(
                log_dir, run_name, "logs", "learner", "logs.csv")
            if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 200:
                has_data = True
                break
        if not has_data:
            continue

        group_key = f"creative{m.group('creative')}_task{m.group('task')}"
        label = _variant_label(suffix if suffix else "_e1024")
        groups[group_key].append((name, label))

    for key in groups:
        groups[key].sort(key=lambda x: x[1])
    return groups


def _coerce(v):
    try:
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return None


def _read_series(log_dir_name: str) -> list[tuple[int, float]]:
    base = os.path.join(LOG_ROOT, log_dir_name)
    for run_name in sorted(os.listdir(base)):
        path = os.path.join(base, run_name, "logs", "learner", "logs.csv")
        if not os.path.isfile(path):
            continue
        pts: list[tuple[int, float]] = []
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    x = _coerce(row.get(X_COL, ""))
                    y = _coerce(row.get(METRIC, ""))
                    if x is not None and y is not None:
                        pts.append((int(x), y))
        except Exception:
            continue
        if pts:
            return pts
    return []


def _subsample(pts, max_pts=400):
    if len(pts) <= max_pts:
        return pts
    step = max(1, len(pts) // max_pts)
    sampled = pts[::step]
    if pts[-1] not in sampled:
        sampled = sampled + [pts[-1]]
    return sampled


def _plot_group(group_key: str, runs: list[tuple[str, str]]) -> str | None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = 0

    for i, (log_dir_name, label) in enumerate(runs):
        pts = _subsample(_read_series(log_dir_name))
        if not pts:
            print(f"  {group_key}/{label}: no data, skipping")
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    creative, task = group_key.split("_task")
    creative_num = creative.replace("creative", "")
    ax.set_title(
        f"BuilderBench Creative {creative_num} Task {task}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Global Steps", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    out_path = os.path.join(FIGS_DIR, f"{group_key}_success.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    groups = _discover_runs()
    if not groups:
        print("No BuilderBench e1024 runs with data found.")
        return

    print(f"Found {len(groups)} task group(s)")
    for group_key, runs in sorted(groups.items()):
        labels = ", ".join(l for _, l in runs)
        print(f"  {group_key}: {labels}")
        out = _plot_group(group_key, runs)
        if out:
            print(f"    → {out}")


if __name__ == "__main__":
    main()
