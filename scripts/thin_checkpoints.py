#!/usr/bin/env python3
"""Thin checkpoint milestones so a run's checkpoints/ dir is below a size budget.

Always keeps ckpt_iter_0000000 (if present), the highest-iteration milestone,
and latest.pkl.  Deletes other milestones on a computed stride to fit budget.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

REPO = '/n/fs/mislresearch/sgcrl/logs'
CKPT_RE = re.compile(r'^ckpt_iter_(\d+)\.pkl$')


def iter_from_name(name: str) -> int:
    m = CKPT_RE.match(name)
    return int(m.group(1)) if m else -1


def dir_pkl_bytes(ckpt_dir: str) -> int:
    total = 0
    for name in os.listdir(ckpt_dir):
        if name.endswith('.pkl'):
            total += os.path.getsize(os.path.join(ckpt_dir, name))
    return total


def list_milestones(ckpt_dir: str) -> List[Tuple[int, str, int]]:
    out = []
    for name in os.listdir(ckpt_dir):
        it = iter_from_name(name)
        if it < 0:
            continue
        path = os.path.join(ckpt_dir, name)
        out.append((it, name, os.path.getsize(path)))
    out.sort(key=lambda x: x[0])
    return out


def choose_keep(milestones: List[Tuple[int, str, int]], max_bytes: int) -> List[str]:
    if not milestones:
        return []
    total = sum(sz for _, _, sz in milestones)
    if total <= max_bytes:
        return [name for _, name, _ in milestones]

    names = [name for _, name, _ in milestones]
    sizes = {name: sz for _, name, sz in milestones}
    keep = {names[0], names[-1]}

    # Increase stride until kept set fits budget (with small margin).
    n = len(names)
    for stride in range(2, n + 1):
        trial = set(keep)
        for i in range(0, n, stride):
            trial.add(names[i])
        kept_size = sum(sizes[nm] for nm in trial)
        if kept_size <= max_bytes * 0.98:
            return sorted(trial, key=iter_from_name)

    return names  # fallback: keep all


def thin_dir(ckpt_dir: str, max_bytes: int, dry_run: bool) -> Tuple[int, int, int]:
    milestones = list_milestones(ckpt_dir)
    if not milestones:
        return 0, 0, 0

    latest_sz = 0
    latest_path = os.path.join(ckpt_dir, 'latest.pkl')
    if os.path.isfile(latest_path):
        latest_sz = os.path.getsize(latest_path)
    budget = max(0, max_bytes - latest_sz)
    before = sum(sz for _, _, sz in milestones) + latest_sz
    keep_names = set(choose_keep(milestones, budget))
    deleted = 0
    freed = 0
    for _, name, sz in milestones:
        if name in keep_names:
            continue
        path = os.path.join(ckpt_dir, name)
        if dry_run:
            deleted += 1
            freed += sz
        else:
            os.remove(path)
            deleted += 1
            freed += sz

    after = before - freed
  # Re-measure (latest.pkl always kept).
    if not dry_run:
        after = dir_pkl_bytes(ckpt_dir)
    return deleted, before, after


def find_large_checkpoint_dirs(root: str, min_bytes: int) -> List[Tuple[int, str]]:
    hits = []
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) != 'checkpoints':
            continue
        total = dir_pkl_bytes(dirpath)
        if total > min_bytes:
            hits.append((total, dirpath))
    hits.sort(reverse=True)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=REPO)
    ap.add_argument('--max-gb', type=float, default=1.0)
    ap.add_argument('--min-gb', type=float, default=1.0,
                    help='Only thin dirs larger than this (GiB).')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    max_bytes = int(args.max_gb * (1024 ** 3))
    min_bytes = int(args.min_gb * (1024 ** 3))

    targets = find_large_checkpoint_dirs(args.root, min_bytes)
    if not targets:
        print('No checkpoint dirs above threshold.')
        return

    print(f'Found {len(targets)} checkpoint dirs > {args.min_gb} GiB')
    total_deleted = 0
    for total, ckpt_dir in targets:
        deleted, before, after = thin_dir(ckpt_dir, max_bytes, args.dry_run)
        if deleted == 0:
            continue
        run = os.path.basename(os.path.dirname(ckpt_dir))
        parent = os.path.basename(os.path.dirname(os.path.dirname(ckpt_dir)))
        mode = 'would delete' if args.dry_run else 'deleted'
        print(
            f'{mode} {deleted:4d} ckpts  '
            f'{before/1e9:.2f}G -> {after/1e9:.2f}G  '
            f'{parent}/{run}'
        )
        total_deleted += deleted

    print(f'\n{"Would delete" if args.dry_run else "Deleted"} {total_deleted} checkpoint files total.')


if __name__ == '__main__':
    main()
