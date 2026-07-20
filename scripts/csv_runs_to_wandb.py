#!/usr/bin/env python3
"""Upload existing Acme-style CSV logs to Weights & Biases.

Training in this repo writes `logs/<...>/<run>/logs/{learner,actor,evaluator}/logs.csv`.
This script creates one wandb run per experiment directory (parent of `logs/`).

Install and auth (once):

  pip install wandb
  wandb login

Example — all kappa-bin variant sweeps:

  python scripts/csv_runs_to_wandb.py \
    --project sgcrl-kappa-bin \
    --group kappa_sweeps \
    --run-glob 'logs/kappa_gamma_sweep_bin*/**/sweep_config.json'

Example — single run directory:

  python scripts/csv_runs_to_wandb.py --project sgcrl \
    --run-dir logs/kappa_gamma_sweep_bin/gkappa_0p7/kappa_sac_sawyer_bin_62

Large logs: use --subsample 5 to log every 5th row per CSV.

Default wandb run names use the folder path under logs/ (see --run-name-style)
so runs are easy to match to disk, e.g.
kappa_gamma_sweep_bin_noclip--gkappa_0p7--kappa_sac_sawyer_bin_62
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

# W&B accepts fairly long names; keep UI usable.
_WANDB_NAME_MAX_LEN = 192

def _flatten_dict(d: dict, parent_key: str = '', sep: str = '/') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _try_float(s: str) -> float | None:
  try:
    v = float(s)
    if math.isnan(v) or math.isinf(v):
      return None
    return v
  except ValueError:
    return None


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
  with path.open(newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
  return fieldnames, rows


def _numeric_row(
    row: dict[str, str],
    fieldnames: list[str],
    *,
    prefix: str,
    step_key: str,
) -> dict[str, float | int]:
  out: dict[str, float | int] = {}
  st = row.get(step_key)
  if st is None or st == "":
    return out
  try:
    out[prefix + "_step"] = int(float(st))
  except ValueError:
    return out
  for k in fieldnames:
    if k == step_key:
      continue
    v = _try_float(row[k])
    if v is None:
      continue
    out[f"{prefix}/{k}"] = v
  return out


def _sanitize_wandb_run_name(raw: str) -> str:
  """Make a valid, readable wandb run name (no slashes; alnum-centric)."""
  s = raw.replace("\\", "/")
  # Use "--" so path segments stay visible (plain "__" would collapse later).
  s = s.replace("/", "--")
  s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
  s = re.sub(r"_+", "_", s).strip("._-")
  if len(s) > _WANDB_NAME_MAX_LEN:
    s = s[: _WANDB_NAME_MAX_LEN - 3] + "..."
  return s or "run"


def _default_wandb_run_name(
    run_dir: Path,
    *,
    style: str,
    name_root: Path,
) -> str:
  """Human-readable default run name from the log directory path."""
  r = run_dir.resolve()
  root = name_root.resolve()
  if style == "basename":
    return _sanitize_wandb_run_name(r.name)

  if style == "path_from_logs":
    logs_anchor = root / "logs"
    if logs_anchor.is_dir():
      try:
        rel = r.relative_to(logs_anchor)
        return _sanitize_wandb_run_name(str(rel))
      except ValueError:
        pass

  # path_rel or path_from_logs fallback
  try:
    rel = r.relative_to(root)
    return _sanitize_wandb_run_name(str(rel))
  except ValueError:
    return _sanitize_wandb_run_name(str(r))


def _upload_one_run(
    run_dir: Path,
    *,
    project: str,
    entity: str | None,
    group: str | None,
    name: str | None,
    run_name_style: str,
    name_root: Path,
    subsample: int,
    dry_run: bool,
) -> None:
  resolved = name or _default_wandb_run_name(
      run_dir, style=run_name_style, name_root=name_root)
  if dry_run:
    print(f"[dry-run] would upload: {run_dir}  ->  wandb name: {resolved}")
    return

  try:
    import wandb
  except ImportError as e:
    raise SystemExit(
        "wandb is not installed. Run: pip install wandb && wandb login"
    ) from e

  cfg: dict = {}
  # Load run_config.json
  cfg_path = run_dir / "run_config.json"
  if cfg_path.is_file():
      with cfg_path.open() as f:
          cfg.update(json.load(f))

  # Load metadata.json (the command)
  meta_path = run_dir / "metadata.json"
  if meta_path.is_file():
      with meta_path.open() as f:
          cfg.update(json.load(f))

  run_name = resolved
  tags = [
      x for x in run_dir.parts
      if x.startswith(("gkappa_", "kappa_gamma_", "kappa_sac_"))
  ][-5:]

  paths = {
      "learner": run_dir / "logs" / "learner" / "logs.csv",
      "actor": run_dir / "logs" / "actor" / "logs.csv",
      "evaluator": run_dir / "logs" / "evaluator" / "logs.csv",
      "checkpoint_eval": run_dir / "logs" / "eval" / "logs.csv",
  }

  flat_cfg = _flatten_dict(cfg)

  init_kwargs: dict = {
      "project": project,
      "name": run_name,
      "config": {
          "run_dir": str(run_dir.resolve()),
          "run_name_style": run_name_style,
          **flat_cfg,
      },
  }
  if entity:
    init_kwargs["entity"] = entity
  if group:
    init_kwargs["group"] = group
  if tags:
    init_kwargs["tags"] = tags

  wandb.init(**init_kwargs)

  # Custom x-axes so learner / actor / evaluator do not share one global step.
  wandb.define_metric("learner/*", step_metric="learner_step")
  wandb.define_metric("actor/*", step_metric="actor_step")
  wandb.define_metric("evaluator/*", step_metric="evaluator_step")
  wandb.define_metric("checkpoint_eval/*", step_metric="checkpoint_eval_step")

  for role, csv_path in paths.items():
    if not csv_path.is_file():
      continue
    fieldnames, rows = _read_rows(csv_path)
    step_col = {
        "learner": "learner_steps",
        "actor": "actor_steps",
        "evaluator": "evaluator_steps",
        "checkpoint_eval": "global_step",
    }[role]
    if step_col not in fieldnames:
      print(f"[warn] {csv_path}: missing column {step_col!r}, skip", file=sys.stderr)
      continue
    prefix = role
    for i, row in enumerate(rows):
      if subsample > 1 and (i % subsample) != 0:
        continue
      payload = _numeric_row(row, fieldnames, prefix=prefix, step_key=step_col)
      if not payload:
        continue
      wandb.log(payload)

  # --- UPLOAD VIDEOS ---
  videos_dir = run_dir / "videos"
  if videos_dir.is_dir():
    mp4_files = list(videos_dir.glob("*.mp4"))
    if mp4_files:
      print(f"Uploading {len(mp4_files)} videos from {videos_dir}...")
      
      # 1. Save directly to the WandB 'Files' tab
      wandb.save(str(videos_dir / "*.mp4"), base_path=str(run_dir), policy="now")
      
      # 2. Log to the media workspace so they are playable directly in UI
      try:
          wandb.log({
              "rollout_videos": [wandb.Video(str(v), format="mp4") for v in mp4_files]
          })
      except Exception as e:
          print(f"[warn] Failed to log videos as media to workspace: {e}", file=sys.stderr)

  wandb.finish()
  print(f"Uploaded {run_dir}")


def _run_dirs_from_globs(patterns: list[str]) -> list[Path]:
  roots: set[Path] = set()
  for pat in patterns:
    for p in sorted(Path().glob(pat)):
      if p.name == "sweep_config.json":
        roots.add(p.parent)
  return sorted(roots)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--project", required=True, help="wandb project name")
  ap.add_argument("--entity", default=None, help="wandb entity (team or user)")
  ap.add_argument("--group", default=None, help="wandb group (e.g. sweep name)")
  ap.add_argument(
      "--run-dir",
      action="append",
      default=[],
      type=Path,
      help="Experiment directory (contains logs/ and sweep_config.json). Repeatable.",
  )
  ap.add_argument(
      "--run-glob",
      action="append",
      default=[],
      help="Glob for sweep_config.json paths; parent dir is treated as one run. Repeatable.",
  )
  ap.add_argument("--name", default=None, help="Override wandb run name")
  ap.add_argument(
      "--name-root",
      type=Path,
      default=Path("."),
      help="Anchor for relative run names (default: current directory).",
  )
  ap.add_argument(
      "--run-name-style",
      choices=("path_from_logs", "path_rel", "basename"),
      default="path_from_logs",
      help="How to build the default wandb run name from the run directory. "
      "path_from_logs: relpath from <name-root>/logs (e.g. sweep/gkappa/run). "
      "path_rel: relpath from --name-root. basename: last folder only.",
  )
  ap.add_argument(
      "--subsample",
      type=int,
      default=1,
      help="Log every Nth row from each CSV (default 1 = all rows).",
  )
  ap.add_argument(
      "--dry-run",
      action="store_true",
      help="Print run dirs that would be uploaded; do not call wandb.",
  )
  args = ap.parse_args()

  run_dirs = [d.resolve() for d in args.run_dir]
  run_dirs += _run_dirs_from_globs(args.run_glob)
  run_dirs = sorted({p for p in run_dirs})

  if not run_dirs:
    raise SystemExit("No runs: pass --run-dir and/or --run-glob")

  name_kw = args.name
  if name_kw is not None and len(run_dirs) > 1:
    print(
        "[warn] --name ignored when uploading multiple runs "
        "(each run uses its directory name).",
        file=sys.stderr,
    )
    name_kw = None

  for rd in run_dirs:
    if not (rd / "logs").is_dir():
      print(f"[skip] not a run dir (no logs/): {rd}", file=sys.stderr)
      continue
    _upload_one_run(
        rd,
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=name_kw,
        run_name_style=args.run_name_style,
        name_root=args.name_root,
        subsample=max(1, args.subsample),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
  main()
