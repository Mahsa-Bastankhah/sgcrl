"""Locate rendered BuilderBench PPO-CRL rollout videos (or missing ones).

Recomputes the same SAFE_NAME hash used by mila/scratch_mila.sh (and
mila/render_videos.sh) to find each run's directory on disk, then prints the
absolute paths of rendered videos under <run_dir>/videos/ -- one per line,
ready to pipe into scp / rsync / xargs. With --missing, prints checkpoint
paths that don't have a corresponding rendered video yet.

IMPORTANT: Keep the "KEEP IN SYNC" block below identical to the
corresponding values in mila/scratch_mila.sh (and mila/render_videos.sh).
SAFE_NAME is an md5 hash of these exact strings, so any drift there will
make this script look in the wrong (or a nonexistent) run directory.

Usage:
  python mila/locate_videos.py
  python mila/locate_videos.py --exp_name reproduce
  python mila/locate_videos.py --exp_name reproduce,reproduce_longer_rollout \
      --env builderbench_creative_4_task1 --seed 0
  python mila/locate_videos.py --missing --verbose
  python mila/locate_videos.py --exp_name reproduce \
      | xargs -I{} scp mila:{} ./local_videos/

  # --filters key=value syntax (mirrors the wandb plotting script):
  python mila/locate_videos.py --filters exp_name=reproduce
  python mila/locate_videos.py --filters exp_name=reproduce,reproduce_longer_rollout \
      env=builderbench_creative_4_task1 seed=0,1
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

# ---------- KEEP IN SYNC WITH mila/scratch_mila.sh ---------------------------
SEEDS: List[int] = [0, 1]

LOG_ROOT = "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"

BASE_FLAGS = (
    '--num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 '
    '--ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 '
    '--ppo_checkpoint_interval=150 --builderbench_use_pd=true '
    '--builderbench_pd_duration=5 --ppo_skip_first_eval=true '
    '--ppo_eval_interval=0 --max_replay_size=10000000 '
    '--ppo_crl_repr_tau=0 --hidden_layer_sizes="256,256,256,256,256,256" '
    '--env=builderbench_creative_3_task1'
)

# Each entry: "exp_name|exp_flags", identical to scratch_mila.sh's EXPERIMENTS.
EXPERIMENTS: List[str] = [
    "reproduce|--env=builderbench_creative_4_task1",
    "reproduce_longer_rollout|--env=builderbench_creative_4_task1 --ppo_rollout_length=128",
    "reproduce_longer_rollout|--env=builderbench_creative_4_task1 --ppo_rollout_length=256",
]
# ------------------------------------------------------------------------------

_DEFAULT_ENV_ID = "builderbench_creative_3_task1"
_ENV_RE = re.compile(r"--env=(\S+)")
_CKPT_ITER_RE = re.compile(r"^ckpt_iter_(\d+)\.pkl$")


_FILTER_KEYS = ("exp_name", "env", "seed")


def _parse_list_filter(value: Optional[str]) -> Optional[List[str]]:
  """Comma-separated CLI filter -> list of strings, or None for wildcard."""
  if value is None or value == "":
    return None
  return [v.strip() for v in value.split(",") if v.strip()]


def parse_cli_filters(cli_filters: Optional[List[str]]) -> Dict[str, List[str]]:
  """Parse ``--filters key=v1,v2 key2=v3`` into ``{key: [v1, v2], ...}``.

  Mirrors the plotting script's ``parse_cli_filters``. Only keys in
  ``_FILTER_KEYS`` (exp_name, env, seed) have any effect; unknown keys are
  parsed but silently ignored so a typo doesn't hard-fail the whole query.
  """
  parsed: Dict[str, List[str]] = {}
  if not cli_filters:
    return parsed
  for item in cli_filters:
    if "=" not in item:
      continue
    key, val_str = item.split("=", 1)
    key = key.strip()
    values = [v.strip() for v in val_str.split(",") if v.strip()]
    if key not in _FILTER_KEYS:
      print(f"[locate_videos] warning: unknown --filters key {key!r} "
            f"(expected one of {_FILTER_KEYS}); ignoring", file=sys.stderr)
      continue
    parsed[key] = values
  return parsed


def _matches(value: str, allowed: Optional[List[str]]) -> bool:
  return allowed is None or value in allowed


def safe_name_for(exp_name: str, exp_flags: str, env_id: str, seed: int) -> str:
  """Reproduce SAFE_NAME exactly as computed by scratch_mila.sh.

  Bash:
    SIG_STR="env=${ENV_ID}_seed=${seed}_base=${BASE_FLAGS}_exp=${EXP_FLAGS}"
    CMD_HASH=$(echo "$SIG_STR" | md5sum | cut -c1-8)
    SAFE_NAME="${EXP_NAME//+/_plus_}_${CMD_HASH}"

  `echo` appends a trailing newline before hashing -- we replicate that
  exactly here, otherwise the md5 hashes (and therefore run dirs) won't
  match what scratch_mila.sh actually created on disk.
  """
  sig_str = f"env={env_id}_seed={seed}_base={BASE_FLAGS}_exp={exp_flags}"
  cmd_hash = hashlib.md5((sig_str + "\n").encode("utf-8")).hexdigest()[:8]
  safe_exp_name = exp_name.replace("+", "_plus_")
  return f"{safe_exp_name}_{cmd_hash}"


def iter_experiments(
    exp_name_filter: Optional[List[str]],
    env_filter: Optional[List[str]],
    seed_filter: Optional[List[str]],
) -> Iterable[Tuple[str, str, str, int, str]]:
  """Yield (exp_name, exp_flags, env_id, seed, safe_name) for matching runs."""
  for experiment in EXPERIMENTS:
    exp_name, _, exp_flags = experiment.partition("|")
    m = _ENV_RE.search(exp_flags)
    env_id = m.group(1) if m else _DEFAULT_ENV_ID

    if not _matches(exp_name, exp_name_filter):
      continue
    if not _matches(env_id, env_filter):
      continue

    for seed in SEEDS:
      if seed_filter is not None and str(seed) not in seed_filter:
        continue
      safe_name = safe_name_for(exp_name, exp_flags, env_id, seed)
      yield exp_name, exp_flags, env_id, seed, safe_name


def run_dir_for(safe_name: str, env_id: str, seed: int) -> str:
  return os.path.join(LOG_ROOT, safe_name, f"ppo_{env_id}_{seed}")


def checkpoint_label(pkl_filename: str) -> Optional[str]:
  """Mirror scripts/ppo_builderbench_rollout_video.py::_enumerate_checkpoints.

  Only ckpt_iter_{N}.pkl and latest.pkl are recognized as renderable
  checkpoints; anything else returns None (rollout_video.py would ignore it
  too, so we shouldn't report it as "missing").
  """
  m = _CKPT_ITER_RE.match(pkl_filename)
  if m:
    return f"iter_{int(m.group(1)):07d}"
  if pkl_filename == "latest.pkl":
    return "latest"
  return None


def list_checkpoint_labels(ckpt_dir: str) -> List[Tuple[str, str]]:
  """Return [(label, checkpoint_path), ...] for recognized checkpoints."""
  out: List[Tuple[str, str]] = []
  if not os.path.isdir(ckpt_dir):
    return out
  for fname in sorted(os.listdir(ckpt_dir)):
    label = checkpoint_label(fname)
    if label is not None:
      out.append((label, os.path.join(ckpt_dir, fname)))
  return out


def video_path_for(video_dir: str, safe_name: str, env_id: str, seed: int,
                    label: str) -> str:
  """Mirror rollout_video.py::_resolve_output_path when --output ends in '/'.

  filename = f'{run_tag}_{label}.mp4', where run_tag is what
  render_videos.sh passes via --run_tag="${SAFE_NAME}_${ENV_ID}_s${seed}".
  """
  run_tag = f"{safe_name}_{env_id}_s{seed}"
  return os.path.join(video_dir, f"{run_tag}_{label}.mp4")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Locate rendered BuilderBench rollout videos (or list "
                  "checkpoints still missing a video).")
  parser.add_argument("--exp_name", default=None,
                      help="Comma-separated EXP_NAME filter (default: all).")
  parser.add_argument("--env", default=None,
                      help="Comma-separated env id filter (default: all).")
  parser.add_argument("--seed", default=None,
                      help="Comma-separated seed filter (default: all).")
  parser.add_argument("--filters", nargs="+", default=None,
                      help="Alternative filter syntax, e.g. "
                          "--filters exp_name=reproduce env=builderbench_creative_4_task1 seed=0,1 . "
                          "Combines with --exp_name/--env/--seed; those take "
                          "precedence over --filters for the same key.")
  parser.add_argument("--missing", action="store_true",
                      help="Print checkpoint paths that do NOT yet have a "
                          "rendered video, instead of listing videos.")
  parser.add_argument("--verbose", action="store_true",
                      help="Print run metadata (to stderr) alongside paths.")
  args = parser.parse_args()

  filters = parse_cli_filters(args.filters)

  exp_name_filter = _parse_list_filter(args.exp_name) or filters.get("exp_name")
  env_filter = _parse_list_filter(args.env) or filters.get("env")
  seed_filter = _parse_list_filter(args.seed) or filters.get("seed")

  n_runs = 0
  n_runs_found = 0
  n_printed = 0

  for exp_name, exp_flags, env_id, seed, safe_name in iter_experiments(
      exp_name_filter, env_filter, seed_filter):
    n_runs += 1
    run_dir = run_dir_for(safe_name, env_id, seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    video_dir = os.path.join(run_dir, "videos")

    if not os.path.isdir(ckpt_dir):
      if args.verbose:
        print(f"[skip] no checkpoints/ dir: {ckpt_dir}", file=sys.stderr)
      continue

    n_runs_found += 1
    if args.verbose:
      print(f"[run] exp={exp_name} env={env_id} seed={seed} "
            f"safe_name={safe_name} run_dir={run_dir}", file=sys.stderr)

    for label, ckpt_path in list_checkpoint_labels(ckpt_dir):
      video_path = video_path_for(video_dir, safe_name, env_id, seed, label)
      video_exists = os.path.isfile(video_path)

      if args.missing:
        if not video_exists:
          print(ckpt_path)
          n_printed += 1
      else:
        if video_exists:
          print(video_path)
          n_printed += 1

  mode = "missing checkpoints" if args.missing else "videos"
  print(f"[locate_videos] matched_experiment_configs={n_runs} "
        f"runs_with_checkpoints={n_runs_found} printed_{mode.replace(' ', '_')}={n_printed}",
        file=sys.stderr)


if __name__ == "__main__":
  main()
