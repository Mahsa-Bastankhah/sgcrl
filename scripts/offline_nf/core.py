"""Domain-independent sampling, evaluation, training, and plots for offline NFs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


def make_train_parser(description: str) -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--dataset_dir", required=True)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--dataset_sizes", default="500000,2000000,8000000")
  parser.add_argument("--updates", type=int, default=5000)
  parser.add_argument("--eval_every", type=int, default=250)
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--eval_nll_samples", type=int, default=2048)
  parser.add_argument("--binary_samples", type=int, default=2048)
  parser.add_argument("--categorical_samples", type=int, default=512)
  parser.add_argument("--categorical_k", type=int, default=32)
  parser.add_argument("--goal_stat_samples", type=int, default=262144)
  parser.add_argument("--discount", type=float, default=0.99)
  parser.add_argument(
      "--max_episode_len", type=int, default=None,
      help="keep only the first N actions / N+1 observations (complete prefixes)")
  parser.add_argument(
      "--horizon_truncate", type=int, default=None,
      help="alias of --max_episode_len")
  parser.add_argument(
      "--rescore_checkpoint", default="",
      help="score an existing NF pickle on the (possibly truncated) fixture, no updates")
  parser.add_argument(
      "--compare_t150_dir", default="",
      help="existing Allegro T=150 sweep dir for the horizon comparison table")
  parser.add_argument(
      "--compare_bb_dir", default="",
      help="existing BuilderBench c3t1 8M sweep dir for the comparison table")
  parser.add_argument(
      "--l2_only", action="store_true",
      help="build the fixture and write L2 rank metrics, then exit (CPU, no JAX)")
  parser.add_argument(
      "--smoke_test", action="store_true",
      help="run one real compact-small NF update with small fixtures")
  return parser


@dataclass(frozen=True)
class EpisodeSet:
  """Episode-major observations/actions with geometric future sampling."""

  obs: np.ndarray
  action: np.ndarray
  episode_ids: np.ndarray
  discount: float
  goal_indices: np.ndarray

  def sample(self, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    episode_ids = rng.choice(self.episode_ids, size=int(count), replace=True)
    horizon = int(self.action.shape[1])
    t = rng.integers(0, horizon, size=int(count))
    max_delta = horizon - t
    trunc_cdf = 1.0 - np.power(
        float(self.discount), max_delta.astype(np.float64))
    u = rng.random(int(count)) * trunc_cdf
    delta = 1 + np.floor(
        np.log1p(-u) / np.log(float(self.discount))).astype(np.int64)
    delta = np.clip(delta, 1, max_delta)
    future = t + delta
    states = np.asarray(self.obs[episode_ids, t], dtype=np.float32)
    next_states = np.asarray(self.obs[episode_ids, t + 1], dtype=np.float32)
    goals = np.asarray(
        self.obs[episode_ids, future][:, self.goal_indices], dtype=np.float32)
    actions = np.asarray(self.action[episode_ids, t], dtype=np.float32)
    return {
        "obs": np.concatenate([states, goals], axis=-1),
        "action": actions,
        "next_obs": np.concatenate([next_states, goals], axis=-1),
        "_episode_id": episode_ids,
        "_t": t.astype(np.int64),
        "_delta": delta.astype(np.int64),
    }


DELTA_BINS = (
    ("1", 1, 1),
    ("2_5", 2, 5),
    ("6_12", 6, 12),
    ("13_20", 13, 20),
    ("21_50", 21, 50),
)


def resolve_horizon_truncate(args) -> int | None:
  """Return the prefix length, or None to keep the stored episode length."""
  left = args.max_episode_len
  right = args.horizon_truncate
  if left is not None and right is not None and int(left) != int(right):
    raise ValueError(
        f"--max_episode_len={left} disagrees with --horizon_truncate={right}")
  value = left if left is not None else right
  return None if value is None else int(value)


def write_json(path: str, value: dict[str, Any]) -> None:
  tmp = path + ".tmp"
  with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
  os.replace(tmp, path)


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
  tmp = path + ".tmp"
  with open(tmp, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  os.replace(tmp, path)


def read_csv(path: str) -> list[dict[str, Any]]:
  with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
  numeric = {
      "step", "dataset_size_label", "train_transitions", "train_nf_loss",
      "train_nll", "val_nll", "train_crl_loss", "binary_accuracy",
      "categorical_accuracy", "categorical_chance", "task_goal_logp",
      "shuffled_context_logp", "positive_score", "negative_score",
      "task_goal_score", "shuffled_context_score",
  }
  for row in rows:
    for key in numeric.intersection(row):
      row[key] = float(row[key])
  return rows


def _load_pyplot():
  try:
    import matplotlib
  except ModuleNotFoundError:
    fallback = (
        "/n/fs/mislresearch/miniconda3/envs/dbc/lib/python3.8/site-packages")
    if not os.path.isdir(os.path.join(fallback, "matplotlib")):
      raise
    sys.path.append(fallback)
    import matplotlib
    print(f"[plot] using matplotlib fallback from {fallback}", flush=True)
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  return plt


def make_eval_fixture(
    val: EpisodeSet,
    *,
    binary_count: int,
    categorical_count: int,
    categorical_k: int,
    seed: int,
) -> dict[str, np.ndarray]:
  """Create fixed held-out contexts and cross-episode candidate goals."""
  rng = np.random.default_rng(seed)
  total = max(int(binary_count), int(categorical_count))
  positive = val.sample(total, rng)
  state_dim = val.obs.shape[-1]
  states = positive["obs"][:, :state_dim]
  actions = positive["action"]
  pos_goals = positive["obs"][:, state_dim:]
  anchor_eps = positive["_episode_id"]

  candidates = np.empty(
      (int(categorical_count), int(categorical_k), len(val.goal_indices)),
      dtype=np.float32)
  candidates[:, 0] = pos_goals[:categorical_count]
  for row in range(int(categorical_count)):
    pool = val.episode_ids[val.episode_ids != anchor_eps[row]]
    negative_eps = rng.choice(
        pool, size=int(categorical_k) - 1,
        replace=len(pool) < int(categorical_k) - 1)
    negative_t = rng.integers(
        0, val.obs.shape[1], size=int(categorical_k) - 1)
    candidates[row, 1:] = val.obs[
        negative_eps, negative_t][:, val.goal_indices]

  binary_neg = np.empty(
      (int(binary_count), len(val.goal_indices)), dtype=np.float32)
  for row in range(int(binary_count)):
    pool = val.episode_ids[val.episode_ids != anchor_eps[row]]
    ep = int(rng.choice(pool))
    t = int(rng.integers(0, val.obs.shape[1]))
    binary_neg[row] = val.obs[ep, t, val.goal_indices]
  return {
      "states": states.astype(np.float32),
      "actions": actions.astype(np.float32),
      "positive_goals": pos_goals.astype(np.float32),
      "binary_negative_goals": binary_neg,
      "categorical_candidates": candidates,
      "positive_deltas": positive["_delta"].astype(np.int64),
      "goal_indices": np.asarray(val.goal_indices, dtype=np.int64),
  }


def _current_achieved(fixture: dict[str, np.ndarray]) -> np.ndarray:
  goal_dim = int(fixture["positive_goals"].shape[-1])
  if "goal_indices" in fixture:
    return np.asarray(fixture["states"][:, np.asarray(fixture["goal_indices"])])
  return np.asarray(fixture["states"][:, :goal_dim])


def _bin_mask(deltas: np.ndarray, lo: int, hi: int) -> np.ndarray:
  return (deltas >= lo) & (deltas <= hi)


def stratify_by_delta(
    deltas: np.ndarray, correct: np.ndarray,
) -> dict[str, float | int]:
  out: dict[str, float | int] = {}
  for name, lo, hi in DELTA_BINS:
    mask = _bin_mask(np.asarray(deltas), lo, hi)
    key = f"delta_{name}"
    out[f"{key}_n"] = int(mask.sum())
    out[f"{key}_acc"] = (
        float(np.mean(correct[mask])) if int(mask.sum()) else None)
  return out


def l2_rank_metrics(fixture: dict[str, np.ndarray]) -> dict[str, Any]:
  """Pick the candidate closest in L2 to the current achieved goal."""
  current = _current_achieved(fixture)
  pos = np.asarray(fixture["positive_goals"])
  neg = np.asarray(fixture["binary_negative_goals"])
  n_bin = int(neg.shape[0])
  dpos = np.linalg.norm(pos[:n_bin] - current[:n_bin], axis=-1)
  dneg = np.linalg.norm(neg - current[:n_bin], axis=-1)
  binary_correct = dpos < dneg
  cats = np.asarray(fixture["categorical_candidates"])
  n_cat = int(cats.shape[0])
  dcat = np.linalg.norm(cats - current[:n_cat, None, :], axis=-1)
  cat_correct = np.argmin(dcat, axis=1) == 0
  metrics: dict[str, Any] = {
      "scorer": "l2_to_current_achieved",
      "binary_accuracy": float(np.mean(binary_correct)),
      "categorical_accuracy": float(np.mean(cat_correct)),
      "categorical_chance": 1.0 / float(cats.shape[1]),
      "binary_n": n_bin,
      "categorical_n": n_cat,
      "mean_l2_positive": float(dpos.mean()),
      "mean_l2_negative": float(dneg.mean()),
  }
  deltas = fixture.get("positive_deltas")
  if deltas is not None:
    metrics["binary_by_delta"] = stratify_by_delta(
        np.asarray(deltas)[:n_bin], binary_correct)
    metrics["categorical_by_delta"] = stratify_by_delta(
        np.asarray(deltas)[:n_cat], cat_correct)
  return metrics


def load_episode_arrays(
    dataset_dir: str, expected_dims: tuple[int, int, int],
    *, horizon_truncate: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int, int]:
  """Load episode-major arrays and optionally keep only a complete prefix."""
  with open(os.path.join(dataset_dir, "metadata.json"), encoding="utf-8") as handle:
    metadata = json.load(handle)
  obs = np.load(
      os.path.join(dataset_dir, metadata["observations_file"]), mmap_mode="r")
  action = np.load(
      os.path.join(dataset_dir, metadata["actions_file"]), mmap_mode="r")
  num_episodes, source_horizon = int(action.shape[0]), int(action.shape[1])
  state_dim, action_dim, goal_dim = expected_dims
  if obs.shape != (num_episodes, source_horizon + 1, state_dim):
    raise ValueError(f"unexpected observation shape {obs.shape}")
  if action.shape[2] != action_dim:
    raise ValueError(f"unexpected action shape {action.shape}")
  if horizon_truncate is not None:
    if horizon_truncate < 1 or horizon_truncate > source_horizon:
      raise ValueError(
          f"horizon_truncate={horizon_truncate} not in [1, {source_horizon}]")
    action = action[:, :horizon_truncate]
    obs = obs[:, : horizon_truncate + 1]
    print(
        f"[horizon] using complete prefixes T={horizon_truncate} "
        f"(source T={source_horizon}, episodes={num_episodes}, "
        f"obs={tuple(obs.shape)} action={tuple(action.shape)})",
        flush=True)
  horizon = int(action.shape[1])
  if len(np.asarray(metadata["achieved_goal_state_indices"])) != goal_dim:
    raise ValueError(
        f"unexpected goal indices {metadata['achieved_goal_state_indices']}")
  return obs, action, metadata, source_horizon, horizon


def goal_stats(
    train: EpisodeSet,
    *,
    sample_count: int,
    seed: int,
    std_min: float,
    task_goal: np.ndarray | None = None,
    task_goal_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
  """Compute replay-goal moments, optionally preserving legacy task-goal mix."""
  rng = np.random.default_rng(seed)
  replay_goals = train.sample(int(sample_count), rng)["obs"][
      :, train.obs.shape[-1]:]
  replay_mean = replay_goals.mean(axis=0, dtype=np.float64)
  replay_second = np.square(replay_goals, dtype=np.float64).mean(axis=0)
  frac = float(task_goal_fraction)
  if frac:
    if task_goal is None or not 0.0 <= frac < 1.0:
      raise ValueError("task-goal mixture needs a goal and fraction in [0,1)")
    mean = (1.0 - frac) * replay_mean + frac * task_goal
    second = (
        (1.0 - frac) * replay_second
        + frac * np.square(task_goal, dtype=np.float64))
  else:
    mean, second = replay_mean, replay_second
  std_raw = np.sqrt(np.maximum(second - np.square(mean), 0.0))
  std = np.maximum(std_raw, float(std_min))
  details = {
      "replay_future_samples": int(sample_count),
      "replay_mean": replay_mean.tolist(),
      "task_goal_fraction": frac,
      "source": "replay-only train goals" if frac == 0 else "replay/task mix",
      "std_before_floor": std_raw.tolist(),
      "std_floor": float(std_min),
  }
  return mean.astype(np.float32), std.astype(np.float32), details


def _annotate_early_steps(axes, batch_size: int) -> None:
  for axis in axes:
    for step, label in (
        (0, "0\npre-train"),
        (250, f"250\n{250 * batch_size:,} draws"),
        (500, f"500\n{500 * batch_size:,} draws"),
    ):
      axis.axvline(step, color="0.75", lw=0.8, ls=":" if step else "--")
      axis.text(
          step, 1.01, label, transform=axis.get_xaxis_transform(),
          ha="left" if step == 0 else "center", va="bottom", fontsize=7)


def plot_learning(
    path: str,
    rows: list[dict[str, Any]],
    label: str,
    *,
    domain_label: str,
    batch_size: int = 256,
    early_max: int | None = None,
) -> None:
  """Plot all requested metrics, with a linear-x dedicated early view."""
  plt = _load_pyplot()
  selected = rows
  suffix = ""
  if early_max is not None:
    selected = [row for row in rows if float(row["step"]) <= early_max]
    suffix = f" — early 0–{early_max} updates"
  x = np.asarray([row["step"] for row in selected], dtype=float)
  fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
  axes[0, 0].plot(
      x, [row["train_nll"] for row in selected], marker="o",
      label="fixed-fixture train NLL")
  axes[0, 0].plot(
      x, [row["train_nf_loss"] for row in selected], marker="s", ls="--",
      label="interval train_nf_loss")
  axes[0, 0].plot(
      x, [row["val_nll"] for row in selected], marker="^",
      label="held-out val NLL")
  axes[0, 0].set_ylabel("NLL / loss")
  axes[0, 0].legend(fontsize=8)
  axes[0, 1].plot(
      x, [row["binary_accuracy"] for row in selected], marker="o",
      label="held-out binary")
  axes[0, 1].axhline(0.5, color="0.5", ls="--", label="chance = 50%")
  axes[0, 1].set_ylabel("binary rank accuracy")
  axes[0, 1].set_ylim(0, 1)
  axes[0, 1].legend(fontsize=8)
  axes[1, 0].plot(
      x, [row["categorical_accuracy"] for row in selected], marker="o",
      label="held-out K=32")
  chance = float(selected[0]["categorical_chance"])
  axes[1, 0].axhline(
      chance, color="0.5", ls="--", label=f"chance = {100 * chance:g}%")
  axes[1, 0].set_ylabel("K=32 categorical accuracy")
  axes[1, 0].set_ylim(0, 1)
  axes[1, 0].legend(fontsize=8)
  axes[1, 1].plot(
      x, [row["task_goal_logp"] for row in selected], marker="o",
      label="task-goal log p")
  axes[1, 1].plot(
      x, [row["shuffled_context_logp"] for row in selected], marker="s",
      label="shuffled-context control")
  axes[1, 1].set_ylabel("mean log p")
  axes[1, 1].legend(fontsize=8)
  for axis in axes.flat:
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("gradient updates")
  if early_max is not None:
    _annotate_early_steps(axes.flat, batch_size)
    axes[1, 0].set_xlim(0, early_max)
  fig.suptitle(f"Offline {domain_label} NF learning — {label}{suffix}")
  fig.text(
      0.5, 0.005,
      "Step 0 = pre-training; first post-train evaluation: "
      f"250 updates = {250 * batch_size:,} replay draws "
      f"(batch {batch_size}).",
      ha="center", fontsize=9)
  fig.tight_layout(rect=(0, 0.035, 1, 1))
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def plot_scaling(
    path: str, final_rows: list[dict[str, Any]], *, domain_label: str,
) -> None:
  plt = _load_pyplot()
  x = np.asarray(
      [row["train_transitions"] for row in final_rows], dtype=float)
  fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
  axes[0].plot(x, [row["val_nll"] for row in final_rows], marker="o")
  axes[0].set_ylabel("final held-out NLL")
  axes[1].plot(
      x, [row["binary_accuracy"] for row in final_rows], marker="o")
  axes[1].axhline(0.5, color="0.5", ls="--")
  axes[1].set_ylabel("final binary accuracy")
  axes[1].set_ylim(0, 1)
  axes[2].plot(
      x, [row["categorical_accuracy"] for row in final_rows], marker="o")
  axes[2].axhline(
      float(final_rows[0]["categorical_chance"]), color="0.5", ls="--")
  axes[2].set_ylabel("final categorical accuracy")
  axes[2].set_ylim(0, 1)
  for axis in axes:
    axis.set_xscale("log")
    axis.set_xlabel("actual train transitions")
    axis.grid(axis="y", alpha=0.25)
  fig.suptitle(f"Offline {domain_label} NF: final dataset-size comparison")
  fig.tight_layout()
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def plot_delta_stratified(
    path: str, rows: list[dict[str, Any]], *, domain_label: str, label: str,
) -> None:
  """Bar chart of binary/categorical accuracy by future Δ on the T-truncated fixture."""
  plt = _load_pyplot()
  last = rows[-1]
  names = [name for name, _, _ in DELTA_BINS]
  x = np.arange(len(names))

  def _acc(key: str) -> float:
    value = last.get(key)
    return float("nan") if value is None else float(value)

  bin_acc = [_acc(f"binary_accuracy_delta_{name}") for name in names]
  cat_acc = [_acc(f"categorical_accuracy_delta_{name}") for name in names]
  counts = [last.get(f"binary_n_delta_{name}", 0) for name in names]
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
  axes[0].bar(x, bin_acc, color="#4C9BE8")
  axes[0].axhline(0.5, color="0.5", ls="--", label="chance 50%")
  axes[0].set_title("binary")
  axes[1].bar(x, cat_acc, color="#E07A3D")
  axes[1].axhline(
      float(last["categorical_chance"]), color="0.5", ls="--",
      label=f"chance {100 * float(last['categorical_chance']):g}%")
  axes[1].set_title("K=32 categorical")
  for axis in axes:
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{name.replace('_', '–')}\nn={n}" for name, n in zip(names, counts)],
        fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
  fig.suptitle(
      f"{domain_label} accuracy by Δ — {label} (final step {int(last['step'])})")
  fig.tight_layout()
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def plot_horizon_comparison(path: str, rows: list[dict[str, Any]]) -> None:
  plt = _load_pyplot()
  labels = [row["label"] for row in rows]
  x = np.arange(len(labels))
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
  axes[0].bar(x, [row["binary_accuracy"] for row in rows], color="#4C9BE8")
  axes[0].axhline(0.5, color="0.5", ls="--")
  axes[0].set_title("binary accuracy")
  axes[1].bar(x, [row["categorical_accuracy"] for row in rows], color="#E07A3D")
  axes[1].axhline(1.0 / 32.0, color="0.5", ls="--")
  axes[1].set_title("K=32 categorical accuracy")
  for axis in axes:
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
  fig.suptitle(
      "Horizon test: Allegro T=50 prefixes vs Allegro T=150 vs BB c3t1")
  fig.tight_layout()
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def _final_8m_row(metrics_dir: str) -> dict[str, Any] | None:
  path = os.path.join(metrics_dir, "final_size_comparison.csv")
  if not os.path.isfile(path):
    return None
  rows = read_csv(path)
  eights = [row for row in rows if int(row["dataset_size_label"]) == 8_000_000]
  return eights[-1] if eights else (rows[-1] if rows else None)


def write_horizon_comparison(
    output_dir: str,
    *,
    t50_l2: dict[str, Any],
    t50_nf: dict[str, Any] | None,
    t150_nf_on_t50: dict[str, Any] | None,
    compare_t150_dir: str,
    compare_bb_dir: str,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []

  def add(label: str, binary: float, categorical: float, **extra: Any) -> None:
    rows.append({
        "label": label,
        "binary_accuracy": float(binary),
        "categorical_accuracy": float(categorical),
        **extra,
    })

  t150_fixture = os.path.join(
      compare_t150_dir, "fixed_heldout_eval_fixture.npz") if compare_t150_dir else ""
  if t150_fixture and os.path.isfile(t150_fixture):
    t150_l2 = l2_rank_metrics(dict(np.load(t150_fixture)))
    add("Allegro T=150 L2", t150_l2["binary_accuracy"],
        t150_l2["categorical_accuracy"], source="existing T=150 fixture")
  t150_nf = _final_8m_row(compare_t150_dir) if compare_t150_dir else None
  if t150_nf is not None:
    add("Allegro T=150 NF 8M", t150_nf["binary_accuracy"],
        t150_nf["categorical_accuracy"], source=compare_t150_dir)
  if t150_nf_on_t50 is not None:
    add("Allegro T=150 NF on T=50 fixture",
        t150_nf_on_t50["binary_accuracy"],
        t150_nf_on_t50["categorical_accuracy"],
        source="no weight update")
  add("Allegro T=50 L2", t50_l2["binary_accuracy"],
      t50_l2["categorical_accuracy"], source="T=50 fixture")
  if t50_nf is not None:
    add("Allegro T=50 NF prefixes", t50_nf["binary_accuracy"],
        t50_nf["categorical_accuracy"], source="retrained compact-small")
  bb_fixture = os.path.join(
      compare_bb_dir, "fixed_heldout_eval_fixture.npz") if compare_bb_dir else ""
  if bb_fixture and os.path.isfile(bb_fixture):
    bb_l2 = l2_rank_metrics(dict(np.load(bb_fixture)))
    add("BB c3t1 L2", bb_l2["binary_accuracy"],
        bb_l2["categorical_accuracy"], source="existing BB fixture")
  bb_nf = _final_8m_row(compare_bb_dir) if compare_bb_dir else None
  if bb_nf is not None:
    add("BB c3t1 NF 8M", bb_nf["binary_accuracy"],
        bb_nf["categorical_accuracy"], source=compare_bb_dir)
  write_csv(os.path.join(output_dir, "horizon_comparison.csv"), rows)
  plot_horizon_comparison(
      os.path.join(output_dir, "horizon_comparison.png"), rows)
  write_json(os.path.join(output_dir, "horizon_comparison.json"), {
      "question": (
          "Does truncating Allegro ITS random 8M episodes to T=50 "
          "(gamma=0.99, 50% task-mix unchanged) raise NF rank accuracy "
          "to BB-like levels?"),
      "rows": rows,
  })
  return rows


def replot_output(
    output_dir: str, *, domain_label: str, batch_size: int = 256,
) -> list[str]:
  """Regenerate full, early, and scaling figures from existing CSV metrics."""
  all_rows = read_csv(os.path.join(output_dir, "all_metrics.csv"))
  labels = sorted({int(row["dataset_size_label"]) for row in all_rows})
  paths: list[str] = []
  for label in labels:
    rows = [
        row for row in all_rows if int(row["dataset_size_label"]) == label]
    model_dir = os.path.join(output_dir, f"size_{label}")
    full = os.path.join(model_dir, "learning_curves.png")
    early = os.path.join(model_dir, "learning_curves_early_0_500.png")
    plot_learning(
        full, rows, f"{label:,} dataset scale",
        domain_label=domain_label, batch_size=batch_size)
    plot_learning(
        early, rows, f"{label:,} dataset scale",
        domain_label=domain_label, batch_size=batch_size, early_max=500)
    paths.extend([full, early])
  final_rows = [
      max(
          (row for row in all_rows
           if int(row["dataset_size_label"]) == label),
          key=lambda row: float(row["step"]))
      for label in labels
  ]
  scaling = os.path.join(output_dir, "final_size_comparison.png")
  plot_scaling(scaling, final_rows, domain_label=domain_label)
  paths.append(scaling)
  return paths


HELDOUT_FIXTURE_KEYS = (
    "states",
    "actions",
    "positive_goals",
    "binary_negative_goals",
    "categorical_candidates",
)


def load_offline_split(
    args, *, expected_dims: tuple[int, int, int],
) -> dict[str, Any]:
  """Load episodes and recreate the deterministic 90/10 episode split."""
  horizon_truncate = resolve_horizon_truncate(args)
  obs, action, metadata, source_horizon, horizon = load_episode_arrays(
      args.dataset_dir, expected_dims, horizon_truncate=horizon_truncate)
  num_episodes = int(action.shape[0])
  state_dim, action_dim, goal_dim = expected_dims
  goal_indices = np.asarray(
      metadata["achieved_goal_state_indices"], dtype=np.int64)
  task_goal = np.asarray(metadata["task_goal"], dtype=np.float32)
  train_pool_episodes = int(np.floor(0.9 * num_episodes))
  val_ids = np.arange(train_pool_episodes, num_episodes, dtype=np.int64)
  print(
      "[split] recreated deterministic 90/10 episode-index split "
      f"(no stored IDs in dataset): train=[0, {train_pool_episodes}), "
      f"val=[{train_pool_episodes}, {num_episodes}); "
      "this matches the original Allegro offline trainer",
      flush=True)
  return {
      "obs": obs,
      "action": action,
      "metadata": metadata,
      "source_horizon": source_horizon,
      "horizon": horizon,
      "horizon_truncate": horizon_truncate,
      "num_episodes": num_episodes,
      "state_dim": state_dim,
      "action_dim": action_dim,
      "goal_dim": goal_dim,
      "goal_indices": goal_indices,
      "task_goal": task_goal,
      "train_pool_episodes": train_pool_episodes,
      "val_ids": val_ids,
  }


def regenerate_heldout_fixture(
    prepared: dict[str, Any], args,
) -> dict[str, np.ndarray]:
  val = EpisodeSet(
      prepared["obs"], prepared["action"], prepared["val_ids"],
      float(args.discount), prepared["goal_indices"])
  return make_eval_fixture(
      val, binary_count=args.binary_samples,
      categorical_count=args.categorical_samples,
      categorical_k=args.categorical_k, seed=int(args.seed) + 91_003)


def assert_same_heldout_fixture(
    existing_path: str, regenerated: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
  """Load an NF fixture and fail if it does not match seed+91003 regeneration."""
  if not existing_path or not os.path.isfile(existing_path):
    raise FileNotFoundError(
        "existing NF held-out fixture is required but missing: "
        f"{existing_path!r}. Refusing to silently create a replacement.")
  loaded = {key: np.asarray(value) for key, value in np.load(existing_path).items()}
  mismatches: list[str] = []
  for key in HELDOUT_FIXTURE_KEYS:
    if key not in loaded:
      mismatches.append(f"{key}: missing from {existing_path}")
      continue
    if key not in regenerated:
      mismatches.append(f"{key}: missing from regenerated fixture")
      continue
    left = np.asarray(loaded[key])
    right = np.asarray(regenerated[key])
    if left.shape != right.shape or left.dtype != right.dtype:
      mismatches.append(
          f"{key}: existing {left.shape}/{left.dtype} vs "
          f"regenerated {right.shape}/{right.dtype}")
    elif not np.array_equal(left, right):
      max_abs = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
      mismatches.append(
          f"{key}: arrays differ (max_abs={max_abs:.3e})")
  extra = ("positive_deltas", "goal_indices")
  for key in extra:
    if key in loaded and key in regenerated:
      if not np.array_equal(np.asarray(loaded[key]), np.asarray(regenerated[key])):
        mismatches.append(f"{key}: existing fixture disagrees with seed+91003")
  if mismatches:
    raise ValueError(
        "NF held-out fixture does not match seed+91003 regeneration "
        f"({existing_path}): " + "; ".join(mismatches))
  print(
      f"[fixture] verified existing NF fixture matches seed+91003: "
      f"{existing_path}",
      flush=True)
  merged = dict(regenerated)
  merged.update(loaded)
  return merged


def prepare_offline_run(
    args, *, expected_dims: tuple[int, int, int],
) -> dict[str, Any]:
  """Load (optionally truncated) episodes, rebuild the 90/10 split, write L2."""
  os.makedirs(args.output_dir, exist_ok=True)
  prepared = load_offline_split(args, expected_dims=expected_dims)
  fixture = regenerate_heldout_fixture(prepared, args)
  np.savez(
      os.path.join(args.output_dir, "fixed_heldout_eval_fixture.npz"),
      **fixture)
  l2 = l2_rank_metrics(fixture)
  write_json(os.path.join(args.output_dir, "l2_rank_metrics.json"), l2)
  print(
      f"[l2] T={prepared['horizon']} fixture binary={l2['binary_accuracy']:.4f} "
      f"cat@{int(1.0 / l2['categorical_chance'])}="
      f"{l2['categorical_accuracy']:.4f} "
      f"(closest candidate to current achieved q)",
      flush=True)
  if l2.get("binary_by_delta"):
    print(f"[l2] binary_by_delta={l2['binary_by_delta']}", flush=True)
    print(f"[l2] categorical_by_delta={l2['categorical_by_delta']}", flush=True)
  prepared["fixture"] = fixture
  prepared["l2"] = l2
  return prepared


def run_l2_only(
    args, *, domain_label: str, expected_dims: tuple[int, int, int],
) -> dict[str, Any]:
  prepared = prepare_offline_run(args, expected_dims=expected_dims)
  write_horizon_comparison(
      args.output_dir, t50_l2=prepared["l2"], t50_nf=None,
      t150_nf_on_t50=None,
      compare_t150_dir=getattr(args, "compare_t150_dir", "") or "",
      compare_bb_dir=getattr(args, "compare_bb_dir", "") or "")
  print(
      f"[l2-only] {domain_label} fixture+L2 written -> {args.output_dir}",
      flush=True)
  return prepared


def run_training(args, *, domain_label: str, expected_dims: tuple[int, int, int],
                 task_goal_stats_fraction: float) -> None:
  """Run the common compact-small fixed-dataset NF diagnostic."""
  if getattr(args, "l2_only", False):
    run_l2_only(args, domain_label=domain_label, expected_dims=expected_dims)
    return

  import jax
  import jax.numpy as jnp
  from contrastive import nf_density

  if args.smoke_test:
    args.dataset_sizes = args.dataset_sizes.split(",", maxsplit=1)[0]
    args.updates = 1
    args.eval_every = 1
    args.eval_nll_samples = min(args.eval_nll_samples, 32)
    args.binary_samples = min(args.binary_samples, 32)
    args.categorical_samples = min(args.categorical_samples, 16)
    args.categorical_k = min(args.categorical_k, 8)
    args.goal_stat_samples = min(args.goal_stat_samples, 1024)
    print("[smoke] compact-small NF init plus one real update requested",
          flush=True)

  prepared = prepare_offline_run(args, expected_dims=expected_dims)
  obs = prepared["obs"]
  action = prepared["action"]
  metadata = prepared["metadata"]
  source_horizon = prepared["source_horizon"]
  horizon = prepared["horizon"]
  num_episodes = prepared["num_episodes"]
  state_dim = prepared["state_dim"]
  action_dim = prepared["action_dim"]
  goal_dim = prepared["goal_dim"]
  goal_indices = prepared["goal_indices"]
  task_goal = prepared["task_goal"]
  train_pool_episodes = prepared["train_pool_episodes"]
  val_ids = prepared["val_ids"]
  fixture = prepared["fixture"]

  nets = nf_density.make_nf_density_networks(
      obs_dim=state_dim, act_dim=action_dim, goal_dim=goal_dim,
      rep_size=64, num_blocks=6, channels=192, goal_enc_size=0,
      sa_hidden=192, sa_num_layers=3, state_only=False)
  optimizer = nf_density.make_nf_optimizers(
      encoder_lr=3e-4, critic_lr=1e-4, critic_weight_decay=1e-6,
      grad_clip=1.0, has_goal_encoder=False)
  update = nf_density.make_nf_density_update_fn(
      nets, optimizer, obs_dim=state_dim, noise_std=0.05)

  @jax.jit
  def score(params, states, actions, goals, mean, std):
    normalized = (goals - mean) / (std + 1e-8)
    return nf_density.nf_log_prob(
        nets, params, states, actions, normalized)

  def evaluate(params, train_fixture, mean, std):
    def scores(states, actions, goals):
      return np.asarray(score(
          params, jnp.asarray(states), jnp.asarray(actions),
          jnp.asarray(goals), jnp.asarray(mean), jnp.asarray(std)))

    train_lp = scores(
        train_fixture["obs"][:, :state_dim], train_fixture["action"],
        train_fixture["obs"][:, state_dim:])
    n_binary = int(args.binary_samples)
    pos = scores(
        fixture["states"][:n_binary], fixture["actions"][:n_binary],
        fixture["positive_goals"][:n_binary])
    neg = scores(
        fixture["states"][:n_binary], fixture["actions"][:n_binary],
        fixture["binary_negative_goals"])
    shuffled_context = scores(
        np.roll(fixture["states"][:n_binary], 1, axis=0),
        np.roll(fixture["actions"][:n_binary], 1, axis=0),
        fixture["positive_goals"][:n_binary])
    shuffled_goal_lp = scores(
        fixture["states"][:n_binary], fixture["actions"][:n_binary],
        np.roll(fixture["positive_goals"][:n_binary], 1, axis=0))
    task_lp = scores(
        fixture["states"][:n_binary], fixture["actions"][:n_binary],
        np.broadcast_to(task_goal, (n_binary, goal_dim)))
    candidates = fixture["categorical_candidates"]
    n_cat, k, _ = candidates.shape
    cat_scores = scores(
        np.repeat(fixture["states"][:n_cat], k, axis=0),
        np.repeat(fixture["actions"][:n_cat], k, axis=0),
        candidates.reshape(-1, goal_dim)).reshape(n_cat, k)
    binary_correct = pos > neg
    cat_correct = np.argmax(cat_scores, axis=1) == 0
    metrics = {
        "train_nll": float(-train_lp.mean()),
        "val_nll": float(-pos.mean()),
        "train_val_gap": float(-pos.mean() + train_lp.mean()),
        "binary_accuracy": float(np.mean(binary_correct)),
        "binary_tie_fraction": float(np.mean(pos == neg)),
        "categorical_accuracy": float(np.mean(cat_correct)),
        "categorical_chance": 1.0 / float(k),
        "positive_logp": float(pos.mean()),
        "negative_logp": float(neg.mean()),
        "binary_margin": float((pos - neg).mean()),
        "shuffled_context_logp": float(shuffled_context.mean()),
        "shuffled_context_binary_accuracy": float(
            np.mean(shuffled_context > neg)),
        "shuffled_goal_logp": float(shuffled_goal_lp.mean()),
        "shuffled_goal_binary_accuracy": float(
            np.mean(shuffled_goal_lp > neg)),
        "task_goal_logp": float(task_lp.mean()),
        "horizon": int(horizon),
    }
    deltas = fixture.get("positive_deltas")
    if deltas is not None:
      bin_strat = stratify_by_delta(
          np.asarray(deltas)[:n_binary], binary_correct)
      cat_strat = stratify_by_delta(
          np.asarray(deltas)[:n_cat], cat_correct)
      for name, _, _ in DELTA_BINS:
        metrics[f"binary_accuracy_delta_{name}"] = bin_strat[f"delta_{name}_acc"]
        metrics[f"binary_n_delta_{name}"] = bin_strat[f"delta_{name}_n"]
        metrics[f"categorical_accuracy_delta_{name}"] = cat_strat[
            f"delta_{name}_acc"]
        metrics[f"categorical_n_delta_{name}"] = cat_strat[f"delta_{name}_n"]
    return metrics

  t150_nf_on_t50 = None
  ckpt_path = getattr(args, "rescore_checkpoint", "") or ""
  if ckpt_path:
    with open(ckpt_path, "rb") as handle:
      ckpt = pickle.load(handle)
    rescore_train = EpisodeSet(
        obs, action, np.arange(min(256, train_pool_episodes), dtype=np.int64),
        float(args.discount), goal_indices).sample(
            int(args.eval_nll_samples),
            np.random.default_rng(int(args.seed) + 41_001))
    t150_nf_on_t50 = evaluate(
        ckpt["q_params"], rescore_train,
        np.asarray(ckpt["goal_mean"], dtype=np.float32),
        np.asarray(ckpt["goal_std"], dtype=np.float32))
    t150_nf_on_t50["checkpoint"] = os.path.abspath(ckpt_path)
    t150_nf_on_t50["weight_update"] = False
    t150_nf_on_t50["normalization"] = "checkpoint T=150 goal mean/std"
    write_json(
        os.path.join(args.output_dir, "t150_nf_on_t50_fixture.json"),
        t150_nf_on_t50)
    print(
        "[rescore] existing T=150 NF on T=50 fixture (no update): "
        f"bin={t150_nf_on_t50['binary_accuracy']:.4f} "
        f"cat={t150_nf_on_t50['categorical_accuracy']:.4f} "
        f"val_nll={t150_nf_on_t50['val_nll']:.3f}",
        flush=True)

  labels = [int(value) for value in args.dataset_sizes.split(",")]
  all_rows: list[dict[str, Any]] = []
  final_rows: list[dict[str, Any]] = []
  run_metadata = {
      "dataset_metadata": metadata,
      "split": {
          "semantics": (
              "shared global 90/10 episode-index split; IDs were not stored "
              "in the dataset, so they are recreated as "
              "arange(floor(0.9*N)) / remaining — identical to the original "
              "Allegro T=150 offline trainer"),
          "train_episode_pool": [0, train_pool_episodes],
          "validation_episode_range": [train_pool_episodes, num_episodes],
          "validation_episodes": len(val_ids),
          "validation_transitions": len(val_ids) * horizon,
          "no_validation_episode_used_by_any_model": True,
          "source_horizon": int(source_horizon),
          "used_horizon": int(horizon),
          "horizon_truncate": prepared["horizon_truncate"],
      },
      "categorical_eval": {
          "k": int(args.categorical_k),
          "chance": 1.0 / float(args.categorical_k),
          "positive": "same-episode gamma=.99 truncated-geometric future",
          "negatives": (
              "K-1 random achieved states from distinct held-out episodes, "
              "each different from anchor episode"),
          "fixture_seed": int(args.seed) + 91_003,
          "fixed_across_steps_and_dataset_sizes": True,
      },
      "binary_eval": {
          "positive": "same-episode gamma=.99 truncated-geometric future",
          "negative": "random achieved state from a different held-out episode",
          "chance": 0.5,
      },
      "recipe": {
          "conditioning": "p(g|s,a)",
          "state_action_encoder": "3x192",
          "rank": 64,
          "flow_blocks": 6,
          "flow_width": 192,
          "goal_encoder": 0,
          "discount": float(args.discount),
          "batch_size": int(args.batch_size),
          "sa_lr": 3e-4,
          "flow_lr": 1e-4,
          "weight_decay": 1e-6,
          "grad_clip": 1.0,
          "normalized_goal_noise_std": 0.05,
          "goal_std_min": 0.02,
          "task_goal_stats_fraction": float(task_goal_stats_fraction),
          "goal_stats_source": (
              "replay-only train goals" if task_goal_stats_fraction == 0
              else "replay/task-goal mixture"),
          "changed_variable": (
              "horizon prefix only; Allegro offline default 50% task-goal "
              "stat mix is unchanged"),
          "optional_gamma_0.95_t150_arm": "skipped to keep the test short",
          "updates": int(args.updates),
          "eval_every": int(args.eval_every),
          "action_dim": int(action_dim),
      },
      "models": [],
  }
  started = time.time()
  for model_index, label in enumerate(labels):
    nominal_episodes = min(label // horizon, num_episodes)
    train_episodes = min(
        int(np.floor(0.9 * nominal_episodes)), train_pool_episodes)
    train_ids = np.arange(train_episodes, dtype=np.int64)
    train = EpisodeSet(
        obs, action, train_ids, float(args.discount), goal_indices)
    model_dir = os.path.join(args.output_dir, f"size_{label}")
    os.makedirs(model_dir, exist_ok=True)
    mean, std, stat_details = goal_stats(
        train, sample_count=int(args.goal_stat_samples),
        seed=int(args.seed) + 7_001 + model_index, std_min=0.02,
        task_goal=task_goal,
        task_goal_fraction=float(task_goal_stats_fraction))
    train_fixture = train.sample(
        int(args.eval_nll_samples),
        np.random.default_rng(int(args.seed) + 31_001 + model_index))
    key = jax.random.PRNGKey(int(args.seed))
    key, init_key = jax.random.split(key)
    params = nf_density.init_nf_params(nets, init_key)
    opt_state = optimizer.init(params)
    model_rows: list[dict[str, Any]] = []
    interval_losses: list[float] = []
    train_rng = np.random.default_rng(int(args.seed) + 101 + model_index)
    for step_i in range(int(args.updates) + 1):
      if step_i % int(args.eval_every) == 0 or step_i == int(args.updates):
        metrics = evaluate(params, train_fixture, mean, std)
        row = {
            "step": step_i,
            "dataset_size_label": label,
            "nominal_subset_transitions": nominal_episodes * horizon,
            "train_episodes": train_episodes,
            "train_transitions": train_episodes * horizon,
            "validation_episodes": len(val_ids),
            "validation_transitions": len(val_ids) * horizon,
            "train_nf_loss": (
                float(np.mean(interval_losses)) if interval_losses
                else metrics["train_nll"]),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        model_rows.append(row)
        all_rows.append(row)
        interval_losses.clear()
        print(
            f"[offline-nf] size={label:,} step={step_i}/{args.updates} "
            f"train_nll={row['train_nll']:.3f} "
            f"val_nll={row['val_nll']:.3f} "
            f"bin={row['binary_accuracy']:.3f} "
            f"cat@{args.categorical_k}={row['categorical_accuracy']:.3f}",
            flush=True)
      if step_i == int(args.updates):
        break
      batch_np = train.sample(int(args.batch_size), train_rng)
      batch = {
          name: jnp.asarray(batch_np[name])
          for name in ("obs", "action", "next_obs")
      }
      key, update_key = jax.random.split(key)
      params, opt_state, _, update_metrics = update(
          params, opt_state, batch, update_key,
          jnp.asarray(mean), jnp.asarray(std))
      interval_losses.append(
          float(np.asarray(update_metrics["density_loss"])))
      if args.smoke_test:
        print(
            "[smoke] real NF update synchronized: "
            f"density_loss={interval_losses[-1]:.6f}", flush=True)

    write_csv(os.path.join(model_dir, "metrics.csv"), model_rows)
    plot_learning(
        os.path.join(model_dir, "learning_curves.png"), model_rows,
        f"{label:,} dataset scale", domain_label=domain_label,
        batch_size=int(args.batch_size))
    plot_learning(
        os.path.join(model_dir, "learning_curves_early_0_500.png"), model_rows,
        f"{label:,} dataset scale", domain_label=domain_label,
        batch_size=int(args.batch_size), early_max=500)
    if "binary_accuracy_delta_1" in model_rows[-1]:
      plot_delta_stratified(
          os.path.join(model_dir, "accuracy_by_delta.png"), model_rows,
          domain_label=domain_label, label=f"{label:,} dataset scale")
    with open(os.path.join(model_dir, "final_model.pkl"), "wb") as handle:
      pickle.dump({
          "q_params": jax.device_get(params),
          "goal_mean": mean,
          "goal_std": std,
          "dataset_size_label": label,
          "train_episode_ids": train_ids,
      }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_meta = {
        "dataset_size_label": label,
        "nominal_subset_episodes": nominal_episodes,
        "nominal_subset_transitions": nominal_episodes * horizon,
        "train_episodes": train_episodes,
        "train_transitions": train_episodes * horizon,
        "goal_mean": mean.tolist(),
        "goal_std": std.tolist(),
        "goal_stats": stat_details,
    }
    write_json(os.path.join(model_dir, "metadata.json"), model_meta)
    run_metadata["models"].append(model_meta)
    final_rows.append(model_rows[-1])

  write_csv(os.path.join(args.output_dir, "all_metrics.csv"), all_rows)
  write_csv(
      os.path.join(args.output_dir, "final_size_comparison.csv"), final_rows)
  plot_scaling(
      os.path.join(args.output_dir, "final_size_comparison.png"), final_rows,
      domain_label=domain_label)
  write_horizon_comparison(
      args.output_dir, t50_l2=prepared["l2"],
      t50_nf=final_rows[-1] if final_rows else None,
      t150_nf_on_t50=t150_nf_on_t50,
      compare_t150_dir=getattr(args, "compare_t150_dir", "") or "",
      compare_bb_dir=getattr(args, "compare_bb_dir", "") or "")
  run_metadata["elapsed_seconds"] = time.time() - started
  run_metadata["l2_rank_metrics"] = prepared["l2"]
  run_metadata["t150_nf_on_t50_fixture"] = t150_nf_on_t50
  write_json(os.path.join(args.output_dir, "summary.json"), run_metadata)
  if args.smoke_test:
    print("[smoke] PASS: compact-small NF initialized and updated", flush=True)
  print(f"[offline-nf] complete -> {args.output_dir}", flush=True)
