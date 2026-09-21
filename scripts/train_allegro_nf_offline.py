#!/usr/bin/env python3
"""Train compact conditional NFs on fixed Allegro ITS episode datasets."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass

# Torch-first cuDNN 8 before the JAX/Acme import graph (shared helper).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

from scripts.offline_nf.cuda_init import initialize_torch_first  # noqa: E402

initialize_torch_first()

import numpy as np  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from contrastive import nf_density  # noqa: E402


@dataclass(frozen=True)
class EpisodeSet:
  obs: np.ndarray
  action: np.ndarray
  episode_ids: np.ndarray
  discount: float
  goal_indices: np.ndarray

  def sample(self, count: int, rng: np.random.Generator) -> dict:
    """EpisodeReplay-equivalent iid episode/timestep/geometric sampling."""
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
    actions = np.asarray(
        self.action[episode_ids, t], dtype=np.float32)
    return {
        "obs": np.concatenate([states, goals], axis=-1),
        "action": actions,
        "next_obs": np.concatenate([next_states, goals], axis=-1),
        "_episode_id": episode_ids,
    }


def _write_json(path: str, value: dict) -> None:
  tmp = path + ".tmp"
  with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
  os.replace(tmp, path)


def _write_csv(path: str, rows: list[dict]) -> None:
  tmp = path + ".tmp"
  with open(tmp, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  os.replace(tmp, path)


def _load_pyplot():
  """Load pyplot, using the cluster's Python-3.8 plotting package if needed."""
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


def _make_eval_fixture(
    val: EpisodeSet,
    *,
    binary_count: int,
    categorical_count: int,
    categorical_k: int,
    seed: int,
) -> dict:
  """Create fixed held-out contexts and cross-episode candidate goals."""
  rng = np.random.default_rng(seed)
  total = max(int(binary_count), int(categorical_count))
  positive = val.sample(total, rng)
  states = positive["obs"][:, :val.obs.shape[-1]]
  actions = positive["action"]
  pos_goals = positive["obs"][:, val.obs.shape[-1]:]
  anchor_eps = positive["_episode_id"]

  candidates = np.empty(
      (int(categorical_count), int(categorical_k), len(val.goal_indices)),
      dtype=np.float32)
  candidates[:, 0] = pos_goals[:categorical_count]
  for row in range(int(categorical_count)):
    # K-1 distinct negative episodes, all different from the anchor episode.
    pool = val.episode_ids[val.episode_ids != anchor_eps[row]]
    replace = len(pool) < int(categorical_k) - 1
    negative_eps = rng.choice(
        pool, size=int(categorical_k) - 1, replace=replace)
    negative_t = rng.integers(
        0, val.obs.shape[1], size=int(categorical_k) - 1)
    candidates[row, 1:] = val.obs[
        negative_eps, negative_t][:, val.goal_indices]

  # Binary negatives use a separately fixed different-episode draw.
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
  }


def _goal_stats(
    train: EpisodeSet,
    task_goal: np.ndarray,
    *,
    sample_count: int,
    seed: int,
    std_min: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
  rng = np.random.default_rng(seed)
  replay_goals = train.sample(int(sample_count), rng)["obs"][
      :, train.obs.shape[-1]:]
  replay_mean = replay_goals.mean(axis=0, dtype=np.float64)
  replay_second = np.square(replay_goals, dtype=np.float64).mean(axis=0)
  # Exact 50/50 mixture moments: replay future goals and copies of task goal.
  mean = 0.5 * (replay_mean + task_goal)
  second = 0.5 * (replay_second + np.square(task_goal))
  std_raw = np.sqrt(np.maximum(second - np.square(mean), 0.0))
  std = np.maximum(std_raw, float(std_min))
  details = {
      "replay_future_samples": int(sample_count),
      "replay_mean": replay_mean.tolist(),
      "task_goal_fraction": 0.5,
      "std_before_floor": std_raw.tolist(),
      "std_floor": float(std_min),
  }
  return mean.astype(np.float32), std.astype(np.float32), details


def _plot_learning(path: str, rows: list[dict], label: str) -> None:
  plt = _load_pyplot()

  x = np.asarray([row["step"] for row in rows])
  fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
  axes[0, 0].plot(x, [row["train_nll"] for row in rows], label="train NLL")
  axes[0, 0].plot(x, [row["val_nll"] for row in rows], label="held-out NLL")
  axes[0, 0].set_ylabel("NLL")
  axes[0, 0].legend()
  axes[0, 1].plot(x, [row["binary_accuracy"] for row in rows])
  axes[0, 1].axhline(0.5, color="0.5", ls="--", label="chance")
  axes[0, 1].set_ylabel("held-out binary accuracy")
  axes[0, 1].set_ylim(0, 1)
  axes[0, 1].legend()
  axes[1, 0].plot(x, [row["categorical_accuracy"] for row in rows])
  axes[1, 0].axhline(
      rows[0]["categorical_chance"], color="0.5", ls="--", label="chance")
  axes[1, 0].set_ylabel("held-out categorical accuracy")
  axes[1, 0].set_ylim(0, 1)
  axes[1, 0].legend()
  axes[1, 1].plot(
      x, [row["task_goal_logp"] for row in rows], label="task-goal log p")
  axes[1, 1].plot(
      x, [row["shuffled_context_logp"] for row in rows],
      label="shuffled-context log p")
  axes[1, 1].set_ylabel("mean log p")
  axes[1, 1].legend()
  for axis in axes.flat:
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("gradient updates")
  fig.suptitle(f"Offline Allegro ITS NF learning — {label}")
  fig.tight_layout()
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def _plot_scaling(path: str, final_rows: list[dict]) -> None:
  plt = _load_pyplot()

  x = np.asarray([row["train_transitions"] for row in final_rows], dtype=float)
  fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
  axes[0].plot(x, [row["val_nll"] for row in final_rows], marker="o")
  axes[0].set_ylabel("final held-out NLL")
  axes[1].plot(x, [row["binary_accuracy"] for row in final_rows], marker="o")
  axes[1].axhline(0.5, color="0.5", ls="--")
  axes[1].set_ylabel("final binary accuracy")
  axes[1].set_ylim(0, 1)
  axes[2].plot(
      x, [row["categorical_accuracy"] for row in final_rows], marker="o")
  axes[2].axhline(
      final_rows[0]["categorical_chance"], color="0.5", ls="--")
  axes[2].set_ylabel("final categorical accuracy")
  axes[2].set_ylim(0, 1)
  for axis in axes:
    axis.set_xscale("log")
    axis.set_xlabel("actual train transitions")
    axis.grid(axis="y", alpha=0.25)
  fig.suptitle("Offline Allegro ITS NF: final dataset-size comparison")
  fig.tight_layout()
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
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
      "--smoke_test", action="store_true",
      help=("run one real compact-small NF update on the first dataset scale "
            "with reduced evaluation fixtures"))
  args = parser.parse_args()

  if args.smoke_test:
    first_size = args.dataset_sizes.split(",", maxsplit=1)[0]
    args.dataset_sizes = first_size
    args.updates = 1
    args.eval_every = 1
    args.eval_nll_samples = min(args.eval_nll_samples, 32)
    args.binary_samples = min(args.binary_samples, 32)
    args.categorical_samples = min(args.categorical_samples, 16)
    args.categorical_k = min(args.categorical_k, 8)
    args.goal_stat_samples = min(args.goal_stat_samples, 1024)
    print(
        "[smoke] compact-small NF init plus one real update requested",
        flush=True)

  os.makedirs(args.output_dir, exist_ok=True)
  with open(os.path.join(args.dataset_dir, "metadata.json"), encoding="utf-8") as h:
    metadata = json.load(h)
  obs = np.load(
      os.path.join(args.dataset_dir, metadata["observations_file"]),
      mmap_mode="r")
  action = np.load(
      os.path.join(args.dataset_dir, metadata["actions_file"]), mmap_mode="r")
  num_episodes, horizon = int(action.shape[0]), int(action.shape[1])
  if obs.shape != (num_episodes, horizon + 1, 16):
    raise ValueError(f"unexpected observation shape {obs.shape}")
  if action.shape[2] != 8:
    raise ValueError(f"unexpected action shape {action.shape}")
  goal_indices = np.asarray(
      metadata["achieved_goal_state_indices"], dtype=np.int64)
  task_goal = np.asarray(metadata["task_goal"], dtype=np.float32)

  # One global 90/10 split. All scales share exactly the same held-out set.
  train_pool_episodes = int(np.floor(0.9 * num_episodes))
  val_ids = np.arange(train_pool_episodes, num_episodes, dtype=np.int64)
  val = EpisodeSet(obs, action, val_ids, float(args.discount), goal_indices)
  fixture = _make_eval_fixture(
      val, binary_count=args.binary_samples,
      categorical_count=args.categorical_samples,
      categorical_k=args.categorical_k, seed=int(args.seed) + 91_003)
  np.savez(
      os.path.join(args.output_dir, "fixed_heldout_eval_fixture.npz"),
      **fixture)

  nets = nf_density.make_nf_density_networks(
      obs_dim=16, act_dim=8, goal_dim=8, rep_size=64, num_blocks=6,
      channels=192, goal_enc_size=0, sa_hidden=192, sa_num_layers=3,
      state_only=False)
  optimizer = nf_density.make_nf_optimizers(
      encoder_lr=3e-4, critic_lr=1e-4, critic_weight_decay=1e-6,
      grad_clip=1.0, has_goal_encoder=False)
  update = nf_density.make_nf_density_update_fn(
      nets, optimizer, obs_dim=16, noise_std=0.05)

  @jax.jit
  def score(params, states, actions, goals, mean, std):
    normalized = (goals - mean) / (std + 1e-8)
    return nf_density.nf_log_prob(
        nets, params, states, actions, normalized)

  def evaluate(params, train_fixture, mean, std) -> dict:
    def scores(states, actions, goals):
      return np.asarray(score(
          params, jnp.asarray(states), jnp.asarray(actions),
          jnp.asarray(goals), jnp.asarray(mean), jnp.asarray(std)))

    train_lp = scores(
        train_fixture["obs"][:, :16], train_fixture["action"],
        train_fixture["obs"][:, 16:])
    val_batch = fixture
    n_binary = int(args.binary_samples)
    pos = scores(
        val_batch["states"][:n_binary], val_batch["actions"][:n_binary],
        val_batch["positive_goals"][:n_binary])
    neg = scores(
        val_batch["states"][:n_binary], val_batch["actions"][:n_binary],
        val_batch["binary_negative_goals"])
    shuffled_context = scores(
        np.roll(val_batch["states"][:n_binary], 1, axis=0),
        np.roll(val_batch["actions"][:n_binary], 1, axis=0),
        val_batch["positive_goals"][:n_binary])
    shuffled_goals = np.roll(val_batch["positive_goals"][:n_binary], 1, axis=0)
    shuffled_goal_lp = scores(
        val_batch["states"][:n_binary], val_batch["actions"][:n_binary],
        shuffled_goals)
    task_lp = scores(
        val_batch["states"][:n_binary], val_batch["actions"][:n_binary],
        np.broadcast_to(task_goal, (n_binary, len(task_goal))))

    candidates = val_batch["categorical_candidates"]
    n_cat, k, goal_dim = candidates.shape
    cat_states = np.repeat(val_batch["states"][:n_cat], k, axis=0)
    cat_actions = np.repeat(val_batch["actions"][:n_cat], k, axis=0)
    cat_scores = scores(
        cat_states, cat_actions, candidates.reshape(-1, goal_dim)).reshape(
            n_cat, k)
    return {
        "train_nll": float(-train_lp.mean()),
        "val_nll": float(-pos.mean()),
        "train_val_gap": float(-pos.mean() + train_lp.mean()),
        "binary_accuracy": float(np.mean(pos > neg)),
        "binary_tie_fraction": float(np.mean(pos == neg)),
        "categorical_accuracy": float(np.mean(np.argmax(cat_scores, axis=1) == 0)),
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
    }

  labels = [int(value) for value in args.dataset_sizes.split(",")]
  all_rows: list[dict] = []
  final_rows: list[dict] = []
  run_metadata = {
      "dataset_metadata": metadata,
      "split": {
          "semantics": "shared global 90/10 episode split",
          "train_episode_pool": [0, train_pool_episodes],
          "validation_episode_range": [train_pool_episodes, num_episodes],
          "validation_episodes": len(val_ids),
          "validation_transitions": len(val_ids) * horizon,
          "no_validation_episode_used_by_any_model": True,
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
          "task_goal_stats_fraction": 0.5,
          "updates": int(args.updates),
          "eval_every": int(args.eval_every),
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
    mean, std, stat_details = _goal_stats(
        train, task_goal, sample_count=int(args.goal_stat_samples),
        seed=int(args.seed) + 7_001 + model_index, std_min=0.02)
    train_eval_rng = np.random.default_rng(
        int(args.seed) + 31_001 + model_index)
    train_fixture = train.sample(int(args.eval_nll_samples), train_eval_rng)
    # Identical initialization across scales isolates the data-size effect.
    key = jax.random.PRNGKey(int(args.seed))
    key, init_key = jax.random.split(key)
    params = nf_density.init_nf_params(nets, init_key)
    opt_state = optimizer.init(params)
    model_rows: list[dict] = []
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
                float(np.mean(interval_losses)) if interval_losses else
                metrics["train_nll"]),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        model_rows.append(row)
        all_rows.append(row)
        interval_losses.clear()
        print(
            f"[offline-nf] size={label:,} step={step_i}/{args.updates} "
            f"train_nll={row['train_nll']:.3f} val_nll={row['val_nll']:.3f} "
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
            f"density_loss={interval_losses[-1]:.6f}",
            flush=True)

    _write_csv(os.path.join(model_dir, "metrics.csv"), model_rows)
    _plot_learning(
        os.path.join(model_dir, "learning_curves.png"), model_rows,
        f"{label:,} dataset scale")
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
    _write_json(os.path.join(model_dir, "metadata.json"), model_meta)
    run_metadata["models"].append(model_meta)
    final_rows.append(model_rows[-1])

  _write_csv(os.path.join(args.output_dir, "all_metrics.csv"), all_rows)
  _write_csv(os.path.join(args.output_dir, "final_size_comparison.csv"), final_rows)
  _plot_scaling(
      os.path.join(args.output_dir, "final_size_comparison.png"), final_rows)
  run_metadata["elapsed_seconds"] = time.time() - started
  _write_json(os.path.join(args.output_dir, "summary.json"), run_metadata)
  if args.smoke_test:
    print("[smoke] PASS: compact-small NF initialized and updated", flush=True)
  print(f"[offline-nf] complete -> {args.output_dir}", flush=True)


def shared_main() -> None:
  """Run the shared trainer while preserving the established Allegro CLI."""
  from scripts.offline_nf.core import make_train_parser, run_training

  parser = make_train_parser(
      "Train compact conditional NFs on fixed Allegro ITS datasets.")
  args = parser.parse_args()
  run_training(
      args, domain_label="Allegro ITS", expected_dims=(16, 8, 8),
      task_goal_stats_fraction=0.5)


if __name__ == "__main__":
  shared_main()
