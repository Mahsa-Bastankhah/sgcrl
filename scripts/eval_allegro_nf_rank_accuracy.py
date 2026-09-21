#!/usr/bin/env python3
"""Evaluate NF positive-vs-negative ranking across Allegro checkpoints.

For each checkpoint, run two stochastic policy rollouts in parallel.  For an
anchor (s_t, a_t), the positive goal is a truncated-geometric future achieved
goal from the same rollout (matching EpisodeReplay).  The negative goal is
sampled from a random timestep of the other rollout.  Report:

  P[log p(g_pos | s_t, a_t) > log p(g_neg | s_t, a_t)]

Positive pairs crossing an environment reset are excluded.  Both rollout
directions are used, giving about 596 comparisons for a 300-step horizon.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts import allegro_kuka_throw_ckpt_video as ckpt_utils


CKPT_RE = re.compile(r"ckpt_iter_(\d+)\.pkl$")


def _iteration(path: str) -> int:
  match = CKPT_RE.search(os.path.basename(path))
  if match is None:
    raise ValueError(f"not a checkpoint filename: {path}")
  return int(match.group(1))


def _load_ckpt(path: str):
  with open(path, "rb") as fh:
    return pickle.load(fh)


def _rollout(env, act, policy_params, key, n_steps: int,
             random_actions: bool = False):
  """Return packed obs, actions, post-action done flags, and new PRNG key."""
  import jax
  import jax.numpy as jnp
  import torch

  obs_t = env.reset()
  packed = []
  actions = []
  dones = []
  act_dim = int(env.action_dim)
  n_env = int(env.num_envs)
  for _ in range(int(n_steps)):
    obs = np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
    if obs.ndim == 1:
      obs = obs[None, :]
    key, subkey = jax.random.split(key)
    if random_actions:
      action = jax.random.uniform(
          subkey, (n_env, act_dim), minval=-1.0, maxval=1.0)
    else:
      action = act(policy_params, jnp.asarray(obs), subkey)
    action_np = np.asarray(action, dtype=np.float32)
    packed.append(obs.copy())
    actions.append(action_np.copy())
    obs_t, _, done_t = env.step(
        torch.as_tensor(action_np, dtype=torch.float32, device=env.device))
    dones.append(np.asarray(done_t.detach().cpu().numpy(), dtype=bool).copy())
  return (
      np.asarray(packed, dtype=np.float32),
      np.asarray(actions, dtype=np.float32),
      np.asarray(dones, dtype=bool),
      key,
  )


def _make_pairs(
    packed: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    obs_dim: int,
    goal_state_indices: np.ndarray,
    discount: float,
    rng: np.random.Generator,
):
  """Create symmetric positive/negative comparisons from two rollouts."""
  if packed.ndim != 3 or packed.shape[1] != 2:
    raise ValueError(f"expected packed shape (T,2,D), got {packed.shape}")
  states = packed[:, :, :obs_dim]
  achieved = states[:, :, goal_state_indices]
  anchor_obs = []
  anchor_actions = []
  pos_goals = []
  neg_goals = []
  deltas = []

  for src, neg_src in ((0, 1), (1, 0)):
    for t in range(packed.shape[0] - 1):
      # done[k] means obs[k+1] belongs to a reset episode. Find the largest
      # available future offset that stays inside this episode.
      max_delta = 0
      for delta_i in range(1, packed.shape[0] - t):
        if np.any(dones[t:t + delta_i, src]):
          break
        max_delta = delta_i
      if max_delta < 1:
        continue
      # Exact truncated-geometric sampler used by EpisodeReplay.sample().
      if 0.0 < discount < 1.0:
        trunc_cdf = 1.0 - float(discount) ** max_delta
        u = float(rng.random()) * trunc_cdf
        delta = 1 + int(np.floor(
            np.log1p(-u) / np.log(float(discount))))
        delta = int(np.clip(delta, 1, max_delta))
      else:
        delta = int(rng.integers(1, max_delta + 1))
      neg_t = int(rng.integers(0, packed.shape[0]))
      anchor_obs.append(states[t, src])
      anchor_actions.append(actions[t, src])
      pos_goals.append(achieved[t + delta, src])
      neg_goals.append(achieved[neg_t, neg_src])
      deltas.append(delta)

  return tuple(
      np.asarray(x, dtype=np.float32)
      for x in (anchor_obs, anchor_actions, pos_goals, neg_goals, deltas)
  )


def _achieved_goal_indices(env) -> np.ndarray:
  """Infer obs→goal coordinates from Allegro environment packing."""
  obs_dim = int(env.obs_dim)
  goal_dim = int(env.goal_dim)
  if bool(getattr(env, "control_sanity_trim_sa", False)):
    # Trimmed state is exactly [controlled normalized q, controlled qd].
    indices = np.arange(goal_dim, dtype=np.int32)
  elif str(getattr(env, "control_sanity_mode", "") or "") in (
      "finger", "hand16", "hand16fig", "index", "six", "two_finger", "three2",
      "four2",
      "four2h", "four2m", "four2mh", "four2mm", "four2mmh", "four2mmx",
      "four2w",
  ):
    hand_indices = getattr(env, "control_sanity_hand_indices", None)
    if hand_indices is None:
      hand_indices = range(goal_dim)
    # Full joint state starts with 7 arm q followed by 16 hand q.
    indices = np.asarray(
        [7 + int(i) for i in hand_indices], dtype=np.int32)
  else:
    # Throw/palm goals are packed at the tail of the physical state.
    indices = np.arange(obs_dim - goal_dim, obs_dim, dtype=np.int32)
  if len(indices) != goal_dim:
    raise ValueError(
        f"achieved-goal index count {len(indices)} != goal_dim {goal_dim}")
  if np.any(indices < 0) or np.any(indices >= obs_dim):
    raise ValueError(
        f"achieved-goal indices {indices.tolist()} outside obs_dim={obs_dim}")
  return indices


def _score_pairs(
    reward_fn,
    nf_params,
    anchors: np.ndarray,
    actions: np.ndarray,
    pos_goals: np.ndarray,
    neg_goals: np.ndarray,
    goal_mean: np.ndarray,
    goal_std: np.ndarray,
):
  import jax.numpy as jnp

  pos_obs = np.concatenate([anchors, pos_goals], axis=-1)
  neg_obs = np.concatenate([anchors, neg_goals], axis=-1)
  pos = np.asarray(
      reward_fn(
          nf_params, jnp.asarray(pos_obs), jnp.asarray(actions),
          jnp.asarray(goal_mean), jnp.asarray(goal_std)),
      dtype=np.float32,
  )
  neg = np.asarray(
      reward_fn(
          nf_params, jnp.asarray(neg_obs), jnp.asarray(actions),
          jnp.asarray(goal_mean), jnp.asarray(goal_std)),
      dtype=np.float32,
  )
  return pos, neg


def _write_csv(path: str, rows: list[dict]) -> None:
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  os.replace(tmp, path)


def _plot_pil(path: str, rows: list[dict]) -> None:
  """Dependency-light fallback for the Isaac Gym env without Matplotlib."""
  from PIL import Image, ImageDraw

  width, height = 1200, 1000
  image = Image.new("RGB", (width, height), "white")
  draw = ImageDraw.Draw(image)
  x_values = np.asarray([r["iteration"] for r in rows], dtype=float)
  panels = (
      ("NF rank accuracy", np.asarray([r["rank_accuracy"] for r in rows]),
       (76, 155, 232), (0.0, 1.0), 0.5),
      ("Mean raw log p", None, None, None, None),
      ("Mean log-p margin", np.asarray([r["margin_mean"] for r in rows]),
       (168, 76, 232), None, 0.0),
  )
  draw.text(
      (40, 18),
      "Allegro NF ranking: same-rollout future vs other-rollout goal",
      fill="black",
  )
  left, right = 90, width - 35
  panel_h = 260

  def draw_series(box, values, color, y_range, label):
    x0, y0, x1, y1 = box
    if y_range is None:
      lo, hi = float(np.min(values)), float(np.max(values))
      pad = max(0.05 * (hi - lo), 0.1)
      lo, hi = lo - pad, hi + pad
    else:
      lo, hi = y_range
    points = []
    for x, y in zip(x_values, values):
      px = x0 + (x - x_values.min()) / max(np.ptp(x_values), 1.0) * (x1 - x0)
      py = y1 - (y - lo) / max(hi - lo, 1e-8) * (y1 - y0)
      points.append((int(px), int(py)))
    if len(points) > 1:
      draw.line(points, fill=color, width=3)
    for px, py in points:
      draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
    draw.text((x0 + 8, y0 + 8), label, fill=color)
    draw.text((8, y0), f"{hi:.2f}", fill="black")
    draw.text((8, y1 - 12), f"{lo:.2f}", fill="black")

  for index, (title, values, color, y_range, reference) in enumerate(panels):
    top = 75 + index * 300
    box = (left, top + 35, right, top + panel_h)
    draw.text((left, top), title, fill="black")
    draw.rectangle(box, outline=(120, 120, 120), width=1)
    if index == 1:
      pos = np.asarray([r["positive_logp_mean"] for r in rows], dtype=float)
      neg = np.asarray([r["negative_logp_mean"] for r in rows], dtype=float)
      lo = float(min(np.min(pos), np.min(neg)))
      hi = float(max(np.max(pos), np.max(neg)))
      pad = max(0.05 * (hi - lo), 0.1)
      draw_series(box, pos, (42, 157, 143), (lo - pad, hi + pad),
                  "positive log p")
      draw_series(box, neg, (232, 131, 76), (lo - pad, hi + pad),
                  "negative log p")
    else:
      draw_series(box, values, color, y_range, title)
    if reference is not None:
      lo, hi = y_range if y_range is not None else (
          float(np.min(values)), float(np.max(values)))
      if lo <= reference <= hi:
        py = box[3] - (reference - lo) / max(hi - lo, 1e-8) * (
            box[3] - box[1])
        draw.line((box[0], int(py), box[2], int(py)),
                  fill=(100, 100, 100), width=1)
    for x in x_values:
      px = box[0] + (x - x_values.min()) / max(np.ptp(x_values), 1.0) * (
          box[2] - box[0])
      draw.text((int(px) - 12, box[3] + 6), str(int(x)), fill="black")

  draw.text((width // 2 - 70, height - 25), "checkpoint iteration", fill="black")
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  tmp = path + ".tmp.png"
  image.save(tmp)
  os.replace(tmp, path)


def _plot(path: str, rows: list[dict]) -> None:
  try:
    import matplotlib
  except ModuleNotFoundError:
    _plot_pil(path, rows)
    return

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  x = np.asarray([r["iteration"] for r in rows], dtype=float)
  acc = np.asarray([r["rank_accuracy"] for r in rows], dtype=float)
  se = np.asarray([r["rank_accuracy_se"] for r in rows], dtype=float)
  pos = np.asarray([r["positive_logp_mean"] for r in rows], dtype=float)
  neg = np.asarray([r["negative_logp_mean"] for r in rows], dtype=float)
  margin = np.asarray([r["margin_mean"] for r in rows], dtype=float)

  fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.2), sharex=True)
  axes[0].plot(x, acc, color="#4C9BE8", lw=2.0, marker="o",
               label=r"$P[\log p(g^+) > \log p(g^-)]$")
  axes[0].fill_between(x, acc - 1.96 * se, acc + 1.96 * se,
                       color="#4C9BE8", alpha=0.2, label="approx. 95% CI")
  axes[0].axhline(0.5, color="0.45", lw=1.0, ls="--", label="chance = 50%")
  axes[0].set_ylim(0.0, 1.0)
  axes[0].set_ylabel("rank accuracy")
  axes[0].legend(fontsize=8)

  axes[1].plot(x, pos, color="#2A9D8F", lw=1.8, marker="o",
               label=r"positive $\log p$")
  axes[1].plot(x, neg, color="#E8834C", lw=1.8, marker="o",
               label=r"negative $\log p$")
  axes[1].set_ylabel("mean raw log p")
  axes[1].legend(fontsize=8)

  axes[2].plot(x, margin, color="#A84CE8", lw=1.8, marker="o",
               label=r"mean $\log p(g^+) - \log p(g^-)$")
  axes[2].axhline(0.0, color="0.45", lw=1.0, ls="--")
  axes[2].set_ylabel("mean log-p margin")
  axes[2].set_xlabel("checkpoint iteration")
  axes[2].legend(fontsize=8)

  for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
  fig.suptitle(
      "Allegro NF ranking: same-rollout future goal vs other-rollout goal",
      fontsize=11,
      fontweight="bold",
  )
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint-dir", required=True)
  parser.add_argument("--output-csv", required=True)
  parser.add_argument("--output-plot", required=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--min-samples", type=int, default=256)
  parser.add_argument(
      "--iterations", default="",
      help="comma-separated checkpoint iterations; empty evaluates all")
  parser.add_argument(
      "--future-discount", type=float, default=0.99,
      help="truncated-geometric future-goal discount (training default 0.99)")
  parser.add_argument("--num-steps", type=int, default=0,
                      help="0 uses the training episode horizon")
  parser.add_argument("--pipeline", choices=("gpu", "cpu"), default="gpu")
  parser.add_argument(
      "--random-actions", action="store_true",
      help="Collect rollouts with Uniform[-1,1] actions instead of π.")
  args = parser.parse_args()

  paths = sorted(
      glob.glob(os.path.join(args.checkpoint_dir, "ckpt_iter_*.pkl")),
      key=_iteration,
  )
  if args.iterations.strip():
    requested = {
        int(value.strip()) for value in args.iterations.split(",")
        if value.strip()
    }
    paths = [path for path in paths if _iteration(path) in requested]
    found = {_iteration(path) for path in paths}
    if found != requested:
      raise FileNotFoundError(
          f"missing requested checkpoint iterations: {sorted(requested-found)}")
  if not paths:
    raise FileNotFoundError(f"no ckpt_iter_*.pkl in {args.checkpoint_dir}")

  # Isaac Gym must initialize before JAX.
  flags = ckpt_utils._load_flags(paths[0])
  env_kwargs = ckpt_utils._load_env_kwargs(
      flags, args.num_steps, args.seed, args.pipeline)
  env_kwargs["num_envs"] = 2
  env_kwargs["enable_cameras"] = False
  env = ckpt_utils._build_env(env_kwargs)
  include_qd = bool(
      flags.get('isaacgym_control_sanity_goal_include_qd', False))
  if include_qd:
    if int(env.goal_dim) != int(env.obs_dim):
      raise ValueError(
          "qd-goal run requires achieved goal to span the full state: "
          f"obs_dim={env.obs_dim}, goal_dim={env.goal_dim}")
    if int(env.action_dim) * 2 != int(env.goal_dim):
      raise ValueError(
          "qd-goal run requires goal=[q,qd] with action=q targets: "
          f"action_dim={env.action_dim}, goal_dim={env.goal_dim}")
  goal_state_indices = _achieved_goal_indices(env)
  print(
      f"[nf-rank] achieved goal = state indices "
      f"{goal_state_indices.tolist()} (obs_dim={env.obs_dim}, "
      f"goal_dim={env.goal_dim}, action_dim={env.action_dim}, "
      f"goal_include_qd={include_qd})",
      flush=True,
  )

  import jax

  first_ckpt = _load_ckpt(paths[0])
  act, _, _, _ = ckpt_utils._build_actor(
      env, first_ckpt, flags, deterministic=False)
  reward_fn, _ = ckpt_utils._build_nf_reward(env, first_ckpt, flags)
  device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]

  results = []
  for path in paths:
    ckpt = _load_ckpt(path)
    iteration = int(ckpt.get("iteration", _iteration(path)))
    policy_params = jax.device_put(ckpt["policy_params"], device)
    nf_key = "q_params_ema" if "q_params_ema" in ckpt else "q_params"
    nf_params = jax.device_put(ckpt[nf_key], device)
    goal_mean, goal_std = ckpt_utils._goal_stats_from_learner(
        path, iteration, int(env.goal_dim))

    key = jax.random.PRNGKey(int(args.seed) + iteration)
    packed, actions, dones, _ = _rollout(
        env, act, policy_params, key, int(env.max_episode_steps),
        random_actions=bool(args.random_actions))
    rng = np.random.default_rng(int(args.seed) + 1_000_003 + iteration)
    anchors, anchor_actions, pos_goals, neg_goals, deltas = _make_pairs(
        packed, actions, dones, int(env.obs_dim), goal_state_indices,
        float(args.future_discount), rng)
    if len(anchors) < int(args.min_samples):
      raise RuntimeError(
          f"iter={iteration}: only {len(anchors)} valid comparisons; "
          f"need at least {args.min_samples}")

    pos, neg = _score_pairs(
        reward_fn, nf_params, anchors, anchor_actions, pos_goals, neg_goals,
        goal_mean, goal_std)
    correct = pos > neg
    ties = pos == neg
    accuracy = float(np.mean(correct))
    se = float(np.sqrt(accuracy * (1.0 - accuracy) / len(correct)))
    row = {
        "iteration": iteration,
        "num_samples": len(correct),
        "rank_accuracy": accuracy,
        "rank_accuracy_se": se,
        "tie_fraction": float(np.mean(ties)),
        "positive_logp_mean": float(np.mean(pos)),
        "positive_logp_std": float(np.std(pos)),
        "negative_logp_mean": float(np.mean(neg)),
        "negative_logp_std": float(np.std(neg)),
        "margin_mean": float(np.mean(pos - neg)),
        "margin_median": float(np.median(pos - neg)),
        "delta1_fraction": float(np.mean(deltas == 1)),
        "future_delta_mean": float(np.mean(deltas)),
        "future_delta_median": float(np.median(deltas)),
        "nf_params_source": nf_key,
    }
    results.append(row)
    print(
        f"[nf-rank] iter={iteration:04d} n={len(correct)} "
        f"acc={accuracy:.3f} pos={np.mean(pos):+.3f} "
        f"neg={np.mean(neg):+.3f} margin={np.mean(pos-neg):+.3f} "
        f"collector={'random' if args.random_actions else 'pi'}",
        flush=True,
    )

  _write_csv(args.output_csv, results)
  _plot(args.output_plot, results)
  meta_path = os.path.splitext(args.output_csv)[0] + "_meta.json"
  with open(meta_path, "w", encoding="utf-8") as fh:
    json.dump({
        "definition": "P[log p(g_pos|s,a) > log p(g_neg|s,a)]",
        "positive": "same-rollout truncated-geometric future achieved goal",
        "negative": "random achieved goal timestep from other rollout",
        "future_discount": float(args.future_discount),
        "goal_state_indices": goal_state_indices.tolist(),
        "symmetric_rollouts": True,
        "exclude_positive_reset_crossings": True,
        "seed": int(args.seed),
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "collector": (
            "random_uniform[-1,1]" if args.random_actions
            else "learned_stochastic"),
    }, fh, indent=2)
  print(f"[nf-rank] wrote {args.output_csv}", flush=True)
  print(f"[nf-rank] wrote {args.output_plot}", flush=True)


if __name__ == "__main__":
  main()
