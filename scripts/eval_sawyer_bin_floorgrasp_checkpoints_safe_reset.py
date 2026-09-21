#!/usr/bin/env python3
"""Evaluate every Sawyer-bin floor-grasp checkpoint after a healthy reset."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import sgcrl_jax_acme_compat  # noqa: F401,E402

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from acme import specs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

import contrastive  # noqa: E402
from contrastive import ppo_learner  # noqa: E402
from contrastive import utils as contrastive_utils  # noqa: E402
import env_utils  # noqa: E402
from ppo_contrastive import fixed_goal_dict  # noqa: E402
from scripts import plot_builderbench_train_success1000 as plot_base  # noqa: E402

STEPS_PER_ITER = 1024
MAX_EPISODE_STEPS = 150
SMOOTH_WINDOW = 5
EVAL_SEED = 910_000
DEFAULT_EPISODES = 10
_CKPT_RE = re.compile(r"ckpt_iter_(\d+)\.pkl$")


def _checkpoint_files(checkpoint_dir: str) -> list[tuple[int, str]]:
  out = []
  for path in glob.glob(os.path.join(checkpoint_dir, "ckpt_iter_*.pkl")):
    match = _CKPT_RE.search(os.path.basename(path))
    if match:
      out.append((int(match.group(1)), path))
  out.sort()
  return out


def _body_name(env, geom_id: int) -> str:
  body_id = int(env.model.geom_bodyid[int(geom_id)])
  return str(env.model.body_id2name(body_id) or "").lower()


def _gripper_object_contacts(env) -> int:
  count = 0
  for i in range(int(env.data.ncon)):
    contact = env.data.contact[i]
    a = _body_name(env, contact.geom1)
    b = _body_name(env, contact.geom2)
    a_grip = any(key in a for key in ("claw", "pad", "hand"))
    b_grip = any(key in b for key in ("claw", "pad", "hand"))
    if (a_grip and "obj" in b) or (b_grip and "obj" in a):
      count += 1
  return count


def _reset_metrics(env, obs: np.ndarray) -> dict[str, float]:
  hand = np.asarray(obs[:3], dtype=np.float64)
  grip = float(obs[3])
  obj = np.asarray(obs[4:7], dtype=np.float64)
  return {
      "hand_object_distance": float(np.linalg.norm(hand - obj)),
      "hand_object_xy_error": float(np.linalg.norm(hand[:2] - obj[:2])),
      "hand_object_dz": float(hand[2] - obj[2]),
      "gripper_width": grip,
      "contact_count": float(_gripper_object_contacts(env)),
  }


def _healthy(metrics: dict[str, float]) -> bool:
  return (
      metrics["hand_object_distance"] <= 0.06
      and metrics["hand_object_xy_error"] <= 0.04
      and 0.0 <= metrics["hand_object_dz"] <= 0.06
      and 0.35 <= metrics["gripper_width"] <= 0.45
      and metrics["contact_count"] >= 1
  )


def _safe_reset(env, max_attempts: int) -> tuple[np.ndarray, dict[str, float], int]:
  for attempt in range(1, max_attempts + 1):
    obs = np.asarray(env.reset(), dtype=np.float32)
    metrics = _reset_metrics(env, obs)
    if _healthy(metrics):
      return obs, metrics, attempt
  raise RuntimeError(
      f"no healthy floor-grasp reset in {max_attempts} attempts; "
      f"last metrics={metrics}")


def _build_networks(seed: int):
  probe_env, obs_dim = contrastive_utils.make_environment(
      "sawyer_bin", start_index=0, end_index=-1, seed=seed,
      fixed_start_end=fixed_goal_dict["sawyer_bin"])
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env
  cfg = contrastive.ContrastiveConfig()
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=cfg.repr_dim,
      repr_norm=cfg.repr_norm,
      twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      actor_min_std=float(cfg.ppo_actor_min_std),
  )
  return networks


def _make_env():
  np.random.seed(EVAL_SEED)
  env, _, _ = env_utils.load(
      "sawyer_bin", fixed_start_end=fixed_goal_dict["sawyer_bin"],
      seed=EVAL_SEED, randomize_init=True)
  return env


def _make_policy_mode(networks):
  @jax.jit
  def policy_mode(params, packed_obs):
    distribution = networks.policy_network.apply(params, packed_obs)
    return networks.sample_eval(distribution, jax.random.PRNGKey(0))
  return policy_mode


def _capture_initializations(
    env, episodes: int, max_reset_attempts: int
) -> list[dict]:
  initializations = []
  for episode in range(episodes):
    seed = EVAL_SEED + episode
    np.random.seed(seed)
    obs, metrics, attempts = _safe_reset(env, max_reset_attempts)
    frame = np.asarray(
        env.render(
            offscreen=True, camera_name="corner2", resolution=(640, 480)),
        dtype=np.uint8)
    initializations.append({
        "episode": episode,
        "seed": seed,
        "obs": obs.copy(),
        "sim_state": env.sim.get_state(),
        "mocap_pos": env.data.mocap_pos.copy(),
        "mocap_quat": env.data.mocap_quat.copy(),
        "metrics": metrics,
        "reset_attempts": attempts,
        "frame": frame,
    })
    print(
        f"[safe-eval] init={episode} seed={seed} attempts={attempts} "
        f"d={metrics['hand_object_distance']:.3f} "
        f"xy={metrics['hand_object_xy_error']:.3f} "
        f"dz={metrics['hand_object_dz']:.3f} "
        f"grip={metrics['gripper_width']:.3f} "
        f"contacts={int(metrics['contact_count'])}", flush=True)
  return initializations


def _restore_initialization(env, initialization: dict) -> np.ndarray:
  env.sim.set_state(initialization["sim_state"])
  env.data.mocap_pos[:] = initialization["mocap_pos"]
  env.data.mocap_quat[:] = initialization["mocap_quat"]
  env.sim.forward()
  if hasattr(env, "curr_path_length"):
    env.curr_path_length = 0
  return np.asarray(env._get_obs(), dtype=np.float32)


def _evaluate_episode(policy_params, policy_mode, env, initialization):
  obs = _restore_initialization(env, initialization)
  success = False
  first_success_step = -1
  for step in range(MAX_EPISODE_STEPS):
    action = np.asarray(
        policy_mode(policy_params, jnp.asarray(obs)[None]))[0].astype(np.float32)
    action = np.clip(
        np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    obs, reward, done, _ = env.step(action)
    obs = np.asarray(obs, dtype=np.float32)
    if float(reward) > 0.0 and not success:
      success = True
      first_success_step = step + 1
    if done:
      break
  return float(success), first_success_step


def _write_initialization_csv(path: str, initializations: list[dict]) -> None:
  rows = []
  for initialization in initializations:
    rows.append({
        "episode": initialization["episode"],
        "seed": initialization["seed"],
        "reset_attempts": initialization["reset_attempts"],
        "rejected_resets": initialization["reset_attempts"] - 1,
        **initialization["metrics"],
    })
  _write_csv(path, rows)


def _plot_initializations(path: str, initializations: list[dict]) -> None:
  columns = 5
  rows = int(np.ceil(len(initializations) / columns))
  fig, axes = plt.subplots(rows, columns, figsize=(18, 3.8 * rows), squeeze=False)
  fig.suptitle(
      "Ten accepted Sawyer-bin floor-grasp initialization states",
      fontsize=14, fontweight="bold")
  for ax, initialization in zip(axes.flat, initializations):
    metrics = initialization["metrics"]
    ax.imshow(initialization["frame"])
    ax.set_title(
        f"episode {initialization['episode']} · seed {initialization['seed']}\n"
        f"d={metrics['hand_object_distance']:.3f}  "
        f"xy={metrics['hand_object_xy_error']:.3f}  "
        f"dz={metrics['hand_object_dz']:.3f}\n"
        f"grip={metrics['gripper_width']:.3f}  "
        f"contacts={int(metrics['contact_count'])}  "
        f"attempts={initialization['reset_attempts']}",
        fontsize=9)
    ax.axis("off")
  for ax in axes.flat[len(initializations):]:
    ax.axis("off")
  fig.tight_layout()
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def _write_csv(path: str, rows: list[dict]) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  os.replace(tmp, path)


def _plot(path: str, rows: list[dict], run_seed: int) -> None:
  xs = [int(row["global_step"]) for row in rows]
  ys = [float(row["success_rate"]) for row in rows]
  fig, ax = plt.subplots(figsize=(9.0, 4.8))
  plot_base._plot_eval_smoothed(
      ax, xs, ys, color="#6A3D9A",
      label=f"Centered rolling success (window={SMOOTH_WINDOW})",
      window=SMOOTH_WINDOW, linewidth=2.6)
  ax.set_ylim(-0.05, 1.05)
  ax.set_xlabel("Environment steps")
  ax.set_ylabel(f"Eval success over 10 episodes (rolling mean, window={SMOOTH_WINDOW})")
  ax.set_title(
      f"Sawyer bin floor-grasp · seed {run_seed} · "
      "ten healthy-reset episodes")
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(plot_base._fmt_steps))
  ax.grid(axis="y", linestyle=":", alpha=0.4)
  ax.spines[["top", "right"]].set_visible(False)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint_dir", required=True)
  parser.add_argument("--csv_output", required=True)
  parser.add_argument("--plot_output", required=True)
  parser.add_argument("--init_csv_output", required=True)
  parser.add_argument("--init_plot_output", required=True)
  parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
  parser.add_argument("--run_seed", type=int, required=True)
  parser.add_argument("--max_reset_attempts", type=int, default=100)
  args = parser.parse_args()

  checkpoints = _checkpoint_files(args.checkpoint_dir)
  if not checkpoints:
    raise FileNotFoundError(f"no checkpoint files under {args.checkpoint_dir}")
  print(f"[safe-eval] checkpoints={len(checkpoints)} deterministic=True "
        f"episodes_per_checkpoint={args.episodes} eval_seed={EVAL_SEED}")
  networks = _build_networks(EVAL_SEED)
  policy_mode = _make_policy_mode(networks)
  env = _make_env()
  initializations = _capture_initializations(
      env, args.episodes, args.max_reset_attempts)
  _write_initialization_csv(args.init_csv_output, initializations)
  _plot_initializations(args.init_plot_output, initializations)
  print(f"→ {args.init_csv_output}")
  print(f"→ {args.init_plot_output}")

  rows = []
  for index, (iteration, path) in enumerate(checkpoints, 1):
    checkpoint = ppo_learner.load_checkpoint(path)
    outcomes = [
        _evaluate_episode(
            checkpoint["policy_params"], policy_mode, env, initialization)
        for initialization in initializations
    ]
    successes = int(sum(success for success, _ in outcomes))
    success_steps = [
        step for success, step in outcomes if success > 0.0]
    row = {
        "iteration": iteration,
        "global_step": int(checkpoint.get(
            "global_step", iteration * STEPS_PER_ITER)),
        "success_rate": successes / args.episodes,
        "successes": successes,
        "episodes": args.episodes,
        "mean_success_step": (
            float(np.mean(success_steps)) if success_steps else -1.0),
        "episode_successes": " ".join(
            str(int(success)) for success, _ in outcomes),
        "total_reset_attempts": sum(
            initialization["reset_attempts"]
            for initialization in initializations),
        "rejected_resets": sum(
            initialization["reset_attempts"] - 1
            for initialization in initializations),
        "checkpoint": path,
    }
    rows.append(row)
    _write_csv(args.csv_output, rows)
    print(f"[safe-eval] {index}/{len(checkpoints)} iter={iteration} "
          f"success={successes}/{args.episodes} "
          f"rate={row['success_rate']:.2f}", flush=True)
  _plot(args.plot_output, rows, args.run_seed)
  print(f"→ {args.csv_output}")
  print(f"→ {args.plot_output}")


if __name__ == "__main__":
  main()
