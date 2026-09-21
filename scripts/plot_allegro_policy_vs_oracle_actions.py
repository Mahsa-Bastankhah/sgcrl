#!/usr/bin/env python3
"""Compare deterministic policy actions with absolute-target oracle actions."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts import allegro_kuka_throw_ckpt_video as vid  # noqa: E402


_ACTION_NAMES = (
    "index base", "index proximal",
    "middle base", "middle proximal",
    "ring base", "ring proximal",
    "thumb base", "thumb proximal",
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--joints", default="1,7",
      help="Comma-separated action dimensions to plot")
  parser.add_argument("--pipeline", choices=("gpu", "cpu"), default="gpu")
  args = parser.parse_args()

  joints = tuple(int(value) for value in args.joints.split(","))
  flags = vid._load_flags(args.checkpoint)
  env_kwargs = vid._load_env_kwargs(flags, 0, args.seed, args.pipeline)
  env_kwargs["enable_cameras"] = False
  env = vid._build_env(env_kwargs)
  ckpt = vid._load_ckpt(args.checkpoint)
  act, policy_params, iteration, _ = vid._build_actor(
      env, ckpt, flags, deterministic=True)

  import jax
  import jax.numpy as jnp
  import torch

  if int(env.action_dim) != int(env.goal_dim):
    raise RuntimeError(
        f"oracle a=g requires action_dim==goal_dim, got "
        f"{env.action_dim}!={env.goal_dim}")
  for joint in joints:
    if joint < 0 or joint >= int(env.action_dim):
      raise ValueError(f"joint {joint} outside action_dim={env.action_dim}")

  vid._set_reset_seed(args.seed)
  obs_t = env.reset()
  oracle = np.asarray(
      env.normalized_goal_action()[0].detach().cpu().numpy(), dtype=np.float32)
  key = jax.random.PRNGKey(args.seed)
  rows = []

  for t in range(int(env.max_episode_steps)):
    obs = np.asarray(
        obs_t.detach().cpu().numpy(), dtype=np.float32).reshape(1, -1)
    key, subkey = jax.random.split(key)
    policy = np.asarray(
        act(policy_params, jnp.asarray(obs), subkey),
        dtype=np.float32).reshape(-1).copy()
    row = {"t": t}
    for j in range(int(env.action_dim)):
      row[f"policy_action_{j}"] = float(policy[j])
      row[f"oracle_action_{j}"] = float(oracle[j])
    rows.append(row)
    obs_t, _, done_t = env.step(
        torch.as_tensor(policy[None], device=env.device))
    done = float(
        np.asarray(done_t.detach().cpu().numpy()).reshape(-1)[0])
    if done >= 0.5:
      break

  os.makedirs(args.out_dir, exist_ok=True)
  stem = f"policy_vs_oracle_actions_iter{iteration:07d}_seed{args.seed}"
  csv_path = os.path.join(args.out_dir, stem + ".csv")
  with open(csv_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  from PIL import Image, ImageDraw

  width = 1200
  panel_h = 360
  margin_l, margin_r = 95, 35
  plot_top, plot_bottom = 68, panel_h - 55
  canvas = Image.new(
      "RGB", (width, panel_h * len(joints)), (250, 250, 250))
  draw = ImageDraw.Draw(canvas)
  ts = np.asarray([row["t"] for row in rows], dtype=np.float64)

  def _xy(t, value, y_offset):
    x = margin_l + int(
        (float(t) / max(float(ts[-1]), 1.0))
        * (width - margin_l - margin_r))
    y = y_offset + plot_bottom - int(
        ((float(value) + 1.0) / 2.0) * (plot_bottom - plot_top))
    return x, y

  for panel, joint in enumerate(joints):
    y_offset = panel * panel_h
    policy_y = np.asarray(
        [row[f"policy_action_{joint}"] for row in rows])
    oracle_y = np.asarray(
        [row[f"oracle_action_{joint}"] for row in rows])
    label = (
        _ACTION_NAMES[joint]
        if joint < len(_ACTION_NAMES) else f"action {joint}")
    draw.text(
        (margin_l, y_offset + 12),
        f"{label} (action {joint}) — checkpoint iteration {iteration}",
        fill=(20, 20, 20))
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
      _, y = _xy(0, value, y_offset)
      draw.line(
          [(margin_l, y), (width - margin_r, y)],
          fill=(220, 220, 220), width=1)
      draw.text((45, y - 7), f"{value:+.1f}", fill=(70, 70, 70))
    policy_pts = [
        _xy(t, value, y_offset) for t, value in zip(ts, policy_y)]
    oracle_pts = [
        _xy(t, value, y_offset) for t, value in zip(ts, oracle_y)]
    draw.line(policy_pts, fill=(51, 102, 204), width=4)
    draw.line(oracle_pts, fill=(220, 57, 18), width=4)
    draw.text(
        (margin_l, y_offset + panel_h - 35),
        "blue: deterministic policy    red: oracle a=g_normalized",
        fill=(30, 30, 30))
    draw.text(
        (width - 160, y_offset + panel_h - 35),
        "rollout timestep", fill=(70, 70, 70))

  png_path = os.path.join(args.out_dir, stem + ".png")
  canvas.save(png_path)

  print(
      f"[policy_vs_oracle] iteration={iteration} steps={len(rows)} "
      f"oracle={np.round(oracle, 4).tolist()}", flush=True)
  print(f"[policy_vs_oracle] wrote {csv_path}", flush=True)
  print(f"[policy_vs_oracle] wrote {png_path}", flush=True)


if __name__ == "__main__":
  main()
