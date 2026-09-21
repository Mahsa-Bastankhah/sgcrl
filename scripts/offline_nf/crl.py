"""Offline production PPO CRL on the same fixtures as the NF diagnostic."""
from __future__ import annotations

import os
import pickle
import time
from typing import Any

import numpy as np

from scripts.offline_nf.core import (
    DELTA_BINS,
    EpisodeSet,
    _annotate_early_steps,
    _load_pyplot,
    assert_same_heldout_fixture,
    load_offline_split,
    plot_delta_stratified,
    regenerate_heldout_fixture,
    stratify_by_delta,
    write_csv,
    write_json,
)

HIDDEN_LAYER_SIZES = (256, 256, 256, 256, 256, 256)
REPR_DIM = 64
LEARNING_RATE = 3e-4
LOGSUMEXP_PENALTY = 0.01


def make_crl_train_parser(description: str):
  from scripts.offline_nf.core import make_train_parser

  parser = make_train_parser(description)
  parser.add_argument(
      "--nf_fixture_path", required=True,
      help="existing NF fixed_heldout_eval_fixture.npz (verified, never replaced)")
  parser.add_argument(
      "--nf_metrics_dir", default="",
      help="existing NF sweep dir for overlay plots (read-only)")
  parser.add_argument(
      "--overlay_dir", default="",
      help="write NF vs CRL overlay figures here")
  return parser


def _dummy_spec(state_dim: int, action_dim: int, goal_dim: int):
  from acme import specs

  packed = int(state_dim + goal_dim)
  return specs.EnvironmentSpec(
      observations=specs.Array(
          shape=(packed,), dtype=np.float32, name="observation"),
      actions=specs.BoundedArray(
          shape=(int(action_dim),), dtype=np.float32,
          minimum=-1.0, maximum=1.0, name="action"),
      rewards=specs.Array(shape=(), dtype=np.float32, name="reward"),
      discounts=specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0.0, maximum=1.0,
          name="discount"),
  )


def plot_crl_learning(
    path: str,
    rows: list[dict[str, Any]],
    label: str,
    *,
    domain_label: str,
    batch_size: int = 256,
    early_max: int | None = None,
) -> None:
  plt = _load_pyplot()
  selected = rows
  suffix = ""
  if early_max is not None:
    selected = [row for row in rows if float(row["step"]) <= early_max]
    suffix = f" — early 0–{early_max} updates"
  x = np.asarray([row["step"] for row in selected], dtype=float)
  fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
  axes[0, 0].plot(
      x, [row["train_crl_loss"] for row in selected], marker="o",
      label="train InfoNCE + 0.01 logsumexp²")
  axes[0, 0].set_ylabel("CRL loss")
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
      x, [row["task_goal_score"] for row in selected], marker="o",
      label="task-goal φ·ψ")
  axes[1, 1].plot(
      x, [row["shuffled_context_score"] for row in selected], marker="s",
      label="shuffled-context φ·ψ")
  axes[1, 1].set_ylabel("mean φ·ψ")
  axes[1, 1].legend(fontsize=8)
  for axis in axes.flat:
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("gradient updates")
  if early_max is not None:
    _annotate_early_steps(axes.flat, batch_size)
    axes[1, 0].set_xlim(0, early_max)
  fig.suptitle(f"Offline {domain_label} CRL learning — {label}{suffix}")
  fig.text(
      0.5, 0.005,
      "Step 0 = pre-training; first post-train evaluation: "
      f"250 updates = {250 * batch_size:,} replay draws "
      f"(batch {batch_size}). Score is raw φ(s,a)·ψ(g).",
      ha="center", fontsize=9)
  fig.tight_layout(rect=(0, 0.035, 1, 1))
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def plot_crl_scaling(
    path: str, final_rows: list[dict[str, Any]], *, domain_label: str,
) -> None:
  plt = _load_pyplot()
  x = np.asarray(
      [row["train_transitions"] for row in final_rows], dtype=float)
  fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
  axes[0].plot(x, [row["train_crl_loss"] for row in final_rows], marker="o")
  axes[0].set_ylabel("final train CRL loss")
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
  fig.suptitle(f"Offline {domain_label} CRL: final dataset-size comparison")
  fig.tight_layout()
  tmp = path + ".tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, path)
  plt.close(fig)


def _check_step0_chance(metrics: dict[str, Any]) -> None:
  binary = float(metrics["binary_accuracy"])
  cat = float(metrics["categorical_accuracy"])
  chance = float(metrics["categorical_chance"])
  print(
      f"[step0] pre-train binary={binary:.4f} (chance 0.5) "
      f"cat={cat:.4f} (chance {chance:.5f})",
      flush=True)
  if not 0.40 <= binary <= 0.60:
    raise RuntimeError(
        f"step-0 binary accuracy {binary:.4f} is far from chance 0.5")
  if not 0.005 <= cat <= 0.10:
    raise RuntimeError(
        f"step-0 categorical accuracy {cat:.4f} is far from chance {chance:.5f}")


def run_crl_training(
    args, *, domain_label: str, expected_dims: tuple[int, int, int],
) -> None:
  """Train production PPO CRL on a frozen random-policy episode dataset."""
  import jax
  import jax.numpy as jnp
  import optax
  from acme.jax import utils as acme_utils
  from contrastive import config as contrastive_config
  from contrastive import networks as contrastive_networks
  from contrastive.ppo_learner import make_crl_update_fn

  if args.smoke_test:
    args.dataset_sizes = args.dataset_sizes.split(",", maxsplit=1)[0]
    args.updates = 1
    args.eval_every = 1
    print(
        "[smoke] one real CRL update; fixture sizes kept so the NF "
        "held-out tensors can be verified",
        flush=True)

  os.makedirs(args.output_dir, exist_ok=True)
  prepared = load_offline_split(args, expected_dims=expected_dims)
  regenerated = regenerate_heldout_fixture(prepared, args)
  fixture = assert_same_heldout_fixture(args.nf_fixture_path, regenerated)
  np.savez(
      os.path.join(args.output_dir, "fixed_heldout_eval_fixture.npz"),
      **fixture)
  write_json(os.path.join(args.output_dir, "fixture_verification.json"), {
      "nf_fixture_path": os.path.abspath(args.nf_fixture_path),
      "fixture_seed": int(args.seed) + 91_003,
      "verified_equal": True,
      "scorer": "phi_dot_psi",
      "goal_preprocessing": "raw achieved goals (no NF mean/std or noise)",
  })

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

  spec = _dummy_spec(state_dim, action_dim, goal_dim)
  dummy_obs = acme_utils.add_batch_dim(acme_utils.zeros_like(spec.observations))
  dummy_action = acme_utils.add_batch_dim(acme_utils.zeros_like(spec.actions))
  print(
      f"[crl] dummy init obs={tuple(dummy_obs.shape)} "
      f"action={tuple(dummy_action.shape)} hidden=6x256 repr={REPR_DIM}",
      flush=True)

  networks = contrastive_networks.make_networks(
      spec,
      obs_dim=state_dim,
      repr_dim=REPR_DIM,
      repr_norm=False,
      twin_q=False,
      hidden_layer_sizes=HIDDEN_LAYER_SIZES,
      state_only=False,
  )
  crl_config = contrastive_config.ContrastiveConfig(
      env_name="",
      obs_dim=state_dim,
      start_index=0,
      end_index=-1,
      ppo_norm_obs=False,
      twin_q=False,
      repr_norm=False,
      repr_dim=REPR_DIM,
      hidden_layer_sizes=HIDDEN_LAYER_SIZES,
      learning_rate=LEARNING_RATE,
      ppo_crl_sf_perturb_prob=0.0,
      ppo_crl_s_perturb_prob=0.0,
      ppo_crl_grad_reg_coef=0.0,
  )
  optimizer = optax.adam(LEARNING_RATE)
  update = make_crl_update_fn(
      networks, optimizer, backward=False, config=crl_config)

  @jax.jit
  def phi_psi(params, states, actions, goals):
    packed = jnp.concatenate([states, goals], axis=-1)
    _, phi, psi = networks.q_network.apply(params, packed, actions)
    return jnp.sum(phi * psi, axis=-1)

  @jax.jit
  def infonce_loss(params, states, actions, goals):
    packed = jnp.concatenate([states, goals], axis=-1)
    logits, _, _ = networks.q_network.apply(params, packed, actions)
    labels = jnp.eye(states.shape[0])
    nce = optax.softmax_cross_entropy(logits=logits, labels=labels)
    penalty = LOGSUMEXP_PENALTY * jax.nn.logsumexp(logits, axis=1) ** 2
    return jnp.mean(nce + penalty)

  def evaluate(params, train_batch) -> dict[str, Any]:
    def scores(states, actions, goals):
      return np.asarray(phi_psi(
          params, jnp.asarray(states), jnp.asarray(actions),
          jnp.asarray(goals)))

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
    task_scores = scores(
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
    train_loss = float(np.asarray(infonce_loss(
        params,
        jnp.asarray(train_batch["obs"][:, :state_dim]),
        jnp.asarray(train_batch["action"]),
        jnp.asarray(train_batch["obs"][:, state_dim:]))))
    metrics: dict[str, Any] = {
        "train_crl_loss": train_loss,
        "binary_accuracy": float(np.mean(binary_correct)),
        "binary_tie_fraction": float(np.mean(pos == neg)),
        "categorical_accuracy": float(np.mean(cat_correct)),
        "categorical_chance": 1.0 / float(k),
        "positive_score": float(pos.mean()),
        "negative_score": float(neg.mean()),
        "binary_margin": float((pos - neg).mean()),
        "shuffled_context_score": float(shuffled_context.mean()),
        "task_goal_score": float(task_scores.mean()),
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

  labels = [int(value) for value in args.dataset_sizes.split(",")]
  all_rows: list[dict[str, Any]] = []
  final_rows: list[dict[str, Any]] = []
  run_metadata = {
      "dataset_metadata": metadata,
      "split": {
          "semantics": (
              "shared global 90/10 episode-index split; IDs were not stored "
              "in the dataset, so they are recreated as "
              "arange(floor(0.9*N)) / remaining — identical to the NF "
              "offline trainer"),
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
          "nf_fixture_path": os.path.abspath(args.nf_fixture_path),
          "fixed_across_steps_and_dataset_sizes": True,
          "scorer": "phi(s,a) · psi(g)",
      },
      "binary_eval": {
          "positive": "same-episode gamma=.99 truncated-geometric future",
          "negative": "random achieved state from a different held-out episode",
          "chance": 0.5,
          "scorer": "phi(s,a) · psi(g)",
      },
      "recipe": {
          "method": "ppo_crl",
          "not": ["nf", "td_infonce"],
          "architecture": "6x256 ReLU ResidualMLP",
          "repr_dim": REPR_DIM,
          "repr_norm": False,
          "twin_q": False,
          "state_only": False,
          "ppo_norm_obs": False,
          "loss": "forward InfoNCE + 0.01 logsumexp^2",
          "lr": LEARNING_RATE,
          "perturb": False,
          "grad_reg": False,
          "goal_preprocessing": "raw achieved goals",
          "discount": float(args.discount),
          "batch_size": int(args.batch_size),
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
    train_fixture = train.sample(
        int(args.eval_nll_samples),
        np.random.default_rng(int(args.seed) + 31_001 + model_index))
    key = jax.random.PRNGKey(int(args.seed))
    key, init_key = jax.random.split(key)
    params = networks.q_network.init(init_key)
    opt_state = optimizer.init(params)
    model_rows: list[dict[str, Any]] = []
    interval_losses: list[float] = []
    train_rng = np.random.default_rng(int(args.seed) + 101 + model_index)
    for step_i in range(int(args.updates) + 1):
      if step_i % int(args.eval_every) == 0 or step_i == int(args.updates):
        metrics = evaluate(params, train_fixture)
        if step_i == 0:
          _check_step0_chance(metrics)
        row = {
            "step": step_i,
            "dataset_size_label": label,
            "nominal_subset_transitions": nominal_episodes * horizon,
            "train_episodes": train_episodes,
            "train_transitions": train_episodes * horizon,
            "validation_episodes": len(val_ids),
            "validation_transitions": len(val_ids) * horizon,
            "interval_train_crl_loss": (
                float(np.mean(interval_losses)) if interval_losses
                else metrics["train_crl_loss"]),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        model_rows.append(row)
        all_rows.append(row)
        interval_losses.clear()
        print(
            f"[offline-crl] size={label:,} step={step_i}/{args.updates} "
            f"train_crl_loss={row['train_crl_loss']:.4f} "
            f"bin={row['binary_accuracy']:.3f} "
            f"cat@{args.categorical_k}={row['categorical_accuracy']:.3f}",
            flush=True)
      if step_i == int(args.updates):
        break
      batch_np = train.sample(int(args.batch_size), train_rng)
      batch = {
          name: jnp.asarray(batch_np[name])
          for name in ("obs", "action")
      }
      key, update_key = jax.random.split(key)
      params, opt_state, update_metrics = update(
          params, opt_state, batch, update_key)
      interval_losses.append(float(np.asarray(update_metrics["crl_loss"])))
      if args.smoke_test:
        print(
            "[smoke] real CRL update synchronized: "
            f"crl_loss={interval_losses[-1]:.6f}",
            flush=True)

    write_csv(os.path.join(model_dir, "metrics.csv"), model_rows)
    plot_crl_learning(
        os.path.join(model_dir, "learning_curves.png"), model_rows,
        f"{label:,} dataset scale", domain_label=domain_label,
        batch_size=int(args.batch_size))
    plot_crl_learning(
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
          "dataset_size_label": label,
          "train_episode_ids": train_ids,
          "method": "ppo_crl",
      }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_meta = {
        "dataset_size_label": label,
        "nominal_subset_episodes": nominal_episodes,
        "nominal_subset_transitions": nominal_episodes * horizon,
        "train_episodes": train_episodes,
        "train_transitions": train_episodes * horizon,
    }
    write_json(os.path.join(model_dir, "metadata.json"), model_meta)
    run_metadata["models"].append(model_meta)
    final_rows.append(model_rows[-1])

  write_csv(os.path.join(args.output_dir, "all_metrics.csv"), all_rows)
  write_csv(
      os.path.join(args.output_dir, "final_size_comparison.csv"), final_rows)
  plot_crl_scaling(
      os.path.join(args.output_dir, "final_size_comparison.png"), final_rows,
      domain_label=domain_label)
  run_metadata["elapsed_seconds"] = time.time() - started
  write_json(os.path.join(args.output_dir, "summary.json"), run_metadata)
  if args.nf_metrics_dir and args.overlay_dir:
    from scripts.plot_offline_nf_vs_crl import write_overlay_plots
    write_overlay_plots(
        nf_dir=args.nf_metrics_dir, crl_dir=args.output_dir,
        output_dir=args.overlay_dir, domain_label=domain_label,
        batch_size=int(args.batch_size))
  if args.smoke_test:
    print("[smoke] PASS: production CRL initialized, updated, and evaluated",
          flush=True)
  print(f"[offline-crl] complete -> {args.output_dir}", flush=True)
