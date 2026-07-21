#!/usr/bin/env python3
"""Standalone PPO+RND for sgcrl's Gym/Acme environments.

This runner intentionally depends only on the repository's environment factory
and the JAX/Haiku/Optax stack.  Environments expose dm_env timesteps through
Acme's GymWrapper; the small vector wrapper below handles automatic resets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sgcrl_jax_acme_compat  # noqa: F401,E402
import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from contrastive.utils import make_environment  # noqa: E402


FIXED_START_END = {
    "point_Impossible": [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([6.0, 8.0], dtype=np.float32),
    ],
    "point_SixteenRooms": [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([20.0, 20.0], dtype=np.float32),
    ],
    "point_SixteenRoomsActual4D": [
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([20.0, 20.0, 0.0, 0.0], dtype=np.float32),
    ],
    "sawyer_push": np.array([0.0, 0.85, 0.02], dtype=np.float32),
    "sawyer_drawer_open": np.array([0.0, 0.54, 0.09], dtype=np.float32),
    "sawyer_button_press": np.array([0.0, 0.8, 0.115], dtype=np.float32),
    "sawyer_peg": np.array([-0.3, 0.6, 0.0], dtype=np.float32),
}


@dataclass
class Args:
    env: str
    seed: int
    num_steps: int
    log_dir: str
    num_envs: int
    rollout_length: int
    num_epochs: int
    num_minibatches: int
    learning_rate: float
    discount: float
    int_discount: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    int_coef: float
    ext_coef: float
    policy_hidden_sizes: Tuple[int, ...]
    value_hidden_sizes: Tuple[int, ...]
    rnd_hidden_sizes: Tuple[int, ...]
    eval_interval: int
    eval_episodes: int
    checkpoint_interval: int
    max_grad_norm: float
    rnd_output_size: int


class RunningMeanStd:
    """Numerically stable host-side running moments."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.shape[0] == 0:
            return
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + np.square(delta) * self.count * batch_count / total
        )
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}


class IntrinsicReturnNormalizer:
    """Normalize rewards by the running std of discounted intrinsic returns."""

    def __init__(self, num_envs: int, discount: float):
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.discount = float(discount)
        self.rms = RunningMeanStd(())

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.discount * self.returns + rewards
        self.rms.update(self.returns)
        normalized = rewards / np.sqrt(self.rms.var + 1e-8)
        self.returns = np.where(dones, 0.0, self.returns)
        return normalized.astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"returns": self.returns, "rms": self.rms.state_dict()}


class VecEnv:
    """Synchronous threaded vectorization for Acme/dm_env environments."""

    def __init__(self, factory: Callable[[int], Any], num_envs: int, seed: int):
        self.envs = [factory(seed + i) for i in range(num_envs)]
        self.num_envs = num_envs
        self.obs_spec = self.envs[0].observation_spec()
        self.action_spec = self.envs[0].action_spec()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_envs, thread_name_prefix="rnd_env"
        )

    def reset(self) -> np.ndarray:
        return np.stack(
            [env.reset().observation for env in self.envs], axis=0
        ).astype(np.float32)

    def step(self, actions: np.ndarray):
        def step_one(i: int):
            action = np.asarray(actions[i], dtype=np.float32)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            if hasattr(self.action_spec, "minimum"):
                action = np.clip(
                    action, self.action_spec.minimum, self.action_spec.maximum
                )
            timestep = self.envs[i].step(action.astype(self.action_spec.dtype))
            reward = 0.0 if timestep.reward is None else float(timestep.reward)
            terminal_obs = np.asarray(timestep.observation, dtype=np.float32)
            done = bool(timestep.last())
            next_obs = (
                np.asarray(self.envs[i].reset().observation, dtype=np.float32)
                if done
                else terminal_obs
            )
            return next_obs, terminal_obs, reward, done

        futures = [
            self.executor.submit(step_one, i) for i in range(self.num_envs)
        ]
        results = [future.result() for future in futures]
        next_obs, terminal_obs, rewards, dones = zip(*results)
        return (
            np.stack(next_obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            np.stack(terminal_obs),
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for env in self.envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()


class Rollout(NamedTuple):
    obs: np.ndarray
    rnd_next_inputs: np.ndarray
    raw_actions: np.ndarray
    logprobs: np.ndarray
    ext_rewards: np.ndarray
    int_rewards: np.ndarray
    dones: np.ndarray
    ext_values: np.ndarray
    int_values: np.ndarray


def parse_sizes(text: str) -> Tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("at least one positive hidden size is required")
    return sizes


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=sorted(FIXED_START_END))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=10_000_000)
    parser.add_argument("--log_dir", default="logs/ppo_rnd")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_length", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--int_discount", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--int_coef", type=float, default=1.0)
    parser.add_argument("--ext_coef", type=float, default=1.0)
    parser.add_argument(
        "--hidden_sizes",
        type=parse_sizes,
        default=None,
        help="Set policy, value, and RND hidden sizes together.",
    )
    parser.add_argument(
        "--policy_hidden_sizes", type=parse_sizes, default=(256, 256)
    )
    parser.add_argument(
        "--value_hidden_sizes", type=parse_sizes, default=(256, 256)
    )
    parser.add_argument("--rnd_hidden_sizes", type=parse_sizes, default=(256, 256))
    parser.add_argument(
        "--eval_interval", type=int, default=50,
        help="Evaluate every N rollout iterations; 0 disables evaluation.",
    )
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument(
        "--checkpoint_interval", type=int, default=500,
        help="Checkpoint every N rollout iterations; 0 disables checkpoints.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--rnd_output_size", type=int, default=256)
    ns = parser.parse_args()
    if ns.hidden_sizes is not None:
        ns.policy_hidden_sizes = ns.hidden_sizes
        ns.value_hidden_sizes = ns.hidden_sizes
        ns.rnd_hidden_sizes = ns.hidden_sizes
    del ns.hidden_sizes
    args = Args(**vars(ns))
    batch_size = args.num_envs * args.rollout_length
    if min(
        args.num_steps,
        args.num_envs,
        args.rollout_length,
        args.num_epochs,
        args.num_minibatches,
        args.eval_episodes,
        args.rnd_output_size,
    ) <= 0:
        parser.error("step, environment, epoch, minibatch, eval, and RND sizes must be positive")
    if batch_size % args.num_minibatches:
        parser.error("num_envs * rollout_length must divide evenly by num_minibatches")
    for name in ("discount", "int_discount", "gae_lambda"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name} must be in [0, 1]")
    return args


def mlp(x: jax.Array, hidden_sizes: Sequence[int], output_size: int) -> jax.Array:
    for width in hidden_sizes:
        x = jax.nn.tanh(hk.Linear(width)(x))
    return hk.Linear(output_size)(x)


def make_networks(
    obs_size: int,
    action_size: int,
    state_size: int,
    args: Args,
):
    def policy_fn(obs):
        output = mlp(obs, args.policy_hidden_sizes, 2 * action_size)
        loc, raw_scale = jnp.split(output, 2, axis=-1)
        scale = jax.nn.softplus(raw_scale) + 1e-3
        return loc, scale

    def ext_value_fn(obs):
        return jnp.squeeze(mlp(obs, args.value_hidden_sizes, 1), axis=-1)

    def int_value_fn(obs):
        return jnp.squeeze(mlp(obs, args.value_hidden_sizes, 1), axis=-1)

    def rnd_predictor_fn(state):
        return mlp(state, args.rnd_hidden_sizes, args.rnd_output_size)

    def rnd_target_fn(state):
        return mlp(state, args.rnd_hidden_sizes, args.rnd_output_size)

    policy = hk.without_apply_rng(hk.transform(policy_fn))
    ext_value = hk.without_apply_rng(hk.transform(ext_value_fn))
    int_value = hk.without_apply_rng(hk.transform(int_value_fn))
    rnd_predictor = hk.without_apply_rng(hk.transform(rnd_predictor_fn))
    rnd_target = hk.without_apply_rng(hk.transform(rnd_target_fn))
    dummy_obs = jnp.zeros((1, obs_size), dtype=jnp.float32)
    dummy_state = jnp.zeros((1, state_size), dtype=jnp.float32)
    return (
        policy,
        ext_value,
        int_value,
        rnd_predictor,
        rnd_target,
        dummy_obs,
        dummy_state,
    )


def normal_log_prob(raw_action, loc, scale):
    log_prob = -0.5 * (
        jnp.square((raw_action - loc) / scale)
        + 2.0 * jnp.log(scale)
        + np.log(2.0 * np.pi)
    )
    correction = 2.0 * (
        np.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)
    )
    return jnp.sum(log_prob - correction, axis=-1)


def normalize_obs(obs, mean, var):
    return jnp.clip((obs - mean) / jnp.sqrt(var + 1e-8), -10.0, 10.0)


def normalize_rnd(state, mean, var):
    return jnp.clip((state - mean) / jnp.sqrt(var + 1e-8), -5.0, 5.0)


def compute_gae(rewards, values, dones, next_value, discount, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(rewards.shape[0])):
        next_values = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + discount * next_values * nonterminal - values[t]
        last_gae = delta + discount * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_pickle(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main(args: Args) -> None:
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    def env_factory(seed: int):
        env, obs_dim = make_environment(
            args.env,
            start_index=0,
            end_index=-1,
            seed=seed,
            fixed_start_end=FIXED_START_END[args.env],
        )
        return env, obs_dim

    probe, state_size = env_factory(args.seed)
    obs_size = int(np.prod(probe.observation_spec().shape))
    action_size = int(np.prod(probe.action_spec().shape))
    max_episode_steps = int(getattr(probe, "_step_limit", 149)) + 1
    del probe

    vec_env = VecEnv(lambda seed: env_factory(seed)[0], args.num_envs, args.seed)
    eval_env = env_factory(args.seed + 100_000)[0]
    (
        policy,
        ext_value,
        int_value,
        rnd_predictor,
        rnd_target,
        dummy_obs,
        dummy_state,
    ) = make_networks(obs_size, action_size, state_size, args)

    k_policy, k_ext, k_int, k_pred, k_target, key = jax.random.split(key, 6)
    params = {
        "policy": policy.init(k_policy, dummy_obs),
        "ext_value": ext_value.init(k_ext, dummy_obs),
        "int_value": int_value.init(k_int, dummy_obs),
        "rnd_predictor": rnd_predictor.init(k_pred, dummy_state),
    }
    rnd_target_params = rnd_target.init(k_target, dummy_state)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.learning_rate, eps=1e-5),
    )
    opt_state = optimizer.init(params)

    obs_rms = RunningMeanStd((obs_size,))
    rnd_obs_rms = RunningMeanStd((state_size,))
    int_return_norm = IntrinsicReturnNormalizer(args.num_envs, args.int_discount)
    obs = vec_env.reset().reshape(args.num_envs, obs_size)
    obs_rms.update(obs)
    rnd_obs_rms.update(obs[:, :state_size])

    run_dir = Path(args.log_dir) / f"ppo_rnd_{args.env}_{args.seed}"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "entrypoint": "baseline-agents/ppo-rnd-gym.py",
                "args": asdict(args),
                "obs_size": obs_size,
                "state_size": state_size,
                "action_size": action_size,
                "fixed_start_end": np.asarray(
                    FIXED_START_END[args.env]
                ).tolist(),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    @jax.jit
    def act_value(policy_params, ext_params, int_params, obs_batch, rng):
        loc, scale = policy.apply(policy_params, obs_batch)
        noise = jax.random.normal(rng, loc.shape)
        raw_action = loc + scale * noise
        action = jnp.tanh(raw_action)
        logprob = normal_log_prob(raw_action, loc, scale)
        return (
            action,
            raw_action,
            logprob,
            ext_value.apply(ext_params, obs_batch),
            int_value.apply(int_params, obs_batch),
        )

    @jax.jit
    def intrinsic_reward(predictor_params, normalized_state):
        prediction = rnd_predictor.apply(predictor_params, normalized_state)
        target = rnd_target.apply(rnd_target_params, normalized_state)
        return 0.5 * jnp.sum(jnp.square(prediction - target), axis=-1)

    @jax.jit
    def bootstrap_values(ext_params, int_params, obs_batch):
        return (
            ext_value.apply(ext_params, obs_batch),
            int_value.apply(int_params, obs_batch),
        )

    def loss_fn(train_params, batch, rng):
        loc, scale = policy.apply(train_params["policy"], batch["obs"])
        new_logprob = normal_log_prob(batch["raw_actions"], loc, scale)
        ratio = jnp.exp(new_logprob - batch["old_logprobs"])
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss = -jnp.mean(
            jnp.minimum(
                ratio * advantages,
                jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                * advantages,
            )
        )
        ext_prediction = ext_value.apply(train_params["ext_value"], batch["obs"])
        int_prediction = int_value.apply(train_params["int_value"], batch["obs"])
        ext_v_loss = 0.5 * jnp.mean(
            jnp.square(ext_prediction - batch["ext_returns"])
        )
        int_v_loss = 0.5 * jnp.mean(
            jnp.square(int_prediction - batch["int_returns"])
        )
        rnd_prediction = rnd_predictor.apply(
            train_params["rnd_predictor"], batch["rnd_next_inputs"]
        )
        target = rnd_target.apply(rnd_target_params, batch["rnd_next_inputs"])
        rnd_loss = 0.5 * jnp.mean(jnp.square(rnd_prediction - target))
        entropy_raw = loc + scale * jax.random.normal(rng, loc.shape)
        entropy = -jnp.mean(normal_log_prob(entropy_raw, loc, scale))
        total = pg_loss + ext_v_loss + int_v_loss + rnd_loss - args.ent_coef * entropy
        logratio = new_logprob - batch["old_logprobs"]
        metrics = {
            "loss": total,
            "policy_loss": pg_loss,
            "ext_value_loss": ext_v_loss,
            "int_value_loss": int_v_loss,
            "rnd_loss": rnd_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((ratio - 1.0) - logratio),
            "clipfrac": jnp.mean(
                (jnp.abs(ratio - 1.0) > args.clip_coef).astype(jnp.float32)
            ),
        }
        return total, metrics

    @jax.jit
    def update_minibatch(train_params, optimizer_state, batch, rng):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_params, batch, rng
        )
        updates, optimizer_state = optimizer.update(
            grads, optimizer_state, train_params
        )
        train_params = optax.apply_updates(train_params, updates)
        return train_params, optimizer_state, metrics

    @jax.jit
    def deterministic_action(policy_params, obs_batch):
        loc, _ = policy.apply(policy_params, obs_batch)
        return jnp.tanh(loc)

    def evaluate(policy_params) -> Dict[str, float]:
        returns = []
        successes = []
        lengths = []
        for _ in range(args.eval_episodes):
            timestep = eval_env.reset()
            ep_return = 0.0
            success = False
            length = 0
            while not timestep.last() and length < max_episode_steps:
                eval_obs = np.asarray(timestep.observation, dtype=np.float32)[None]
                normalized = normalize_obs(
                    jnp.asarray(eval_obs), obs_rms.mean, obs_rms.var
                )
                action = np.asarray(
                    deterministic_action(policy_params, normalized)[0]
                )
                timestep = eval_env.step(action.astype(eval_env.action_spec().dtype))
                reward = 0.0 if timestep.reward is None else float(timestep.reward)
                ep_return += reward
                success = success or reward > 0.0
                length += 1
            returns.append(ep_return)
            successes.append(float(success))
            lengths.append(length)
        return {
            "episode_return": float(np.mean(returns)),
            "success": float(np.mean(successes)),
            "episode_length": float(np.mean(lengths)),
        }

    batch_size = args.num_envs * args.rollout_length
    num_iterations = args.num_steps // batch_size
    if num_iterations < 1:
        raise ValueError("num_steps must cover at least one rollout")
    print(
        f"[ppo_rnd] env={args.env} seed={args.seed} obs={obs_size} "
        f"state={state_size} action={action_size} batch={batch_size} "
        f"iterations={num_iterations} backend={jax.default_backend()}"
    )

    global_step = 0
    start_time = time.time()
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    completed_returns = []

    try:
        for iteration in range(1, num_iterations + 1):
            obs_mean = obs_rms.mean.astype(np.float32)
            obs_var = obs_rms.var.astype(np.float32)
            rnd_mean = rnd_obs_rms.mean.astype(np.float32)
            rnd_var = rnd_obs_rms.var.astype(np.float32)
            storage = {
                "obs": [],
                "rnd_next_inputs": [],
                "raw_actions": [],
                "logprobs": [],
                "ext_rewards": [],
                "int_rewards": [],
                "dones": [],
                "ext_values": [],
                "int_values": [],
            }
            observed = []
            rnd_observed = []

            for _ in range(args.rollout_length):
                key, action_key = jax.random.split(key)
                normalized_obs = normalize_obs(
                    jnp.asarray(obs), obs_mean, obs_var
                )
                action, raw_action, logprob, ext_v, int_v = act_value(
                    params["policy"],
                    params["ext_value"],
                    params["int_value"],
                    normalized_obs,
                    action_key,
                )
                next_obs, ext_reward, done, terminal_obs = vec_env.step(
                    np.asarray(action)
                )
                # RND always sees only the true state, never the appended goal.
                rnd_source = np.where(done[:, None], terminal_obs, next_obs)
                rnd_source = rnd_source[:, :state_size]
                rnd_input = normalize_rnd(
                    jnp.asarray(rnd_source), rnd_mean, rnd_var
                )
                raw_int_reward = np.asarray(
                    intrinsic_reward(params["rnd_predictor"], rnd_input)
                )
                int_reward = int_return_norm.normalize(raw_int_reward, done)

                storage["obs"].append(np.asarray(normalized_obs))
                storage["rnd_next_inputs"].append(np.asarray(rnd_input))
                storage["raw_actions"].append(np.asarray(raw_action))
                storage["logprobs"].append(np.asarray(logprob))
                storage["ext_rewards"].append(ext_reward)
                storage["int_rewards"].append(int_reward)
                storage["dones"].append(done)
                storage["ext_values"].append(np.asarray(ext_v))
                storage["int_values"].append(np.asarray(int_v))

                episode_returns += ext_reward
                if done.any():
                    completed_returns.extend(episode_returns[done].tolist())
                    episode_returns[done] = 0.0
                observed.append(next_obs)
                rnd_observed.append(rnd_source)
                obs = next_obs.reshape(args.num_envs, obs_size)
                global_step += args.num_envs

            rollout = Rollout(**{
                name: np.asarray(values) for name, values in storage.items()
            })
            normalized_next_obs = normalize_obs(
                jnp.asarray(obs), obs_mean, obs_var
            )
            next_ext_v, next_int_v = bootstrap_values(
                params["ext_value"], params["int_value"], normalized_next_obs
            )
            ext_adv, ext_returns = compute_gae(
                rollout.ext_rewards,
                rollout.ext_values,
                rollout.dones,
                np.asarray(next_ext_v),
                args.discount,
                args.gae_lambda,
            )
            int_adv, int_returns = compute_gae(
                rollout.int_rewards,
                rollout.int_values,
                rollout.dones,
                np.asarray(next_int_v),
                args.int_discount,
                args.gae_lambda,
            )
            combined_adv = args.ext_coef * ext_adv + args.int_coef * int_adv
            flat_batch = {
                "obs": rollout.obs.reshape(batch_size, obs_size),
                "rnd_next_inputs": rollout.rnd_next_inputs.reshape(
                    batch_size, state_size
                ),
                "raw_actions": rollout.raw_actions.reshape(batch_size, action_size),
                "old_logprobs": rollout.logprobs.reshape(batch_size),
                "advantages": combined_adv.reshape(batch_size),
                "ext_returns": ext_returns.reshape(batch_size),
                "int_returns": int_returns.reshape(batch_size),
            }

            metric_rows = []
            for _ in range(args.num_epochs):
                key, permutation_key = jax.random.split(key)
                permutation = np.asarray(
                    jax.random.permutation(permutation_key, batch_size)
                )
                # Shuffle a genuinely flat T*E batch, not only the time axis.
                for indices in np.split(permutation, args.num_minibatches):
                    key, loss_key = jax.random.split(key)
                    minibatch = {
                        name: jnp.asarray(value[indices])
                        for name, value in flat_batch.items()
                    }
                    params, opt_state, metrics = update_minibatch(
                        params, opt_state, minibatch, loss_key
                    )
                    metric_rows.append(
                        {name: float(value) for name, value in metrics.items()}
                    )

            # Moment updates happen after PPO, preserving on-policy logprobs and
            # exact reward/loss RND normalization throughout this iteration.
            obs_rms.update(np.concatenate(observed, axis=0))
            rnd_obs_rms.update(np.concatenate(rnd_observed, axis=0))
            learner_row = {
                "iteration": iteration,
                "learner_steps": global_step,
                "steps_per_second": global_step / max(time.time() - start_time, 1e-6),
                "ext_reward_mean": float(rollout.ext_rewards.mean()),
                "int_reward_mean": float(rollout.int_rewards.mean()),
                "int_return_std": float(np.sqrt(int_return_norm.rms.var + 1e-8)),
                "episode_return_mean_100": (
                    float(np.mean(completed_returns[-100:]))
                    if completed_returns
                    else float("nan")
                ),
                **{
                    name: float(np.mean([row[name] for row in metric_rows]))
                    for name in metric_rows[0]
                },
            }
            append_csv(run_dir / "logs" / "learner" / "logs.csv", learner_row)
            print(
                f"[ppo_rnd] iter={iteration}/{num_iterations} steps={global_step} "
                f"ext={learner_row['ext_reward_mean']:.4g} "
                f"int={learner_row['int_reward_mean']:.4g} "
                f"loss={learner_row['loss']:.4g} "
                f"sps={learner_row['steps_per_second']:.0f}"
            )

            if args.eval_interval and (
                iteration == 1 or iteration % args.eval_interval == 0
            ):
                eval_row = {
                    "iteration": iteration,
                    "learner_steps": global_step,
                    **evaluate(params["policy"]),
                }
                append_csv(run_dir / "logs" / "eval" / "logs.csv", eval_row)
                print(
                    f"[eval] iter={iteration} return={eval_row['episode_return']:.4g} "
                    f"success={eval_row['success']:.3f}"
                )

            if args.checkpoint_interval and (
                iteration % args.checkpoint_interval == 0
                or iteration == num_iterations
            ):
                payload = {
                    "params": params,
                    "rnd_target_params": rnd_target_params,
                    "optimizer_state": opt_state,
                    "key": key,
                    "iteration": iteration,
                    "global_step": global_step,
                    "obs_rms": obs_rms.state_dict(),
                    "rnd_obs_rms": rnd_obs_rms.state_dict(),
                    "int_return_normalizer": int_return_norm.state_dict(),
                    "args": asdict(args),
                }
                checkpoint = checkpoint_dir / f"ckpt_iter_{iteration:07d}.pkl"
                atomic_pickle(checkpoint, payload)
                shutil.copyfile(checkpoint, checkpoint_dir / "latest.pkl")
    finally:
        vec_env.close()
        close = getattr(eval_env, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main(parse_args())
