"""CRL (Contrastive RL) on Isaac Gym AllegroKukaThrow.

Single task-goal data collection: the actor is always conditioned on the
fixed task goal (packed in obs[obs_dim:]) during rollout collection only.
Both the critic and actor gradient updates use HER future goals sampled
from EpisodeReplay.

This is a direct port of the baseline CRL code to Isaac Gym — the only
conceptual difference from that baseline is data collection using the
task goal instead of a hindsight/future goal.

Defaults: tableside palm-goal task (table_push=True, palm_goal=True, 6-D goal).

Usage:
    python crl_isaacgym.py \
        --num_envs=1024 \
        --num_timesteps=300000000 \
        --wandb_dir=logs/crl_allegrohand/ \
        --track=true
"""

import sys as _sys


# ---------------------------------------------------------------------------
# Pre-bootstrap: create PhysX BEFORE any JAX/torch import
# ---------------------------------------------------------------------------
def _pre_bootstrap() -> None:
    """Parse flags directly from sys.argv and construct the Allegro env."""

    def _get(name: str, default: str = '') -> str:
        for i, a in enumerate(_sys.argv):
            if a.startswith(f'--{name}='):
                return a[len(f'--{name}='):]
            if a == f'--{name}' and i + 1 < len(_sys.argv):
                return _sys.argv[i + 1]
        return default

    def _getbool(name: str, default: bool = True) -> bool:
        v = _get(name, 'true' if default else 'false').strip().lower()
        return v not in ('false', 'f', '0', 'no', 'n', 'off')

    def _getxyz(name: str, default: str) -> tuple:
        raw = _get(name, default)
        xyz = tuple(float(x) for x in raw.split(','))
        if len(xyz) != 3:
            raise ValueError(
                f'--{name} must be 3 comma-separated floats, got {raw!r}')
        return xyz

    num_envs       = int(_get('num_envs', '1024'))
    seed           = int(_get('seed', '1'))
    episode_length = int(_get('episode_length', '300'))
    pipeline       = _get('pipeline', 'gpu') or 'gpu'
    randomize_init = _getbool('randomize_init', True)
    randomize_obj  = _getbool('randomize_object_shape', True)
    palm_goal      = _getbool('palm_goal', True)
    table_push     = _getbool('table_push', True)
    fixed_tgt      = _getxyz('fixed_target_xyz', '0.5,-0.3,0.4')
    push_raw       = _get('table_push_xyz', '')
    table_push_xyz = tuple(float(x) for x in push_raw.split(',')) if push_raw else None
    palm_raw       = _get('palm_goal_xyz', '')
    palm_goal_xyz  = tuple(float(x) for x in palm_raw.split(',')) if palm_raw else None

    print(
        f'[crl_isaacgym] bootstrap PhysX (before JAX): '
        f'E={num_envs} ep={episode_length} pipeline={pipeline} '
        f'randomize_init={randomize_init} randomize_object_shape={randomize_obj} '
        f'palm_goal={palm_goal} table_push={table_push} '
        f'table_push_xyz={table_push_xyz} palm_goal_xyz={palm_goal_xyz}',
        flush=True,
    )

    throw_success = _get('isaacgym_throw_success', 'in_bucket') or 'in_bucket'
    from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv
    import envs.isaacgym_physx_bootstrap as _boot_mod
    _boot_mod._PREBUILT = AllegroKukaThrowVecEnv(
        num_envs=num_envs,
        seed=seed * 31,
        episode_length=episode_length,
        fixed_target_xyz=fixed_tgt,
        pipeline=pipeline,
        headless=True,
        randomize_init=randomize_init,
        randomize_object_shape=randomize_obj,
        palm_goal=palm_goal,
        palm_goal_xyz=palm_goal_xyz,
        table_push=table_push,
        table_push_xyz=table_push_xyz,
        throw_success=throw_success,
    )
    print(
        f'[crl_isaacgym] PhysX ready '
        f'(device={_boot_mod._PREBUILT.device}, E={_boot_mod._PREBUILT.num_envs})',
        flush=True,
    )


_pre_bootstrap()


# ---------------------------------------------------------------------------
# Remaining imports (after PhysX bootstrap)
# ---------------------------------------------------------------------------
import os
import time
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import wandb

try:
    import wandb_osh
    from wandb_osh.hooks import TriggerWandbSyncHook
except ImportError:  # optional; offline runs still write local wandb dirs
    wandb_osh = None
    TriggerWandbSyncHook = None

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState

from contrastive import ppo_learner_isaacgym as _ppo_ig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Args:
    # Experiment
    seed: int = 1
    exp_name: str = ''

    # W&B logging
    track: bool = False
    wandb_project_name: str = 'sgcrl'
    wandb_entity: Optional[str] = None
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'default'
    wandb_name_tag: str = ''

    # Checkpointing
    save_checkpoint: bool = True
    checkpoint_interval: int = 100
    checkpoint_keep_last: int = 0

    # Isaac Gym environment
    num_envs: int = 1024
    episode_length: int = 300
    pipeline: str = 'gpu'
    randomize_init: bool = True
    randomize_object_shape: bool = True
    # Tableside palm-goal task: 6-D goal = [palm xyz | object xyz].
    palm_goal: bool = True
    table_push: bool = True
    table_push_xyz: str = '0.20,-0.15,0.555'   # object target on desk
    palm_goal_xyz: str = '0.20,-0.15,0.620'     # palm hover above target
    fixed_target_xyz: str = '0.5,-0.3,0.4'      # bucket (parked off-table)

    # CRL algorithm
    num_timesteps: int = 300_000_000
    batch_size: int = 256
    updates_per_env_step: int = 3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    discount: float = 0.99
    entropy_cost: float = 0.1
    logsumexp_cost: float = 0.1
    rep_size: int = 64
    max_replay_size: int = 1_000_000
    min_replay_size: int = 10_000
    log_interval_sim_steps: int = 300


def _parse_args(argv=None) -> Args:
    """Parse ``--key=value`` / ``--key value`` into ``Args`` (no tyro)."""
    argv = list(_sys.argv[1:] if argv is None else argv)
    defaults = Args()
    values = {f.name: getattr(defaults, f.name) for f in Args.__dataclass_fields__.values()}

    def _consume(name: str, raw: str) -> None:
        typ = type(getattr(defaults, name))
        if typ is bool:
            values[name] = raw.strip().lower() not in (
                'false', 'f', '0', 'no', 'n', 'off')
        elif typ is int:
            values[name] = int(raw)
        elif typ is float:
            values[name] = float(raw)
        else:
            values[name] = raw

    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith('--'):
            raise SystemExit(f'unexpected argument: {a}')
        a = a[2:]
        if '=' in a:
            name, raw = a.split('=', 1)
            if name not in values:
                raise SystemExit(f'unknown flag: --{name}')
            _consume(name, raw)
            i += 1
            continue
        name = a
        if name not in values:
            raise SystemExit(f'unknown flag: --{name}')
        if isinstance(values[name], bool) and (
                i + 1 >= len(argv) or argv[i + 1].startswith('--')):
            values[name] = True
            i += 1
            continue
        if i + 1 >= len(argv):
            raise SystemExit(f'--{name} requires a value')
        _consume(name, argv[i + 1])
        i += 2
    return Args(**values)


# ---------------------------------------------------------------------------
# Networks (verbatim from CRL baseline)
# ---------------------------------------------------------------------------
class SA_encoder(nn.Module):
    rep_size: int
    norm_type: str = 'layer_norm'

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, 'fan_in', 'uniform')
        b_init = nn.initializers.zeros
        norm = (
            (lambda x: nn.LayerNorm()(x))
            if self.norm_type == 'layer_norm'
            else (lambda x: x)
        )
        x = jnp.concatenate([s, a], axis=-1)
        for _ in range(4):
            x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=b_init)(x)
            x = norm(x)
            x = nn.swish(x)
        return nn.Dense(self.rep_size, kernel_init=lecun_uniform, bias_init=b_init)(x)


class G_encoder(nn.Module):
    rep_size: int
    norm_type: str = 'layer_norm'

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, 'fan_in', 'uniform')
        b_init = nn.initializers.zeros
        norm = (
            (lambda x: nn.LayerNorm()(x))
            if self.norm_type == 'layer_norm'
            else (lambda x: x)
        )
        x = g
        for _ in range(4):
            x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=b_init)(x)
            x = norm(x)
            x = nn.swish(x)
        return nn.Dense(self.rep_size, kernel_init=lecun_uniform, bias_init=b_init)(x)


class Actor(nn.Module):
    action_size: int
    norm_type: str = 'layer_norm'
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(
        self, s: jnp.ndarray, g_repr: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        lecun_uniform = variance_scaling(1 / 3, 'fan_in', 'uniform')
        b_init = nn.initializers.zeros
        norm = (
            (lambda x: nn.LayerNorm()(x))
            if self.norm_type == 'layer_norm'
            else (lambda x: x)
        )
        x = jnp.concatenate([s, g_repr], axis=-1)
        for _ in range(4):
            x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=b_init)(x)
            x = norm(x)
            x = nn.swish(x)
        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=b_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=b_init)(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main(args: Args) -> None:
    tag = f"{args.wandb_name_tag + '__' if args.wandb_name_tag else ''}"
    args.exp_name = f'{tag}crl_allegrohand__{args.seed}__{int(time.time())}'

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            group=args.wandb_group,
            name=args.exp_name,
            config=vars(args),
            save_code=True,
        )
        trigger_sync = None
        if args.wandb_mode == 'offline':
            if wandb_osh is not None and TriggerWandbSyncHook is not None:
                wandb_osh.set_log_level('ERROR')
                trigger_sync = TriggerWandbSyncHook()
            else:
                print(
                    '[crl_isaacgym] wandb_osh not installed; '
                    'offline wandb logs will not auto-sync',
                    flush=True,
                )

    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f'checkpoints/{args.exp_name}'
        os.makedirs(save_path, exist_ok=True)

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # ---- Isaac Gym env (reuses prebuilt PhysX from _pre_bootstrap) ----
    def _parse_xyz(s: str) -> Optional[tuple]:
        s = s.strip()
        if not s:
            return None
        parts = tuple(float(x) for x in s.split(','))
        return parts if len(parts) == 3 else None

    vec_env = _ppo_ig.IsaacGymVecEnv(
        env_name='allegro_kuka_throw',
        num_envs=args.num_envs,
        seed=args.seed,
        isaacgym_kwargs={
            'episode_length':        args.episode_length,
            'fixed_target_xyz':      _parse_xyz(args.fixed_target_xyz),
            'pipeline':              args.pipeline,
            'randomize_init':        args.randomize_init,
            'randomize_object_shape': args.randomize_object_shape,
            'palm_goal':             args.palm_goal,
            'palm_goal_xyz':         _parse_xyz(args.palm_goal_xyz),
            'table_push':            args.table_push,
            'table_push_xyz':        _parse_xyz(args.table_push_xyz),
        },
    )
    E             = vec_env.num_envs
    obs_dim_total = vec_env.observation_shape[0]  # 58 (tableside palm-goal)
    goal_dim      = vec_env.goal_dim              #  6 (palm xyz + object xyz)
    obs_dim       = obs_dim_total - goal_dim       # 52 (q, qd, palm, obj)
    act_dim       = vec_env.action_shape[0]        # 23
    t_max         = vec_env.episode_length         # 300

    # HER slice: state[her_start:her_end] = achieved goal at each timestep.
    # For palm-goal (obs_dim=52, goal_dim=6): state[46:52] = [palm xyz | obj xyz].
    her_start = obs_dim - goal_dim   # 46
    her_end   = obs_dim              # 52

    print(
        f'[crl_isaacgym] obs_dim={obs_dim} goal_dim={goal_dim} '
        f'act_dim={act_dim} her=[{her_start}:{her_end}]',
        flush=True,
    )

    # ---- Networks ----
    sa_encoder = SA_encoder(rep_size=args.rep_size)
    g_encoder  = G_encoder(rep_size=args.rep_size)
    actor_net  = Actor(action_size=act_dim)

    key, k_actor, k_sa, k_g = jax.random.split(key, 4)

    # Init on CPU: GPU cuSolver QR is broken after PhysX create_sim.
    with _ppo_ig._haiku_init_device(True):
        actor_state = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(
                k_actor,
                np.zeros([1, obs_dim],     dtype=np.float32),
                np.zeros([1, args.rep_size], dtype=np.float32),
            ),
            tx=optax.adam(args.actor_learning_rate, eps=1e-7),
        )
        sa_params = sa_encoder.init(
            k_sa,
            np.zeros([1, obs_dim], dtype=np.float32),
            np.zeros([1, act_dim], dtype=np.float32),
        )
        g_params = g_encoder.init(
            k_g,
            np.zeros([1, goal_dim], dtype=np.float32),
        )
        critic_state = TrainState.create(
            apply_fn=None,
            params={'sa_encoder': sa_params, 'g_encoder': g_params},
            tx=optax.adam(args.critic_learning_rate, eps=1e-7),
        )

    actor_state  = _ppo_ig._isaacgym_init_on_cpu_then_gpu(True, actor_state)
    critic_state = _ppo_ig._isaacgym_init_on_cpu_then_gpu(True, critic_state)

    # ---- JIT-compiled update functions ----

    @jax.jit
    def update_critic(
        critic_state: TrainState,
        state: jnp.ndarray,
        action: jnp.ndarray,
        future_goal: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """InfoNCE + logsumexp regularisation. Matches baseline exactly."""
        del key

        def loss_fn(params):
            sa_repr = sa_encoder.apply(params['sa_encoder'], state, action)
            g_repr  = g_encoder.apply(params['g_encoder'],  future_goal)

            # (B, B) Euclidean-distance logit matrix
            logits = -jnp.sqrt(
                jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))

            # InfoNCE (matches baseline)
            infonce = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            # Logsumexp regularizer (+1e-6 matches baseline)
            lse = jax.nn.logsumexp(logits + 1e-6, axis=1)
            lse_reg = args.logsumexp_cost * jnp.mean(lse ** 2)

            B = logits.shape[0]
            I = jnp.eye(B)
            correct = jnp.argmax(logits, axis=1) == jnp.arange(B)
            lp = jnp.sum(logits * I) / B
            ln = jnp.sum(logits * (1 - I)) / (B * (B - 1))
            return infonce + lse_reg, (correct, lp, ln, lse)

        (loss, (correct, lp, ln, lse)), grad = jax.value_and_grad(
            loss_fn, has_aux=True)(critic_state.params)
        new_state = critic_state.apply_gradients(grads=grad)
        metrics = {
            'critic_loss':          loss,
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos':           lp,
            'logits_neg':           ln,
            'logsumexp':            lse.mean(),
        }
        return new_state, metrics

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        critic_params: dict,
        state: jnp.ndarray,
        her_goal: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """Actor: maximize φ(s,a)·ψ(g) minus entropy cost.

        Trained with HER future goals (same as critic) for dense gradient
        signal.  Task goal is only used during rollout collection (batched_act).
        Critic weights are stop-gradiented.
        """
        def loss_fn(actor_params):
            g_repr = g_encoder.apply(
                jax.lax.stop_gradient(critic_params['g_encoder']), her_goal)
            means, log_stds = actor_net.apply(actor_params, state, g_repr)
            stds = jnp.exp(log_stds)
            x_t  = means + stds * jax.random.normal(key, shape=means.shape)
            action = nn.tanh(x_t)
            log_prob = jax.scipy.stats.norm.logpdf(x_t, loc=means, scale=stds)
            log_prob -= jnp.log(1.0 - jnp.square(action) + 1e-6)
            log_prob = log_prob.sum(-1)   # (B,)
            sa_repr = sa_encoder.apply(
                jax.lax.stop_gradient(critic_params['sa_encoder']), state, action)
            qf = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
            return jnp.mean(args.entropy_cost * log_prob - qf), log_prob

        (loss, log_prob), grad = jax.value_and_grad(
            loss_fn, has_aux=True)(actor_state.params)
        new_state = actor_state.apply_gradients(grads=grad)
        metrics = {
            'actor_loss':     loss,
            'sample_entropy': jnp.mean(-log_prob),
        }
        return new_state, metrics

    @jax.jit
    def batched_act(
        actor_params: dict,
        critic_params: dict,
        obs_jax: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Stochastic rollout policy conditioned on task goal (obs[obs_dim:])."""
        state     = obs_jax[:, :obs_dim]
        task_goal = obs_jax[:, obs_dim:]
        g_repr = g_encoder.apply(critic_params['g_encoder'], task_goal)
        means, log_stds = actor_net.apply(actor_params, state, g_repr)
        stds = jnp.exp(log_stds)
        return nn.tanh(means + stds * jax.random.normal(key, shape=means.shape))

    # ---- Replay buffer (future-goal HER) ----
    replay = _ppo_ig.EpisodeReplay(
        capacity=args.max_replay_size,
        obs_dim=obs_dim,
        discount=args.discount,
        start_index=her_start,
        end_index=her_end,
    )
    np_rng = np.random.default_rng(args.seed + 17)

    # ---- Initial env reset ----
    obs = vec_env.reset()   # (E, obs_dim_total)

    # Task goal: constant for all envs (fixed desk target + palm hover).
    # Cached once; only used in batched_act for rollout collection.
    task_goal_ref = jnp.asarray(obs[0, obs_dim:])   # (goal_dim,)

    ep_obs     = np.zeros((E, t_max + 1, obs_dim_total), dtype=np.float32)
    ep_act     = np.zeros((E, t_max,     act_dim),        dtype=np.float32)
    ep_len     = np.zeros(E, dtype=np.int32)
    ep_success = np.zeros(E, dtype=np.float32)
    ep_obs[:, 0] = obs
    rows = np.arange(E)

    recent_success: list = []
    recent_lengths: list = []

    n_updates     = 0
    n_log_updates = 0
    global_step   = 0
    log_metrics: dict = {}

    num_sim_steps = args.num_timesteps // E
    log_every     = max(1, args.log_interval_sim_steps)
    ckpt_interval = args.checkpoint_interval
    start_time    = time.time()

    print(
        f'[crl_isaacgym] training: sim_steps={num_sim_steps} '
        f'log_every={log_every} E={E} batch={args.batch_size} '
        f'updates/step={args.updates_per_env_step} '
        f'min_replay={args.min_replay_size} max_replay={args.max_replay_size}',
        flush=True,
    )

    for sim in range(num_sim_steps):

        # ---- Act: uniform random before min_replay_size, trained policy after ----
        if replay.size < args.min_replay_size:
            action = np_rng.uniform(-1.0, 1.0, size=(E, act_dim)).astype(np.float32)
        else:
            key, k_act = jax.random.split(key)
            action = np.asarray(
                batched_act(
                    actor_state.params, critic_state.params,
                    jnp.asarray(obs), k_act,
                ),
                dtype=np.float32,
            )

        # ---- Step env ----
        next_obs, env_rew, dones, terminal_obs, _ = vec_env.step(action)
        success = np.asarray(vec_env.last_success, dtype=np.float32)

        overflow = ep_len >= t_max
        if np.any(overflow):
            dones = np.logical_or(dones, overflow)

        write_t = np.minimum(ep_len, t_max - 1)
        ep_act[rows, write_t] = action
        ep_len     = np.minimum(ep_len + 1, t_max)
        ep_obs[rows, ep_len] = next_obs
        ep_success = np.maximum(ep_success, success)

        # ---- Episode boundary: flush to replay ----
        for i in np.flatnonzero(dones).tolist():
            t_i = min(int(ep_len[i]), t_max)
            obs_traj = ep_obs[i, :t_i + 1].copy()
            obs_traj[t_i] = terminal_obs[i]
            act_traj = ep_act[i, :t_i].copy()
            succeeded = bool(ep_success[i] >= 0.5)
            try:
                replay.add_episode(obs_traj, act_traj, successful=succeeded)
            except AssertionError:
                pass
            recent_success.append(float(succeeded))
            recent_lengths.append(int(t_i))
            if len(recent_success) > 1000:
                del recent_success[:-1000]
                del recent_lengths[:-1000]
            ep_obs[i, 0] = next_obs[i]
            ep_len[i]    = 0
            ep_success[i] = 0.0

        obs = next_obs
        global_step += E

        # ---- CRL gradient steps ----
        if replay.size >= args.min_replay_size:
            for _ in range(args.updates_per_env_step):
                batch = replay.sample(args.batch_size, np_rng)

                state_b       = jnp.asarray(batch['obs'][:, :obs_dim])
                future_goal_b = jnp.asarray(batch['obs'][:, obs_dim:])
                action_b      = jnp.asarray(batch['action'])

                key, k_c, k_a = jax.random.split(key, 3)

                # Critic: HER future goal as positive label
                critic_state, crit_met = update_critic(
                    critic_state, state_b, action_b, future_goal_b, k_c)

                # Actor: HER future goal; task goal only for rollout (batched_act)
                actor_state, act_met = update_actor(
                    actor_state, critic_state.params, state_b, future_goal_b, k_a)

                n_updates     += 1
                n_log_updates += 1
                for k, v in {**crit_met, **act_met}.items():
                    log_metrics[k] = log_metrics.get(k, 0.0) + float(v)

        # ---- Logging ----
        sim_done = sim + 1
        if sim_done % log_every != 0 and sim_done != num_sim_steps:
            continue

        iteration = sim_done // log_every
        elapsed   = time.time() - start_time
        sps       = global_step / max(1e-6, elapsed)
        smoothed  = {k: v / max(1, n_log_updates) for k, v in log_metrics.items()}

        log = {
            'iteration':          float(iteration),
            'global_step':        float(global_step),
            'sps':                sps,
            'replay_size':        float(replay.size),
            'replay_episodes':    float(replay.num_episodes),
            'ep_length_mean':     (
                float(np.mean(recent_lengths[-100:])) if recent_lengths
                else float('nan')),
            'train_success_mean': (
                float(np.mean(recent_success[-100:])) if recent_success
                else float('nan')),
            'train_success_1000': (
                float(np.mean(recent_success[-1000:])) if recent_success
                else float('nan')),
            'sgd_steps':          float(n_updates),
            **{f'training/{k}': v for k, v in smoothed.items()},
        }
        pprint.pprint(log)

        if args.track:
            wandb.log(log, step=iteration)
            if args.wandb_mode == 'offline' and trigger_sync is not None:
                trigger_sync()

        log_metrics.clear()
        n_log_updates = 0

        # ---- Checkpoint ----
        if (args.save_checkpoint
                and ckpt_interval > 0
                and (iteration % ckpt_interval == 0 or sim_done == num_sim_steps)):
            _ckpt_kw = dict(
                policy_params=actor_state.params,
                value_params=None,
                q_params=critic_state.params,
                ppo_opt_state=actor_state.opt_state,
                q_opt_state=critic_state.opt_state,
                iteration=int(iteration),
                global_step=int(global_step),
                key=key,
            )
            milestone = str(save_path / f'ckpt_iter_{iteration:07d}.pkl')
            _ppo_ig._save_checkpoint(milestone, **_ckpt_kw)
            _ppo_ig._save_checkpoint(str(save_path / 'latest.pkl'), **_ckpt_kw)
            _ppo_ig._prune_old_checkpoints(str(save_path), args.checkpoint_keep_last)

        print(
            f'[crl_isaacgym] iter={iteration} step={global_step} '
            f'sps={sps:.0f} replay={int(replay.size)} '
            f'ep_len={log["ep_length_mean"]:.1f} '
            f'succ1000={log["train_success_1000"]:.4f}',
            flush=True,
        )

    if args.track:
        wandb.finish()


if __name__ == '__main__':
    args = _parse_args()
    main(args)
