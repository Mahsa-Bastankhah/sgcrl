import os
import sys
from pathlib import Path

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

# Allegro / Isaac Gym: create GPU PhysX BEFORE JAX takes the CUDA context.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from envs.isaacgym_physx_bootstrap import (  # noqa: E402
    maybe_create_from_argv as _ig_boot,
    take_prebuilt_env as _ig_take,
)
_ig_boot()

# TFP 0.25 still reads jax.interpreters.xla.pytype_aval_mappings; JAX 0.10
# removed it. Must run after PhysX bootstrap and before import distrax.
import sgcrl_jax_acme_compat  # noqa: F401,E402

import json
import pickle
import subprocess
import time
import tyro
import numpy as np
import functools
import pprint
import re
wandb = None
wandb_osh = None
TriggerWandbSyncHook = None

import jax
import flax
import optax
import distrax
import flax.linen as nn
import jax.numpy as jnp

from flax.training.train_state import TrainState
from dataclasses import dataclass, field
from typing import Any, Sequence, NamedTuple
import utils.running_statistics as running_statistics


def count_parameters(params):
    return sum(
        int(np.prod(np.asarray(p.shape)))
        for p in jax.tree_util.tree_leaves(params))


def _rnd_isaacgym_flags(args) -> dict:
  return {
      name: getattr(args, name)
      for name in dir(args)
      if name.startswith('isaacgym_')
  }


def _maybe_render_rnd_video(args, ckpt_path: str, es: int, env_steps: int) -> None:
  """Render ~num_train_videos stochastic rollouts spread across evals."""
  nvid = int(getattr(args, 'num_train_videos', 0) or 0)
  if nvid <= 0 or not ckpt_path:
    return
  n_eval = max(1, int(args.num_eval_steps))
  marks = {
      max(1, int(round((k + 1) * n_eval / float(nvid))))
      for k in range(nvid)
  }
  if int(es) not in marks:
    return
  vid_dir = Path(args.wandb_dir) / 'videos'
  vid_dir.mkdir(parents=True, exist_ok=True)
  flags_path = vid_dir / 'rnd_video_flags.json'
  if not flags_path.is_file():
    flags_path.write_text(
        json.dumps(_rnd_isaacgym_flags(args), indent=2, sort_keys=True),
        encoding='utf-8')
  out = vid_dir / f'iter_{int(es):07d}_stoch.mp4'
  script = str(_REPO_ROOT / 'scripts' / 'allegro_kuka_throw_ckpt_video.py')
  ep = int(getattr(args, 'isaacgym_episode_length', 0) or args.rollout_length)
  cmd = [
      sys.executable, '-u', script,
      f'--checkpoint={ckpt_path}',
      f'--flags-json={flags_path}',
      f'--output={out}',
      f'--num-steps={ep}',
      '--episodes=1',
      '--fps=30',
      f'--seed={int(args.seed) + int(es)}',
      f'--pipeline={args.isaacgym_pipeline}',
      '--mark-horizon=0',
  ]
  print(f'[ppo_rnd] train video es={es} steps={int(env_steps)} -> {out}',
        flush=True)
  try:
    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=300)
    if proc.returncode == 0 and out.is_file():
      print(f'[ppo_rnd] wrote {out}', flush=True)
    else:
      tail = (proc.stderr or proc.stdout or '').strip().splitlines()
      print(f'[ppo_rnd] video FAILED rc={proc.returncode}\n'
            + '\n'.join(tail[-8:]), flush=True)
  except Exception as exc:
    print(f'[ppo_rnd] video FAILED: {exc}', flush=True)


def save_params(path: str, params: Any):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(params))


class MLP(nn.Module):
    """Local copy of builderbench.utils.networks.MLP (no etils)."""

    layer_sizes: Sequence[int]
    activation: Any = nn.relu
    kernel_init: Any = nn.initializers.lecun_uniform()
    bias_init: Any = nn.initializers.zeros
    final_kernet_init: Any = nn.initializers.lecun_uniform()
    final_bias_init: Any = nn.initializers.zeros
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    def get_penultimate(self, data: jnp.ndarray):
        return self(data, return_penultimate=True)

    @nn.compact
    def __call__(self, data: jnp.ndarray, return_penultimate: bool = False):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes[:-1]):
            hidden = nn.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.bias,
            )(hidden)
            hidden = self.activation(hidden)
            if self.layer_norm:
                hidden = nn.LayerNorm()(hidden)
        if return_penultimate:
            return hidden
        final_layer_index = len(self.layer_sizes) - 1
        hidden = nn.Dense(
            self.layer_sizes[-1],
            name=f'hidden_{final_layer_index}',
            kernel_init=self.final_kernet_init,
            bias_init=self.final_bias_init,
            use_bias=self.bias,
        )(hidden)
        if self.activate_final:
            hidden = self.activation(hidden)
            if self.layer_norm:
                hidden = nn.LayerNorm()(hidden)
        return hidden

class HardSuccessRewardWrapper:
    """Expose only BuilderBench's binary hard-success reward."""

    def __init__(self, env):
        self.env = env

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(reward=state.metrics["success"])

    def step(self, state, action):
        state = self.env.step(state, action)
        return state.replace(reward=state.metrics["success"])

    def __getattr__(self, name):
        return getattr(self.env, name)


class FixedGoalWrapper:
    """Pin target_goal (and mocaps) on every reset so train and eval match."""

    def __init__(self, env, fixed_goal):
        from envs.builderbench_utils import set_task_mocap_pos
        self.env = env
        self._fixed_goal = jnp.asarray(fixed_goal, dtype=jnp.float32).reshape(-1)
        self._set_task_mocap_pos = set_task_mocap_pos
        inner = env
        while hasattr(inner, 'env') and not hasattr(inner, '_task_mocap_targets'):
            inner = inner.env
        self._mocap_targets = inner._task_mocap_targets
        self._n_task_cubes = int(self._fixed_goal.size // 3)

    def _apply(self, state):
        fixed_pos = self._fixed_goal.reshape(self._n_task_cubes, 3)
        info = dict(state.info)
        info['target_goal'] = jnp.broadcast_to(
            self._fixed_goal, state.info['target_goal'].shape)
        if 'target_mocap_pos' in info:
            info['target_mocap_pos'] = jnp.broadcast_to(
                fixed_pos, state.info['target_mocap_pos'].shape)
        mocap_pos = self._set_task_mocap_pos(
            state.data.mocap_pos, self._mocap_targets, fixed_pos)
        return state.replace(data=state.data.replace(mocap_pos=mocap_pos), info=info)

    def reset(self, rng):
        return self._apply(self.env.reset(rng))

    def step(self, state, action):
        return self._apply(self.env.step(state, action))

    def __getattr__(self, name):
        return getattr(self.env, name)

@dataclass
class Args:
    # experiment
    agent: str = "ppo-rnd"
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    
    # logging and checkpointing
    track: bool = False
    wandb_project_name: str = "builderbench"
    wandb_entity: str = 'raj19'
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'default'
    wandb_name_tag: str = ''

    num_eval_steps: int = 50             # number of evaluation / logging / saving steps
    num_reset_steps: int = 50            # number of times to call true resets (env.reset) instead of soft resets (AutoResetWrapper)

    save_checkpoint: bool = True

    # environment
    env_id: str = 'creative-1-task1'
    num_envs: int = 2048
    num_eval_envs: int = 128
    env_early_termination: bool = True
    env_episode_length: int = None
    permutation_invariant_reward: bool = True   # invariance to the order of cubes in any structure
    # Init randomization (matches NF / CRL jobs: fixed x, no lane permutation, fixed goal).
    permute_start_boxes: bool = False  # False = keep task-file lane order (no shuffle)
    fixed_start_x: float = 0.1        # cube init x fixed; <0 keeps [0.05,0.1] range
    fix_goal: bool = True             # fix target_goal to sampling-midpoint + task offsets
    # PD waypoint control (matches builderbench ppo_pd / sgcrl CRL jobs).
    use_pd: bool = False
    pd_duration: int = 5

    # Allegro / Isaac Gym (ignored for BuilderBench env_id). Bootstrap reads
    # the same --isaacgym-* argv before JAX import; keep names aligned.
    isaacgym_table_push: bool = False
    isaacgym_table_push_xyz: str = '0.20,-0.15,0.555'
    isaacgym_randomize_init: bool = False
    isaacgym_randomize_object_xyz: bool = True
    isaacgym_randomize_object_shape: bool = False
    isaacgym_episode_length: int = 300
    isaacgym_pipeline: str = 'gpu'
    isaacgym_palm_goal: bool = False
    isaacgym_palm_goal_xyz: str = '0.17,0.08,0.57'
    isaacgym_palm_and_object_success: bool = False
    isaacgym_throw_success: str = 'in_bucket'
    isaacgym_large_table: bool = False
    # Tableside spawn (bootstrap reads the same argv before JAX).
    isaacgym_table_spawn: bool = False
    isaacgym_table_spawn_object_xy: str = '0.0,0.0'
    isaacgym_table_spawn_behind: bool = False
    isaacgym_table_spawn_behind_dy: float = 0.14
    isaacgym_table_spawn_behind_above: float = 0.08
    isaacgym_table_spawn_correlated_xy: float = 0.0
    isaacgym_table_spawn_finger_curl_scale: float = 1.0
    isaacgym_table_spawn_finger_noise: float = 0.0
    isaacgym_table_spawn_arm_noise: float = 0.0
    isaacgym_table_spawn_in_hand: bool = False
    isaacgym_table_spawn_in_hand_offset: float = 0.042
    isaacgym_table_spawn_in_hand_obj_noise: float = 0.012
    isaacgym_table_spawn_in_hand_keep_arm: bool = False
    isaacgym_table_spawn_in_hand_wrist_offset: float = -3.141592653589793
    isaacgym_table_spawn_in_hand_wrist_noise: float = 0.10
    isaacgym_fixed_target_xyz: str = '0.5,-0.3,0.4'
    isaacgym_goal_z: float = -1.0
    isaacgym_hide_table: bool = False
    num_train_videos: int = 0
    # Object-free control sanity (off|finger|hand16|hand16fig|hand16ok|...).
    # Bootstrap reads the same argv before JAX; keep the spelling aligned.
    isaacgym_control_sanity_mode: str = 'off'
    # Restrict state+action to goal hand joints (bootstrap argv).
    isaacgym_control_sanity_trim_sa: bool = False
    # Trim-SA finger init: curled (near-default band) or full_range (URDF).
    isaacgym_control_sanity_trim_init_mode: str = 'curled'
    # Obs/goal/action coords: mixed | physical | fully_scaled (bootstrap argv).
    isaacgym_coordinate_mode: str = 'mixed'

    # algorithm
    num_timesteps: int = 50000000
    policy_hidden_sizes: list = field(default_factory=lambda: [256, 256, 256, 256])
    value_hidden_sizes: list = field(default_factory=lambda: [256, 256, 256, 256])
    # Opt-in architecture matching contrastive/networks.py::ResidualMLP:
    # 6x256 hidden layers, LayerNorm + Swish, with skips every two layers.
    use_residual_mlp: bool = False
    # PD-only hybrid actor: shared xyz/yaw Gaussian plus categorical cube select.
    categorical_select: bool = False
    rollout_length: int = 160
    num_minibatches_per_rollout: int = 32
    num_epochs_per_rollout: int = 8    
    learning_rate: float = 1e-4
    discount: float = 0.99
    int_discount: float = 0.99
    int_loss_cost: float = 1.0
    ext_loss_cost: float = 2.0
    entropy_cost: float = 0.05
    entropy_cost_final: float = 0.01   # linearly annealed to this by end of training
    reward_scaling: float = 1.0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    normalize_advantage: bool = True

    diagnostic: bool = False

@flax.struct.dataclass
class PPOTrainingState(TrainState):
  """Contains training state for the learner."""
  int_reward_normalizer_params: Any
  normalizer_params: Any
  env_steps: float

class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()

@flax.struct.dataclass
class PPONetworks:
    policy_network: Any
    value_network: Any
    int_value_network: Any
    rnd_network: Any


_RESIDUAL_HIDDEN_SIZES = (256,) * 6
_RESIDUAL_SKIP_EVERY = 2


class ResidualMLP(nn.Module):
    """Flax equivalent of contrastive.networks.ResidualMLP."""

    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    skip_every: int = _RESIDUAL_SKIP_EVERY
    use_layer_norm: bool = True
    activate_final: bool = False

    def get_penultimate(self, data):
        return self(data, return_penultimate=True)

    @nn.compact
    def __call__(self, data, return_penultimate=False):
        hidden = data
        skip = None
        for i, hidden_size in enumerate(self.layer_sizes[:-1]):
            hidden = nn.Dense(hidden_size, name=f"linear_{i}")(hidden)
            if self.use_layer_norm:
                hidden = nn.LayerNorm(name=f"ln_{i}")(hidden)
            hidden = self.activation(hidden)
            if self.skip_every and (i + 1) % self.skip_every == 0:
                if skip is not None and skip.shape[-1] == hidden.shape[-1]:
                    hidden = skip + hidden
                skip = hidden
        if return_penultimate:
            return hidden
        final_index = len(self.layer_sizes) - 1
        hidden = nn.Dense(
            self.layer_sizes[-1], name=f"linear_{final_index}")(hidden)
        if self.activate_final:
            hidden = self.activation(hidden)
        return hidden


def _select_cube_centers(num_cubes):
    ids = jnp.arange(num_cubes, dtype=jnp.float32)
    return (((2.0 * ids + 1.0) * jnp.pi / num_cubes) - jnp.pi) / jnp.pi


def _select_action_to_cube_id(select, num_cubes):
    bins = jnp.arange(1, num_cubes + 1) * (2.0 * jnp.pi / num_cubes)
    cube_id = jnp.digitize(jnp.pi * select + jnp.pi, bins)
    return jnp.clip(cube_id, 0, num_cubes - 1)


class HybridSelectDistribution:
    """Tanh-Gaussian continuous controls plus categorical PD cube select."""

    hybrid_select = True

    def __init__(self, continuous_dist, categorical_dist, num_select_classes):
        self.continuous_dist = continuous_dist
        self.categorical_dist = categorical_dist
        self.num_select_classes = int(num_select_classes)
        self._bijector = distrax.Tanh()
        self._centers = _select_cube_centers(self.num_select_classes)

    def _pack(self, raw_continuous, cube_id):
        continuous = self._bijector.forward(raw_continuous)
        select = self._centers[cube_id][..., None]
        return jnp.concatenate([continuous, select], axis=-1)

    def sample(self, seed):
        continuous_key, select_key = jax.random.split(seed)
        raw_continuous = self.continuous_dist.sample(seed=continuous_key)
        cube_id = self.categorical_dist.sample(seed=select_key)
        return self._pack(raw_continuous, cube_id)

    def mode(self):
        return self._pack(
            self.continuous_dist.mode(), self.categorical_dist.mode())

    def log_prob(self, action):
        continuous = jnp.clip(
            action[..., :-1], -1.0 + 1e-6, 1.0 - 1e-6)
        raw_continuous = self._bijector.inverse(continuous)
        continuous_log_prob = self.continuous_dist.log_prob(raw_continuous)
        continuous_log_prob -= self._bijector.forward_log_det_jacobian(
            raw_continuous)
        continuous_log_prob = jnp.sum(continuous_log_prob, axis=-1)
        cube_id = _select_action_to_cube_id(
            action[..., -1], self.num_select_classes)
        return continuous_log_prob + self.categorical_dist.log_prob(cube_id)

    def entropy(self, seed):
        # One-sample estimate of entropy for the transformed joint policy.
        return -self.log_prob(self.sample(seed))


def _network_class(use_residual_mlp):
    return ResidualMLP if use_residual_mlp else MLP


def _hidden_sizes(configured_sizes, use_residual_mlp):
    if use_residual_mlp:
        return list(_RESIDUAL_HIDDEN_SIZES)
    return list(configured_sizes)


def categorical_select_classes(args):
    """Return cube count for a valid CatSelect run, or None when disabled."""
    if not args.categorical_select:
        return None
    if not args.use_pd:
        raise ValueError("--categorical-select requires --use-pd")
    match = re.fullmatch(r"creative-(\d+)-task\d+", args.env_id)
    if match is None:
        raise ValueError(
            "--categorical-select requires a creative-N-taskK BuilderBench env")
    return int(match.group(1))


def make_ppo_networks(args, action_size, include_auxiliary=True):
    """Build PPO+RND modules with architecture inferred from Args."""
    num_select_classes = categorical_select_classes(args)
    if num_select_classes is not None and action_size != 5:
        raise ValueError(
            "CatSelect expects the 5D PD action [xyz, yaw, select], "
            f"but the environment has action_size={action_size}")
    policy_hidden_sizes = _hidden_sizes(
        args.policy_hidden_sizes, args.use_residual_mlp)
    value_hidden_sizes = _hidden_sizes(
        args.value_hidden_sizes, args.use_residual_mlp)
    policy_output_size = (
        2 * (action_size - 1) + num_select_classes
        if num_select_classes is not None else action_size * 2
    )
    policy = Actor(
        layer_sizes=policy_hidden_sizes + [policy_output_size],
        use_residual_mlp=args.use_residual_mlp,
        num_select_classes=num_select_classes or 0,
    )
    if not include_auxiliary:
        return PPONetworks(policy, None, None, None)
    return PPONetworks(
        policy_network=policy,
        value_network=Value(
            layer_sizes=value_hidden_sizes + [1],
            use_residual_mlp=args.use_residual_mlp,
        ),
        int_value_network=Value(
            layer_sizes=value_hidden_sizes + [1],
            use_residual_mlp=args.use_residual_mlp,
        ),
        rnd_network=RND(
            layer_sizes=value_hidden_sizes + (
                [256] if args.use_residual_mlp else []),
            use_residual_mlp=args.use_residual_mlp,
        ),
    )


class Actor(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = False
    _min_std: float = 0.001
    _var_scale: float = 1
    use_residual_mlp: bool = False
    num_select_classes: int = 0
    
    def setup(self):
        network_cls = _network_class(self.use_residual_mlp)
        if self.use_residual_mlp:
            self.actor_net = network_cls(
                self.layer_sizes, activation=self.activation)
        else:
            self.actor_net = network_cls(
                self.layer_sizes, activation=self.activation,
                layer_norm=self.layer_norm)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
        stats = self.actor_net(x)
        if self.num_select_classes:
            num_continuous = (
                stats.shape[-1] - self.num_select_classes) // 2
            loc = stats[..., :num_continuous]
            raw_scale = stats[..., num_continuous:2 * num_continuous]
            logits = stats[..., 2 * num_continuous:]
            scale = (
                jax.nn.softplus(raw_scale) + self._min_std) * self._var_scale
            return HybridSelectDistribution(
                distrax.Normal(loc=loc, scale=scale),
                distrax.Categorical(logits=logits),
                self.num_select_classes,
            )
        loc, scale = jnp.split(stats, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale

        return distrax.Normal(loc=loc, scale=scale)

    def get_representation(self, x, normalizer_params=None):
        """Returns the activated penultimate layer output."""
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
        representation = self.actor_net.get_penultimate(x)
        return representation

class Value(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = False
    use_residual_mlp: bool = False
    
    def setup(self):
        network_cls = _network_class(self.use_residual_mlp)
        if self.use_residual_mlp:
            self.value_net = network_cls(
                self.layer_sizes, activation=self.activation)
        else:
            self.value_net = network_cls(
                self.layer_sizes, activation=self.activation,
                layer_norm=self.layer_norm)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
        value = self.value_net(x)
        return jnp.squeeze(value, axis=-1)
    
    def get_representation(self, x, normalizer_params=None):
        """Returns the activated penultimate layer output."""
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
        representation = self.value_net.get_penultimate(x)
        return representation

class RND(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = False
    use_residual_mlp: bool = False

    def setup(self):
        network_cls = _network_class(self.use_residual_mlp)
        if self.use_residual_mlp:
            self.prediction_net = network_cls(
                self.layer_sizes, activation=self.activation)
            self.target_net = network_cls(
                self.layer_sizes, activation=self.activation)
        else:
            self.prediction_net = network_cls(
                self.layer_sizes, activation=self.activation,
                layer_norm=self.layer_norm)
            self.target_net = network_cls(
                self.layer_sizes, activation=self.activation,
                layer_norm=self.layer_norm)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
            x = jnp.clip(x, -5, 5)
        prediction = self.prediction_net(x)
        target = jax.lax.stop_gradient( (self.target_net(x)) )
        return prediction, target

def make_inference_fn(ppo_networks):
    """Creates params and inference function for the PPO agent."""
    def make_policy(params, deterministic: bool = False):
        policy_network = ppo_networks.policy_network
        bijector = distrax.Tanh()  

        def policy(observations, goals, key_sample):
            inputs = jnp.concatenate([observations, goals], axis=-1)
            policy_dist = policy_network.apply(params['policy'], inputs, params['normalizer'])

            if getattr(policy_dist, "hybrid_select", False):
                actions = (
                    policy_dist.mode() if deterministic
                    else policy_dist.sample(seed=key_sample)
                )
                return actions, {
                    'log_prob': policy_dist.log_prob(actions),
                    # PPO stores this field for both policy types.
                    'raw_action': actions,
                }

            if deterministic:
                return bijector.forward( policy_dist.mode() ), {}
                
            raw_actions = policy_dist.sample(seed=key_sample)
                
            log_prob = policy_dist.log_prob(raw_actions) - bijector.forward_log_det_jacobian(raw_actions)
            log_prob = jnp.sum(log_prob, axis=-1)  
            postprocessed_actions = bijector.forward(
                raw_actions
            )
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions,
            }

        return policy
    return make_policy


def is_allegro_env(env_id: str) -> bool:
    return str(env_id).startswith('allegro_kuka')


def _parse_xyz(text: str):
    xyz = tuple(float(x.strip()) for x in str(text).split(',') if x.strip())
    if len(xyz) != 3:
        raise ValueError(f'expected 3 floats, got {text!r}')
    return xyz


def _torch_to_np(tensor):
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)


def _allegro_obs_goal(packed: np.ndarray, obs_dim: int):
    packed = np.asarray(packed, dtype=np.float32)
    return packed[:, :obs_dim], packed[:, obs_dim:]


def allegro_generate_unroll(
    env,
    packed_obs,
    policy,
    key,
    unroll_length: int,
    obs_dim: int,
):
    """Host-side Isaac Gym collect; policy is JAX."""
    import torch

    device = env.device
    act_low = -1.0
    act_high = 1.0
    obs_t, act_t, rew_t, disc_t, next_t = [], [], [], [], []
    logp_t, raw_t, trunc_t, succ_t = [], [], [], []
    packed = np.asarray(packed_obs, dtype=np.float32)
    for _ in range(unroll_length):
        key, k_step = jax.random.split(key)
        state, goal = _allegro_obs_goal(packed, obs_dim)
        actions, extras = policy(
            jnp.asarray(state), jnp.asarray(goal), k_step)
        actions_np = np.clip(
            np.asarray(actions, dtype=np.float32), act_low, act_high)
        act_torch = torch.from_numpy(actions_np).to(device)
        next_obs_t, _, done_t = env.step(act_torch)
        success_t = env.success()
        next_np = _torch_to_np(next_obs_t)
        done_np = _torch_to_np(done_t).reshape(-1)
        succ_np = _torch_to_np(success_t).reshape(-1)
        obs_t.append(packed)
        act_t.append(actions_np)
        rew_t.append(succ_np)
        disc_t.append(1.0 - done_np)
        next_t.append(next_np)
        logp_t.append(np.asarray(extras['log_prob'], dtype=np.float32))
        raw_t.append(np.asarray(extras['raw_action'], dtype=np.float32))
        trunc_t.append(done_np)
        succ_t.append(succ_np)
        packed = next_np
    stacked = lambda xs: jnp.asarray(np.stack(xs, axis=0))
    transition = Transition(
        observation=stacked(obs_t),
        action=stacked(act_t),
        reward=stacked(rew_t),
        discount=stacked(disc_t),
        next_observation=stacked(next_t),
        extras={
            'policy_extras': {
                'log_prob': stacked(logp_t),
                'raw_action': stacked(raw_t),
            },
            'state_extras': {
                'truncation': stacked(trunc_t),
            },
        },
    )
    metrics = {'success': stacked(succ_t)}
    return packed, transition, metrics, key


def allegro_run_eval(env, policy, key, episode_length: int, obs_dim: int):
    """Deterministic episode on the train sim (one PhysX instance per process)."""
    import torch

    packed = _torch_to_np(env.reset())
    ep_success = np.zeros((env.num_envs,), dtype=np.float32)
    ep_easy = np.zeros((env.num_envs,), dtype=np.float32)
    ep_very_easy = np.zeros((env.num_envs,), dtype=np.float32)
    ep_jfrac10 = np.zeros((env.num_envs,), dtype=np.float32)
    ep_jfrac20 = np.zeros((env.num_envs,), dtype=np.float32)
    ep_mae = np.full((env.num_envs,), np.inf, dtype=np.float32)
    step_success = []
    has_levels = callable(getattr(env, 'success_levels', None))
    for _ in range(int(episode_length)):
        key, k_step = jax.random.split(key)
        state, goal = _allegro_obs_goal(packed, obs_dim)
        actions, _ = policy(jnp.asarray(state), jnp.asarray(goal), k_step)
        actions_np = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        next_obs_t, _, _ = env.step(
            torch.from_numpy(actions_np).to(env.device))
        if has_levels:
            levels = env.success_levels()
            succ = _torch_to_np(levels['hard']).reshape(-1)
            easy = _torch_to_np(levels['easy']).reshape(-1)
            veasy = _torch_to_np(levels['very_easy']).reshape(-1)
            j10 = _torch_to_np(levels['joint_frac_010']).reshape(-1)
            j20 = _torch_to_np(levels['joint_frac_020']).reshape(-1)
            mae = _torch_to_np(levels['mean_abs_joint_err']).reshape(-1)
            ep_easy = np.maximum(ep_easy, easy)
            ep_very_easy = np.maximum(ep_very_easy, veasy)
            ep_jfrac10 = np.maximum(ep_jfrac10, j10)
            ep_jfrac20 = np.maximum(ep_jfrac20, j20)
            ep_mae = np.minimum(ep_mae, mae)
        else:
            succ = _torch_to_np(env.success()).reshape(-1)
        ep_success = np.maximum(ep_success, succ)
        step_success.append(succ)
        packed = _torch_to_np(next_obs_t)
    packed = _torch_to_np(env.reset())
    out = {
        'eval/episode_success': float(ep_success.mean()),
        'eval/success_mean': float(np.mean(step_success)),
        'eval/num_envs': int(env.num_envs),
    }
    if has_levels:
        out.update({
            'eval/easy_success': float(ep_easy.mean()),
            'eval/very_easy_success': float(ep_very_easy.mean()),
            'eval/joint_frac_010': float(ep_jfrac10.mean()),
            'eval/joint_frac_020': float(ep_jfrac20.mean()),
            'eval/mean_abs_joint_err': float(
                np.mean(ep_mae[np.isfinite(ep_mae)])
                if np.any(np.isfinite(ep_mae)) else float('nan')),
        })
    return packed, key, out


def main(args: Args):
    categorical_select_classes(args)
    use_allegro = is_allegro_env(args.env_id)
    if use_allegro and args.use_pd:
        raise ValueError('Allegro Isaac Gym does not support --use-pd')
    if use_allegro and args.categorical_select:
        raise ValueError('Allegro Isaac Gym does not support --categorical-select')

    args.num_training_step = args.num_timesteps // ( args.num_envs * args.rollout_length )
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps
    args.num_training_steps_per_real_reset = args.num_training_step // max(1, args.num_reset_steps)
    args.minibatch_size = args.num_envs * args.rollout_length // ( args.num_minibatches_per_rollout )
        
    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = {args.num_minibatches_per_rollout * args.num_epochs_per_rollout}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")
    print(f"Data to update ratio = {  ( args.num_envs * args.rollout_length ) / (args.num_minibatches_per_rollout * args.num_epochs_per_rollout)}")    

    args.exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag != '' else ''}{args.env_id}__{args.seed}__{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"
    
    # Initialize wandb if tracking is enabled
    if args.track:
        global wandb, wandb_osh, TriggerWandbSyncHook
        import wandb as _wandb
        wandb = _wandb
        try:
            import wandb_osh as _wandb_osh
            from wandb_osh.hooks import TriggerWandbSyncHook as _Trigger
            wandb_osh = _wandb_osh
            TriggerWandbSyncHook = _Trigger
        except ImportError:
            wandb_osh = None
            TriggerWandbSyncHook = None
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

        if args.wandb_mode == 'offline':
            if wandb_osh is None or TriggerWandbSyncHook is None:
                raise ImportError('offline wandb sync requires wandb_osh')
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
        
    key = jax.random.PRNGKey(args.seed)
    local_key, key_env, key_eval, key_policy, key_value, key_rnd = jax.random.split(key, 6)

    # Initialize environment
    allegro_env = None
    allegro_packed = None
    evaluator = None
    reset_fn = None
    env_state = None
    log_data_metric_keys = ()
    if use_allegro:
        allegro_env = _ig_take()
        if allegro_env is None:
            raise RuntimeError(
                'Allegro PhysX sim was not created before JAX. '
                'Pass --env-id=allegro_kuka_throw (or *_slide) so '
                'isaacgym_physx_bootstrap can run at import time.')
        args.num_envs = int(allegro_env.num_envs)
        args.num_reset_steps = 0
        episode_length = int(getattr(
            allegro_env, 'max_episode_steps', args.isaacgym_episode_length))
        obs_size = int(allegro_env.obs_dim)
        goal_size = int(allegro_env.goal_dim)
        action_size = int(allegro_env.action_dim)
        allegro_packed = _torch_to_np(allegro_env.reset())
        push_xyz = _parse_xyz(args.isaacgym_table_push_xyz)
        sanity = str(getattr(allegro_env, 'control_sanity_mode', '') or 'off')
        if sanity and sanity != 'off':
            rew_msg = f'reward=env.success() (control_sanity={sanity} hard)'
        else:
            rew_msg = 'reward=env.success() (object within 7.5cm of goal)'
        print(
            f'[ppo_rnd] allegro env_id={args.env_id} E={args.num_envs} '
            f'ep_len={episode_length} obs={obs_size} goal={goal_size} '
            f'act={action_size} table_push={bool(allegro_env.table_push)} '
            f'table_push_xyz={push_xyz} '
            f'control_sanity={sanity} '
            f'large_table={bool(getattr(allegro_env, "large_table", False))} '
            f'table_spawn={bool(args.isaacgym_table_spawn)} '
            f'table_spawn_behind={bool(args.isaacgym_table_spawn_behind)} '
            f'behind_dy={args.isaacgym_table_spawn_behind_dy:g} '
            f'behind_above={args.isaacgym_table_spawn_behind_above:g} '
            f'correlated_xy={args.isaacgym_table_spawn_correlated_xy:g} '
            f'finger_curl={args.isaacgym_table_spawn_finger_curl_scale:g} '
            f'finger_noise={args.isaacgym_table_spawn_finger_noise:g} '
            f'arm_noise={args.isaacgym_table_spawn_arm_noise:g} '
            f'randomize_init={bool(args.isaacgym_randomize_init)} '
            f'randomize_object_xyz={bool(allegro_env.randomize_object_xyz)} '
            f'randomize_object_shape={bool(allegro_env.randomize_object_shape)} '
            f'{rew_msg}',
            flush=True)
    else:
        from builderbench.env_utils import make_env
        from utils.evaluation import Evaluator
        from utils.wrapper import wrap_env, PDWrapper

        env_class, default_config = make_env(args)
        # BuilderBench defaults to MJX warp; prefer jax unless warp is available.
        default_config.impl = os.environ.get("BUILDERBENCH_MJX_IMPL", "jax")
        default_config.permute_start_boxes = args.permute_start_boxes
        print(f"MJX impl={default_config.impl}")
        print(f"permute_start_boxes={args.permute_start_boxes} "
              f"fixed_start_x={args.fixed_start_x} fix_goal={args.fix_goal}")
        _fixed_goal = None
        if args.fix_goal:
            import re as _re
            from envs.builderbench_utils import default_fixed_target_goal
            _nc = int(_re.search(r"creative-(\d+)", args.env_id).group(1))
            _ti = int(_re.search(r"task(\d+)", args.env_id).group(1)) - 1
            _fixed_goal = jnp.asarray(
                default_fixed_target_goal(_nc, _ti), dtype=jnp.float32)
            print(f"fix_goal=True  fixed_goal={_fixed_goal.tolist()}")
            print("eval uses the same fixed target_goal as train")
        def _make_controlled_env():
          from envs.builderbench_utils import apply_fixed_start_x
          base = env_class(config=default_config)
          apply_fixed_start_x(base, args.fixed_start_x if args.fixed_start_x >= 0 else None)
          if args.use_pd:
            base = PDWrapper(base, duration=args.pd_duration)
          if _fixed_goal is not None:
            base = FixedGoalWrapper(base, _fixed_goal)
          return base

        if args.use_pd:
          assert default_config.episode_length % args.pd_duration == 0, (
              "Environment episode length must be divisible by pd_duration")
          episode_length = default_config.episode_length // args.pd_duration
          print(f"Control mode: PD waypoint controls (pd_duration={args.pd_duration}) "
                f"macro_ep_len={episode_length}")
        else:
          episode_length = default_config.episode_length
          print("Control mode: raw controls")
        env = wrap_env(HardSuccessRewardWrapper(_make_controlled_env()), episode_length)
        eval_env = wrap_env(
            HardSuccessRewardWrapper(_make_controlled_env()), episode_length)

        reset_fn = jax.jit(env.reset)
        key_envs = jax.random.split(key_env, args.num_envs)
        env_state = reset_fn(key_envs)
        obs_size = env.observation_size
        action_size = env.action_size
        goal_size = env.goal_size

        log_data_metric_keys = []
        for k in ("obj_reached_once", "obj_lifted", "obj_moved",
                  "success", "easy_success", "very_hard_success"):
            if k in env_state.metrics.keys():
                log_data_metric_keys.append(k)
        log_data_metric_keys = tuple(log_data_metric_keys)

    # Initialize checkpoint folder
    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f"checkpoints/{args.exp_name}/"
        os.makedirs(save_path, exist_ok=True)

    # Initialize PPO networks
    ppo_network = make_ppo_networks(args, action_size)
    training_state = PPOTrainingState.create(
        apply_fn=None,
        params={
            'policy': ppo_network.policy_network.init( key_policy, x=jnp.zeros((1, obs_size+goal_size)) ),
            'value': ppo_network.value_network.init( key_value, x=jnp.zeros((1, obs_size+goal_size)) ),
            'int_value': ppo_network.int_value_network.init( key_value, x=jnp.zeros((1, obs_size+goal_size)) ),
            'rnd': ppo_network.rnd_network.init( key_value, x=jnp.zeros((1, obs_size+goal_size)) ),
        },
        tx=optax.adam(learning_rate=args.learning_rate),  
        normalizer_params=running_statistics.init_state((obs_size+goal_size,) ),
        int_reward_normalizer_params=running_statistics.init_state(()),
        env_steps=np.zeros((), dtype=np.float64),
    )
    make_policy = make_inference_fn(ppo_network)

    print(f'\nNumber of parameters in actor critic network are: {count_parameters(training_state.params)}\n')

    # Initialize evaluators (BuilderBench only; Allegro eval uses the train sim).
    evaluator = None
    if not use_allegro:
        evaluator = Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=True),
            num_eval_envs=args.num_eval_envs,
            episode_length=episode_length,
            key=key_eval,
        )

    def generate_unroll(
        env,
        env_state,
        policy,
        key,
        unroll_length,
        extra_fields,
    ):
        """Collect trajectories of given unroll_length."""        
        @jax.jit
        def f(carry, unused_t):
            env_state, key = carry
            key, next_key = jax.random.split(key)
            actions, policy_extras = policy(env_state.obs, env_state.info['target_goal'], key)  
            
            next_env_state = env.step(env_state, actions)

            state_extras = {x: next_env_state.info[x] for x in extra_fields}

            metrics = {x: next_env_state.metrics[x] for x in log_data_metric_keys}

            transition = Transition(
                observation=jnp.concatenate( [env_state.obs, env_state.info['target_goal']], axis=-1),
                action=actions,
                reward=next_env_state.reward,
                discount=1 - next_env_state.done,
                next_observation=jnp.concatenate( [next_env_state.obs, next_env_state.info['target_goal']], axis=-1),
                extras={'policy_extras': policy_extras, 'state_extras': state_extras},
            )
            
            return (next_env_state, next_key), (transition, metrics)

        (final_env_state, _), (data, data_metrics) = jax.lax.scan(
            f, (env_state, key), (), length=unroll_length
        )
        return final_env_state, data, data_metrics

    @jax.jit
    def data_collect_step(training_state, env_state, key_generate_rollout):
        policy = make_policy({
            'policy': training_state.params['policy'], 
            'normalizer': training_state.normalizer_params,
            })
        
        env_state, data, data_metrics = generate_unroll(
                env,
                env_state,
                policy,
                key_generate_rollout,
                args.rollout_length,
                extra_fields=('truncation',),
            )
        
        # Updating collected data with intrinsic rewards
        int_prediction, int_target = ppo_network.rnd_network.apply(training_state.params['rnd'], data.next_observation, training_state.normalizer_params)
        int_rewards = jnp.sum( (int_prediction - int_target) ** 2, axis=-1) / 2
        data.extras['policy_extras']['int_reward'] = int_rewards
        
        # Update normalization params.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
        )
        int_reward_normalizer_params = running_statistics.update(
            training_state.int_reward_normalizer_params,
            jnp.sum( int_rewards * ( args.int_discount ** jnp.arange(args.rollout_length)[:, None] ), axis=0),
        )

        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            int_reward_normalizer_params=int_reward_normalizer_params,
            env_steps=training_state.env_steps + args.rollout_length * args.num_envs,
        )

        return training_state, env_state, data, data_metrics

    @jax.jit
    def attach_rnd_rewards(training_state, data):
        int_prediction, int_target = ppo_network.rnd_network.apply(
            training_state.params['rnd'], data.next_observation,
            training_state.normalizer_params)
        int_rewards = jnp.sum((int_prediction - int_target) ** 2, axis=-1) / 2
        data.extras['policy_extras']['int_reward'] = int_rewards
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
        )
        int_reward_normalizer_params = running_statistics.update(
            training_state.int_reward_normalizer_params,
            jnp.sum(
                int_rewards * (
                    args.int_discount ** jnp.arange(args.rollout_length)[:, None]
                ),
                axis=0,
            ),
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            int_reward_normalizer_params=int_reward_normalizer_params,
            env_steps=training_state.env_steps + args.rollout_length * args.num_envs,
        )
        return training_state, data

    def allegro_data_collect_step(training_state, packed, key_generate_rollout):
        policy = jax.jit(make_policy({
            'policy': training_state.params['policy'],
            'normalizer': training_state.normalizer_params,
        }))
        packed, data, data_metrics, key_generate_rollout = allegro_generate_unroll(
            allegro_env, packed, policy, key_generate_rollout,
            args.rollout_length, obs_size)
        training_state, data = attach_rnd_rewards(training_state, data)
        return training_state, packed, data, data_metrics
    
    def compute_gae(
        truncation: jnp.ndarray,
        termination: jnp.ndarray,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        bootstrap_value: jnp.ndarray,
        lambda_: float = 1.0,
        discount: float = 0.99,
    ):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = jnp.concatenate(
            [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = jnp.zeros_like(bootstrap_value)
        vs_minus_v_xs = []

        def compute_vs_minus_v_xs(carry, target_t):
            lambda_, acc = carry
            truncation_mask, delta, termination = target_t
            acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
            return (lambda_, acc), (acc)

        (_, _), (vs_minus_v_xs) = jax.lax.scan(
            compute_vs_minus_v_xs,
            (lambda_, acc),
            (truncation_mask, deltas, termination),
            length=int(truncation_mask.shape[0]),
            reverse=True,
        )
        # Add V(x_s) to get v_s.
        vs = jnp.add(vs_minus_v_xs, values)

        vs_t_plus_1 = jnp.concatenate(
            [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        advantages = (
            rewards + discount * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


    def compute_ppo_loss(
        params,
        normalizer_params,
        init_normalizer_params,
        data,
        rng,
        entropy_cost,
    ):
        bijector = distrax.Tanh()  
        policy_apply = ppo_network.policy_network.apply
        value_apply = ppo_network.value_network.apply
        int_value_apply = ppo_network.int_value_network.apply
        rnd_apply = ppo_network.rnd_network.apply

        data, value_targets, int_value_targets, advantages = data

        # Policy function loss
        policy_dist = policy_apply(params['policy'], data.observation, normalizer_params)
        if getattr(policy_dist, "hybrid_select", False):
            target_action_log_probs = policy_dist.log_prob(
                data.extras['policy_extras']['raw_action'])
        else:
            target_action_log_probs = policy_dist.log_prob(
                data.extras['policy_extras']['raw_action'])
            target_action_log_probs -= bijector.forward_log_det_jacobian(
                data.extras['policy_extras']['raw_action'])
            target_action_log_probs = jnp.sum(
                target_action_log_probs, axis=-1)
        behaviour_action_log_probs = data.extras['policy_extras']['log_prob']
        rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = (jnp.clip(rho_s, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon) * advantages)
        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Forward loss — use same normalized+clipped obs as bonus computation
        predict_next_state_feature, target_next_state_feature = rnd_apply(params['rnd'], data.next_observation, normalizer_params)
        forward_loss = jnp.mean( (predict_next_state_feature-target_next_state_feature)**2 )

        # Value function loss
        baseline = value_apply(params['value'], data.observation, normalizer_params)
        v_error = value_targets - baseline
        v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

        # Intrinsic Value function loss
        int_baseline = int_value_apply(params['int_value'], data.observation, normalizer_params)
        int_v_error = int_value_targets - int_baseline
        int_v_loss = jnp.mean(int_v_error * int_v_error) * 0.5 * 0.5

        # Entropy loss
        if getattr(policy_dist, "hybrid_select", False):
            entropy = jnp.mean(policy_dist.entropy(seed=rng))
        else:
            entropy = policy_dist.entropy() + bijector.forward_log_det_jacobian(
                policy_dist.sample(seed=rng))
            entropy = jnp.mean(jnp.sum(entropy, axis=-1))
        entropy_loss = entropy_cost * -entropy

        total_loss = policy_loss + v_loss + int_v_loss + entropy_loss + forward_loss
        return total_loss, {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'v_loss': v_loss,
            'int_v_loss': int_v_loss,
            'entropy_loss': entropy_loss,
            'forward_loss': forward_loss,
        }
    
    @jax.jit
    def learn_step(training_state, data, key_sgd, entropy_cost):

        def _learn_step(carry, unused_t):
            
            def _train_minibatch_step(carry, data):
                training_state, key = carry
                key, key_loss = jax.random.split(key)
                
                (_, metrics), grads = jax.value_and_grad(compute_ppo_loss, has_aux=True)(training_state.params, training_state.normalizer_params, training_state.int_reward_normalizer_params, data, key_loss, entropy_cost)
                training_state = training_state.apply_gradients(grads=grads)
                
                return (training_state, key), metrics
            
            training_state, data, value_targets, int_value_targets, advantages, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)
        
            def shuffle_and_reshape(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (args.num_minibatches_per_rollout, -1) + x.shape[2:])
                return x

            batch_data = ( data, value_targets, int_value_targets, advantages )
            shuffled_batch_data = jax.tree_util.tree_map(shuffle_and_reshape, batch_data)

            (training_state, _), metrics = jax.lax.scan(
                _train_minibatch_step,
                (training_state, key_grad),
                shuffled_batch_data,
                length=args.num_minibatches_per_rollout,
            )
            return (training_state, data, value_targets, int_value_targets, advantages, key), metrics

        # calculate gae
        baseline = ppo_network.value_network.apply(training_state.params['value'], data.observation, training_state.normalizer_params)
        int_baseline = ppo_network.int_value_network.apply(training_state.params['int_value'], data.observation, training_state.normalizer_params)

        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        bootstrap_value = ppo_network.value_network.apply(training_state.params['value'], terminal_obs, training_state.normalizer_params)
        int_bootstrap_value = ppo_network.int_value_network.apply(training_state.params['int_value'], terminal_obs, training_state.normalizer_params)

        rewards = data.reward * args.reward_scaling
        int_rewards = data.extras['policy_extras']['int_reward'] / training_state.int_reward_normalizer_params.std
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)

        value_targets, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=args.gae_lambda,
            discount=args.discount,
        )
        int_value_targets, int_advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=int_rewards,
            values=int_baseline,
            bootstrap_value=int_bootstrap_value,
            lambda_=args.gae_lambda,
            discount=args.int_discount,
        )

        advantages = int_advantages * args.int_loss_cost + advantages * args.ext_loss_cost
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        (training_state, _, _, _, _, _), metrics = jax.lax.scan(
            _learn_step,
            (training_state, data, value_targets, int_value_targets, advantages, key_sgd),
            (),
            length=args.num_epochs_per_rollout,
        )

        return training_state, metrics

    def compute_feature_diagnostics(features):
        # adapted from https://github.com/roger-creus/stable-deep-rl-at-scale/blob/main/src/models/agent.py
        """Computes different approximations of the rank of the feature matrices.

        Args:
            feature_matrices (torch.Tensor): A tensor of shape (B_matrices, N_obs, D_dims).

        (1) Effective rank.
        A continuous approximation of the rank of a matrix.
        Definition 2.1. in Roy & Vetterli, (2007) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7098875
        Also used in Huh et al. (2023) https://arxiv.org/pdf/2103.10427.pdf

        (2) Approximate rank.
        Threshold at the dimensions explaining 99% of the variance in a PCA analysis.
        Section 2 in Yang et al. (2020) https://arxiv.org/pdf/1909.12255.pdf

        (3) srank.
        Another (incorrect?) version of (2).
        Section 3 in Kumar et al. https://arxiv.org/pdf/2010.14498.pdf

        (4) Feature rank.
        A threshold rank: normalize by dim size and discard dimensions with singular values below 0.01.
        Equations (4) and (5). Lyle et al. (2022) https://arxiv.org/pdf/2204.09560.pdf
        """

        feature_matrices = jnp.expand_dims(features, axis=0)

        cutoff = 0.01
        threshold = 1 - cutoff

        # svals shape: (1, K) where K = min(N_obs, D_dims)
        svals = jnp.linalg.svdvals(feature_matrices)

        # (1) Effective rank. Roy & Vetterli (2007)
        sval_sum = jnp.sum(svals, axis=1)  # Shape: (1,)
        sval_dist = svals / sval_sum[..., None] # Use [..., None] for unsqueeze
        # Replace 0 with 1 to avoid log(0) = -inf
        sval_dist_fixed = jnp.where(sval_dist == 0, jnp.ones_like(sval_dist), sval_dist)
        effective_ranks = jnp.exp(-jnp.sum(sval_dist_fixed * jnp.log(sval_dist_fixed), axis=1))

        # (2) Approximate rank. PCA variance. Yang et al. (2020)
        sval_squares = svals**2
        sval_squares_sum = jnp.sum(sval_squares, axis=1) # Shape: (1,)
        cumsum_squares = jnp.cumsum(sval_squares, axis=1)
        threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum[..., None])
        # Use jnp.logical_not for '~' on boolean arrays
        approximate_ranks = jnp.logical_not(threshold_crossed).sum(axis=-1) + 1

        # (3) srank. Weird. Kumar et al. (2020)
        cumsum = jnp.cumsum(svals, axis=1)
        threshold_crossed_srank = cumsum >= threshold * sval_sum[..., None]
        sranks = jnp.logical_not(threshold_crossed_srank).sum(axis=-1) + 1

        # (4) Feature rank. Most basic. Lyle et al. (2022)
        # Get N_obs directly from the shape
        n_obs = jnp.array(feature_matrices.shape[1], dtype=svals.dtype) 
        svals_of_normalized = svals / jnp.sqrt(n_obs)
        over_cutoff = svals_of_normalized > cutoff
        feature_ranks = over_cutoff.sum(axis=-1)

        return {
            'effective_rank_vetterli': effective_ranks,
            'approximate_rank_pca': approximate_ranks,
            'srank_kumar': sranks,
            'feature_rank_lyle': feature_ranks,
        }

    @jax.jit
    def extra_log_step(training_state, data, key_extra_log):
        
        bijector = distrax.Tanh()
        policy_apply = ppo_network.policy_network.apply
        value_apply = ppo_network.value_network.apply

        def _extra_log_step(carry, data):
            policy_feature = policy_apply(training_state.params['policy'], data.observation, training_state.normalizer_params, method=ppo_network.policy_network.get_representation)
            value_feature = value_apply(training_state.params['value'], data.observation, training_state.normalizer_params, method=ppo_network.value_network.get_representation)
            policy_feature_ranks = compute_feature_diagnostics(policy_feature)
            value_feature_ranks = compute_feature_diagnostics(value_feature)
            
            policy_feature_ranks = {f"policy_{k}": v for k, v in policy_feature_ranks.items()}
            value_feature_ranks = {f"value_{k}": v for k, v in value_feature_ranks.items()}

            return carry, {**policy_feature_ranks, **value_feature_ranks}
        
        (_), metrics = jax.lax.scan(
                _extra_log_step,
                (key_extra_log),
                data,
                length=args.rollout_length,
            )

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return metrics

    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    xt = time.time()
    metrics = None
    for ts in range(1, args.num_training_step + 1):
        
        key_sgd, key_generate_unroll, key = jax.random.split(key, 3)

        data_collect_start = time.time()
        if use_allegro:
            training_state, allegro_packed, training_data, data_metrics = (
                allegro_data_collect_step(
                    training_state, allegro_packed, key_generate_unroll))
        else:
            training_state, env_state, training_data, data_metrics = (
                data_collect_step(
                    training_state, env_state, key_generate_unroll))
        data_collect_step_time += time.time() - data_collect_start
        
        learn_step_start = time.time()
        frac = (ts - 1) / max(1, args.num_training_step - 1)
        current_entropy_cost = jnp.float32(
            args.entropy_cost + (args.entropy_cost_final - args.entropy_cost) * frac)
        training_state, training_metrics = learn_step(training_state, training_data, key_sgd, current_entropy_cost)
        learn_step_time += time.time() - learn_step_start

        if metrics is None:
            metrics = {**data_metrics, **training_metrics}
        else:
            metrics = jax.tree_util.tree_map(
                lambda x, y: x + y, metrics, {**data_metrics, **training_metrics}
            )

        if (not use_allegro and args.num_reset_steps > 0
                and ts % args.num_training_steps_per_real_reset == 0):
            key_env, key = jax.random.split(key, 2)
            key_envs = jax.random.split(key_env, args.num_envs)
            env_state = reset_fn(key_envs)

        if ts % args.num_training_steps_per_eval == 0:
            es = ts // args.num_training_steps_per_eval
            
            metrics = jax.tree_util.tree_map(
                lambda x: x / args.num_training_steps_per_eval, metrics
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            
            training_step_time = time.time() - xt            
            training_walltime += training_step_time

            sps = (
                args.num_training_steps_per_eval
                * args.num_envs * args.rollout_length
            ) / training_step_time

            if args.diagnostic:
                key_extra_log, key = jax.random.split(key, 2)
                extra_metrics = extra_log_step(training_state, training_data, key_extra_log)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), extra_metrics)
            else:
                extra_metrics = {}

            metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                'training/data_collection_time_fraction' : data_collect_step_time / training_step_time,
                'training/learning_time_fraction' : learn_step_time / training_step_time,
                'training/env_steps': training_state.env_steps,
                'training/entropy_cost': float(current_entropy_cost),
                'normalizer/count' : training_state.normalizer_params.count,
                'normalizer/mean' : jnp.mean( training_state.normalizer_params.mean ),
                'normalizer/summer_variance' : jnp.mean( training_state.normalizer_params.summed_variance ),
                'int_reward_normalizer/std' : jnp.mean( training_state.normalizer_params.std ),
                'int_reward_normalizer/count' : training_state.int_reward_normalizer_params.count,
                'int_reward_normalizer/mean' : jnp.mean( training_state.int_reward_normalizer_params.mean ),
                'int_reward_normalizer/summer_variance' : jnp.mean( training_state.int_reward_normalizer_params.summed_variance ),
                'int_reward_normalizer/std' : jnp.mean( training_state.int_reward_normalizer_params.std ),
                **{f'training/{name}': value for name, value in metrics.items()},
                **{f'diagnostic/{name}': value for name, value in extra_metrics.items()},
            }

            if use_allegro:
                key_eval, key = jax.random.split(key, 2)
                eval_policy = make_policy(
                    {
                        'policy': training_state.params['policy'],
                        'normalizer': training_state.normalizer_params,
                    },
                    deterministic=True,
                )
                allegro_packed, key_eval, eval_metrics = allegro_run_eval(
                    allegro_env, eval_policy, key_eval, episode_length, obs_size)
                metrics.update(eval_metrics)
                # Persist Allegro eval (incl. joint_frac) for offline plots.
                try:
                    import csv as _csv
                    _csv_path = Path(args.wandb_dir) / 'eval_metrics.csv'
                    _csv_path.parent.mkdir(parents=True, exist_ok=True)
                    _row = {
                        'eval_step': int(es),
                        'env_steps': int(training_state.env_steps),
                        **{k: float(v) for k, v in eval_metrics.items()},
                    }
                    _write_header = not _csv_path.exists()
                    with _csv_path.open('a', newline='') as _fh:
                        _w = _csv.DictWriter(_fh, fieldnames=list(_row.keys()))
                        if _write_header:
                            _w.writeheader()
                        _w.writerow(_row)
                except Exception as _csv_exc:
                    print(f'[ppo_rnd] eval_metrics.csv write failed: {_csv_exc}',
                          flush=True)
            else:
                metrics = evaluator.run_evaluation(
                    policy_params={'policy':training_state.params['policy'], 'normalizer':training_state.normalizer_params},
                    training_metrics=metrics,
                )

            print(f'\nEvaluation step {es}:\n')
            pprint.pprint(metrics)
            if args.track:
                wandb.log(metrics, step=es)
                if args.wandb_mode == 'offline':
                    trigger_sync()
            metrics = None

            if args.save_checkpoint:
                save_params(
                    f"{save_path}/params_{es}.pkl", 
                    params = (
                        training_state.params,
                        training_state.normalizer_params,
                        training_state.int_reward_normalizer_params,
                    )
                )
                if use_allegro:
                  _maybe_render_rnd_video(
                      args,
                      f'{save_path}/params_{es}.pkl',
                      int(es),
                      int(training_state.env_steps),
                  )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        wandb.finish()
            
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)