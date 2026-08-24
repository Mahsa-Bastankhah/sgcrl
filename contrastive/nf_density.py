"""Normalizing-flow density estimator  p_θ(g | s, a)  as a CRL drop-in.

Overview
--------
    z, log_det  = RealNVP(G_encoder(goal) ;  y = SA_encoder(s, a))
    log p_NF(g | s, a) = log N(z;0,I) + Σ log|det J|

With ``state_only=True`` the encoder drops the action:
    y = S_encoder(s)   →   log p_NF(g | s)   (PPO reward r(s) instead of r(s,a)).

SA encoder: ``sa_num_layers × Dense(sa_hidden) + LN + swish → Dense(rep_size)``.
  Defaults match the original reference (4×1024). Compact runs use e.g. 3×256.
Goal encoder is a compact MLP: 2×(Dense(256) + LN + swish) → Dense(goal_enc_size).
  - goal_enc_size=0  disables the encoder (raw normalized goal fed to flow, legacy).
  - goal_enc_size>0  encodes goal first; flow operates in goal_enc_size-dim space.
RealNVP matches reference MetaBlocks (InvertiblePLU + affine coupling).

Optimizers (reference nf_sac.py):
    SA encoder   : Adam(lr=actor_lr)      default 3e-4
    Goal encoder : Adam(lr=actor_lr)      default 3e-4  (same as SA encoder)
    NF flow      : AdamW(lr=critic_lr, weight_decay=critic_weight_decay)
                   default 1e-4, 1e-6
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme.jax import networks as networks_lib
from scipy.linalg import lu as scipy_lu


# ---------------------------------------------------------------------------
# SA encoder  (Haiku port of reference SA_encoder)
# ---------------------------------------------------------------------------

def _sa_encoder(state: jnp.ndarray, action: jnp.ndarray,
                rep_size: int,
                sa_hidden: int = 1024,
                sa_num_layers: int = 4,
                state_only: bool = False) -> jnp.ndarray:
    """Conditioning encoder → sa_num_layers×(Dense(sa_hidden) + LN + swish) → Dense(rep_size).

    Default: concat([s, a]).  With state_only=True: state only (action unused).
    """
    lecun = hk.initializers.VarianceScaling(1 / 3, 'fan_in', 'uniform')
    zero_bias = hk.initializers.Constant(0.)

    x = state if state_only else jnp.concatenate([state, action], axis=-1)
    for i in range(int(sa_num_layers)):
        x = hk.Linear(int(sa_hidden), w_init=lecun, b_init=zero_bias,
                      name=f'dense_{i}')(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                         name=f'ln_{i}')(x)
        x = jax.nn.swish(x)
    return hk.Linear(rep_size, w_init=lecun, b_init=zero_bias, name='out')(x)


# ---------------------------------------------------------------------------
# Goal encoder  (compact: 2×(Dense(256)+LN+swish) → Dense(goal_enc_size))
# ---------------------------------------------------------------------------

def _goal_encoder(goal: jnp.ndarray, goal_enc_size: int) -> jnp.ndarray:
    """goal → 2×(Dense256 + LN + swish) → Dense(goal_enc_size)."""
    lecun = hk.initializers.VarianceScaling(1 / 3, 'fan_in', 'uniform')
    zero_bias = hk.initializers.Constant(0.)

    x = goal
    for i in range(2):
        x = hk.Linear(256, w_init=lecun, b_init=zero_bias, name=f'dense_{i}')(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                         name=f'ln_{i}')(x)
        x = jax.nn.swish(x)
    return hk.Linear(goal_enc_size, w_init=lecun, b_init=zero_bias, name='out')(x)


# ---------------------------------------------------------------------------
# InvertiblePLU  (Haiku port of the Flax class in the reference code)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _host_plu_init(d: int, block_idx: int):
    """Orthogonal + LU on the host (no GPU cuSolver).

    After Isaac Gym PhysX owns the CUDA context, ``jax.random.orthogonal`` /
    ``jax.scipy.linalg.lu`` fail with ``gpusolverDnCreate``. Numpy QR + SciPy
    LU run only inside Haiku parameter init (same reason as OrthogonalHostQR).
    """
    rng = np.random.RandomState(int(block_idx) * 1337 + 42)
    z = rng.randn(d, d).astype(np.float32)
    q, r = np.linalg.qr(z)
    w0 = q * np.sign(np.diag(r))
    P0, L0, U0 = scipy_lu(w0.astype(np.float32, copy=False))
    s0 = np.diag(U0).astype(np.float32, copy=True)
    L_strict0 = np.tril(L0, k=-1).astype(np.float32, copy=False)
    U_strict0 = np.triu(U0, k=1).astype(np.float32, copy=False)
    P_inv0 = np.linalg.inv(P0).astype(np.float32, copy=False)
    return (P0.astype(np.float32, copy=False), P_inv0, L_strict0, U_strict0, s0)


def _triangular_inv(A: jnp.ndarray, n: int, *, lower: bool,
                    unit_diagonal: bool) -> jnp.ndarray:
    """Invert triangular ``A`` by substitution. Pure JAX, no cuSolver."""
    eye = jnp.eye(n, dtype=A.dtype)
    arange = jnp.arange(n)

    def _solve(b):
        def body(t, x):
            i = t if lower else (n - 1 - t)
            mask = (arange < i) if lower else (arange > i)
            aii = (jnp.ones((), dtype=A.dtype) if unit_diagonal else A[i, i])
            xi = (b[i] - jnp.dot(A[i], jnp.where(mask, x, 0))) / aii
            return x.at[i].set(xi)
        return jax.lax.fori_loop(0, n, body, jnp.zeros_like(b))

    return jax.vmap(_solve, in_axes=1, out_axes=1)(eye)


class InvertiblePLU(hk.Module):
    """Invertible 1×1 linear layer with PLU parameterization."""

    def __init__(self, features: int, block_idx: int = 0,
                 name: str = 'inv_plu'):
        super().__init__(name=name)
        self.features = features
        self.block_idx = block_idx

    def __call__(self, x: jnp.ndarray,
                 reverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        d = int(self.features)
        idx = int(self.block_idx)

        P = jax.lax.stop_gradient(
            hk.get_parameter(
                'P', (d, d), jnp.float32,
                init=lambda sh, dt: jnp.asarray(_host_plu_init(d, idx)[0], dt)))
        P_inv = jax.lax.stop_gradient(
            hk.get_parameter(
                'P_inv', (d, d), jnp.float32,
                init=lambda sh, dt: jnp.asarray(_host_plu_init(d, idx)[1], dt)))
        L_free = hk.get_parameter(
            'L', (d, d), jnp.float32,
            init=lambda sh, dt: jnp.asarray(_host_plu_init(d, idx)[2], dt))
        U_free = hk.get_parameter(
            'U', (d, d), jnp.float32,
            init=lambda sh, dt: jnp.asarray(_host_plu_init(d, idx)[3], dt))
        s = hk.get_parameter(
            's', (d,), jnp.float32,
            init=lambda sh, dt: jnp.asarray(_host_plu_init(d, idx)[4], dt))

        L = jnp.tril(L_free, k=-1) + jnp.eye(d)
        U = jnp.triu(U_free, k=1)
        W = P @ L @ (U + jnp.diag(s))
        logdet_scalar = jnp.sum(jnp.log(jnp.abs(s)))

        if not reverse:
            return jnp.dot(x, W), logdet_scalar
        U_inv = _triangular_inv(U + jnp.diag(s), d, lower=False,
                                unit_diagonal=False)
        L_inv = _triangular_inv(L, d, lower=True, unit_diagonal=True)
        W_inv = U_inv @ L_inv @ P_inv
        return jnp.dot(x, W_inv), -logdet_scalar


# ---------------------------------------------------------------------------
# 1.  Network container
# ---------------------------------------------------------------------------

class NFDensityNetworks(NamedTuple):
    """SA encoder + optional goal encoder + RealNVP flow.

    goal_encoder_net is None when goal_enc_size == 0 (raw goal fed to flow).
    flow_dim reflects the actual dimensionality the flow operates in:
        goal_enc_size > 0 → flow_dim = goal_enc_size
        goal_enc_size == 0 → flow_dim = goal_dim
    state_only=True means the conditioning encoder ignores action (p(g|s)).
    """
    sa_encoder_net: networks_lib.FeedForwardNetwork
    flow_net: networks_lib.FeedForwardNetwork
    goal_encoder_net: Optional[networks_lib.FeedForwardNetwork]
    flow_dim: int   # dimension of the space the flow operates in
    state_only: bool = False


# ---------------------------------------------------------------------------
# 2.  Network factory
# ---------------------------------------------------------------------------

def make_nf_density_networks(
    obs_dim: int,
    act_dim: int,
    goal_dim: int,
    hidden_layer_sizes: Sequence[int] = (),   # unused; kept for API compat
    rep_size: int = 64,
    num_blocks: int = 8,
    channels: int = 256,
    goal_enc_size: int = 0,                   # 0 = no goal encoder (legacy)
    sa_hidden: int = 1024,
    sa_num_layers: int = 4,
    state_only: bool = False,
    scale_tanh: bool = False,
    scale_tanh_c: float = 2.0,
    bb_pixel_obs: bool = False,
    bb_num_cubes: int = 0,
    bb_goal_color_indices: Optional[Sequence[int]] = None,
) -> NFDensityNetworks:
    """Build SA encoder + optional goal encoder + conditional RealNVP flow.

    When goal_enc_size > 0 the goal encoder maps the (normalized) goal into a
    goal_enc_size-dimensional latent and the flow operates in that latent space.
    When goal_enc_size == 0 the raw normalized goal is fed directly to the flow
    (original behaviour).

    state_only=False (default): learn log p_NF(g|s,a); encoder input is concat([s,a]).
    state_only=True: learn log p_NF(g|s); encoder input is s only (action ignored
    at apply time so call-site APIs stay the same).

    scale_tanh: if True, coupling scale is ``s = c * tanh(s_raw)`` (FrEIA soft
    clamp).  No extra parameters; old checkpoints still load.  Off by default.

    bb_pixel_obs: if True, rasterize compact BB xyz → CNN before the SA /
    goal MLPs.  Requires ``bb_num_cubes`` and ``bb_goal_color_indices``.
    When ``goal_enc_size==0``, a goal encoder is created with size
    ``rep_size`` so the flow runs on CNN embeddings (not raw xyz).
    """
    del hidden_layer_sizes
    assert goal_dim >= 1, f'NF density requires goal_dim >= 1, got {goal_dim}'
    assert int(sa_hidden) >= 1, f'sa_hidden must be >= 1, got {sa_hidden}'
    assert int(sa_num_layers) >= 1, (
        f'sa_num_layers must be >= 1, got {sa_num_layers}')

    _bb_pixel = bool(bb_pixel_obs)
    _bb_num_cubes = int(bb_num_cubes)
    _bb_goal_colors = tuple(int(i) for i in (bb_goal_color_indices or ()))
    if _bb_pixel:
      if _bb_num_cubes < 1 or not _bb_goal_colors:
        raise ValueError(
            'bb_pixel_obs requires bb_num_cubes>=1 and bb_goal_color_indices')
      if int(goal_enc_size) <= 0:
        goal_enc_size = max(int(rep_size), 2)

    # Effective dimensionality the flow operates in.
    flow_dim = goal_enc_size if goal_enc_size > 0 else goal_dim
    assert flow_dim >= 2, (
        f'Flow dim must be >= 2; got flow_dim={flow_dim} '
        f'(goal_dim={goal_dim}, goal_enc_size={goal_enc_size})')

    split_cond = (flow_dim + 1) // 2
    split_trans = flow_dim // 2

    w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    zero_init = hk.initializers.Constant(0.)
    _sa_hidden = int(sa_hidden)
    _sa_num_layers = int(sa_num_layers)
    _state_only = bool(state_only)
    _scale_tanh = bool(scale_tanh)
    _scale_tanh_c = float(scale_tanh_c)

    def _sa_fn(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        if _bb_pixel:
            from envs.builderbench_raster import (
                BBPixelTorso, rasterize_bb_state)
            state = BBPixelTorso(name='bb_pixel_torso')(
                rasterize_bb_state(state, _bb_num_cubes))
        return _sa_encoder(state, action, rep_size,
                           sa_hidden=_sa_hidden, sa_num_layers=_sa_num_layers,
                           state_only=_state_only)

    def _goal_enc_fn(goal: jnp.ndarray) -> jnp.ndarray:
        if _bb_pixel:
            from envs.builderbench_raster import (
                BBPixelTorso, rasterize_bb_goal)
            goal = BBPixelTorso(name='bb_pixel_torso')(
                rasterize_bb_goal(
                    goal, _bb_goal_colors, num_cubes=_bb_num_cubes))
        return _goal_encoder(goal, goal_enc_size)

    def _affine_st(i: int, cond_in: jnp.ndarray):
        """Coupling scale/shift. Names must match in forward and inverse."""
        s_h = hk.Linear(channels, w_init=w_init, name=f's_{i}_l1')(cond_in)
        s_h = jax.nn.leaky_relu(s_h)
        s_h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                           name=f's_{i}_ln1')(s_h)
        s_h = hk.Linear(channels, w_init=w_init, name=f's_{i}_l2')(s_h)
        s_h = jax.nn.leaky_relu(s_h)
        s_h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                           name=f's_{i}_ln2')(s_h)
        s_raw = hk.Linear(split_trans, w_init=zero_init, b_init=zero_init,
                          name=f's_{i}_out')(s_h)
        s = (_scale_tanh_c * jnp.tanh(s_raw)) if _scale_tanh else s_raw

        t_h = hk.Linear(channels, w_init=w_init, name=f't_{i}_l1')(cond_in)
        t_h = jax.nn.leaky_relu(t_h)
        t_h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                           name=f't_{i}_ln1')(t_h)
        t_h = hk.Linear(channels, w_init=w_init, name=f't_{i}_l2')(t_h)
        t_h = jax.nn.leaky_relu(t_h)
        t_h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                           name=f't_{i}_ln2')(t_h)
        t = hk.Linear(split_trans, w_init=zero_init, b_init=zero_init,
                      name=f't_{i}_out')(t_h)
        return s, t, s_raw

    def _flow_log_prob(goal_enc: jnp.ndarray, y: jnp.ndarray):
        """goal_enc is already encoded (or raw if no goal encoder).

        Returns ``(log_p, s_raw_mean, s_mean, z, log_det)``.  ``flow_net.apply``
        unwraps to ``log_p`` unless ``return_stats`` / ``return_forward``.
        ``z`` is the RealNVP latent ``f(g|s,a)``; ``log_det`` is
        ``log |det ∂z/∂g|``.
        """
        x = goal_enc
        log_dets = jnp.zeros(x.shape[0], dtype=x.dtype)
        s_raw_sum = jnp.zeros((), dtype=x.dtype)
        s_sum = jnp.zeros((), dtype=x.dtype)

        for i in range(num_blocks):
            plu = InvertiblePLU(features=flow_dim, block_idx=i, name=f'plu_{i}')
            x, plu_logdet = plu(x)
            log_dets = log_dets + plu_logdet

            x_cond = x[:, :split_cond]
            x_trans = x[:, split_cond:]
            cond_in = jnp.concatenate([x_cond, y], axis=-1)
            s, t, s_raw = _affine_st(i, cond_in)
            s_raw_sum = s_raw_sum + jnp.mean(s_raw)
            s_sum = s_sum + jnp.mean(s)

            x_trans_new = (x_trans - t) * jnp.exp(-s)
            log_dets = log_dets - jnp.sum(s, axis=-1)
            x = jnp.concatenate([x_cond, x_trans_new], axis=-1)

        log_norm = -0.5 * flow_dim * jnp.log(2.0 * jnp.pi)
        log_prior = -0.5 * jnp.sum(x ** 2, axis=-1) + log_norm
        denom = jnp.asarray(max(int(num_blocks), 1), dtype=x.dtype)
        return (log_prior + log_dets, s_raw_sum / denom, s_sum / denom,
                x, log_dets)

    def _flow_inverse(z: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Latent ``z`` → flow-space ``x`` (normalized goal). Same params as forward."""
        x = z
        for i in reversed(range(num_blocks)):
            x_cond = x[:, :split_cond]
            x_trans = x[:, split_cond:]
            cond_in = jnp.concatenate([x_cond, y], axis=-1)
            s, t, _ = _affine_st(i, cond_in)
            x_trans_new = x_trans * jnp.exp(s) + t
            x = jnp.concatenate([x_cond, x_trans_new], axis=-1)
            plu = InvertiblePLU(features=flow_dim, block_idx=i, name=f'plu_{i}')
            x, _ = plu(x, reverse=True)
        return x

    sa_transformed   = hk.without_apply_rng(hk.transform(_sa_fn))
    flow_transformed = hk.without_apply_rng(hk.transform(_flow_log_prob))
    flow_inv_transformed = hk.without_apply_rng(hk.transform(_flow_inverse))

    dummy_state  = np.zeros((1, obs_dim),    dtype=np.float32)
    dummy_action = np.zeros((1, act_dim),    dtype=np.float32)
    dummy_goal   = np.zeros((1, goal_dim),   dtype=np.float32)
    dummy_goal_enc = np.zeros((1, flow_dim), dtype=np.float32)
    dummy_y      = np.zeros((1, rep_size),   dtype=np.float32)

    sa_encoder_net = networks_lib.FeedForwardNetwork(
        init=lambda key: sa_transformed.init(key, dummy_state, dummy_action),
        apply=sa_transformed.apply,
    )
    def _flow_apply(params, goal_enc, y, return_stats=False,
                    return_forward=False, reverse=False):
        if reverse:
            return flow_inv_transformed.apply(params, goal_enc, y)
        log_p, s_raw_mean, s_mean, z, log_dets = flow_transformed.apply(
            params, goal_enc, y)
        if return_forward:
            return z, log_dets, log_p
        if return_stats:
            return log_p, s_raw_mean, s_mean
        return log_p

    flow_net = networks_lib.FeedForwardNetwork(
        init=lambda key: flow_transformed.init(key, dummy_goal_enc, dummy_y),
        apply=_flow_apply,
    )

    goal_encoder_net = None
    if goal_enc_size > 0:
        goal_enc_transformed = hk.without_apply_rng(hk.transform(_goal_enc_fn))
        goal_encoder_net = networks_lib.FeedForwardNetwork(
            init=lambda key: goal_enc_transformed.init(key, dummy_goal),
            apply=goal_enc_transformed.apply,
        )

    return NFDensityNetworks(
        sa_encoder_net=sa_encoder_net,
        flow_net=flow_net,
        goal_encoder_net=goal_encoder_net,
        flow_dim=flow_dim,
        state_only=_state_only,
    )


def make_nf_optimizers(
    encoder_lr: float = 3e-4,
    critic_lr: float = 1e-4,
    critic_weight_decay: float = 1e-6,
    grad_clip: float = 1.0,
    has_goal_encoder: bool = False,
) -> optax.GradientTransformation:
    """SA encoder + optional goal encoder + RealNVP flow optimizers.

    grad_clip > 0 prepends clip_by_global_norm to each optimizer chain.
    Set grad_clip=0 to disable clipping (legacy behaviour).
    has_goal_encoder=True adds a 'goal_encoder' transform group (same lr as SA encoder).
    """
    def _chain(optimizer):
        if grad_clip > 0:
            return optax.chain(optax.clip_by_global_norm(grad_clip), optimizer)
        return optimizer

    transforms = {
        'sa_encoder':   _chain(optax.adam(encoder_lr)),
        'nf_flow':      _chain(optax.adamw(critic_lr, weight_decay=critic_weight_decay)),
    }
    if has_goal_encoder:
        transforms['goal_encoder'] = _chain(optax.adam(encoder_lr))

    def _label_fn(params):
        labeled = {
            'sa_encoder': jax.tree_util.tree_map(lambda _: 'sa_encoder', params['sa_encoder']),
            'nf_flow':    jax.tree_util.tree_map(lambda _: 'nf_flow',    params['nf_flow']),
        }
        if 'goal_encoder' in params:
            labeled['goal_encoder'] = jax.tree_util.tree_map(
                lambda _: 'goal_encoder', params['goal_encoder'])
        return labeled

    return optax.multi_transform(transforms, _label_fn)


def init_nf_params(nf_networks: NFDensityNetworks, key) -> dict:
    """Init combined param dict with sa_encoder, nf_flow, and optionally goal_encoder."""
    k_sa, k_flow, k_ge = jax.random.split(key, 3)
    params = {
        'sa_encoder': nf_networks.sa_encoder_net.init(k_sa),
        'nf_flow':    nf_networks.flow_net.init(k_flow),
    }
    if nf_networks.goal_encoder_net is not None:
        params['goal_encoder'] = nf_networks.goal_encoder_net.init(k_ge)
    return params


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def _encode_goal(nf_networks: NFDensityNetworks, params: dict,
                 goal: jnp.ndarray) -> jnp.ndarray:
    """Encode goal if goal_encoder_net is present, else return raw goal."""
    if nf_networks.goal_encoder_net is not None:
        return nf_networks.goal_encoder_net.apply(params['goal_encoder'], goal)
    return goal


def nf_log_prob(nf_networks: NFDensityNetworks, params, state, action, goal):
    """log p_NF(goal | state, action) or log p_NF(goal | state) if state_only.

    goal must already be normalized.  ``action`` is ignored when
    ``nf_networks.state_only`` is True (kept in the signature for call-site compat).
    """
    y = nf_networks.sa_encoder_net.apply(params['sa_encoder'], state, action)
    g = _encode_goal(nf_networks, params, goal)
    return nf_networks.flow_net.apply(params['nf_flow'], g, y)


def nf_forward(nf_networks: NFDensityNetworks, params, state, action, goal):
    """RealNVP forward: ``z, log|det ∂z/∂g|, log p_NF``.

    ``z = f(g|s,a)`` (or ``f(g|s)`` if state_only).  ``goal`` must already
    be normalized.  ``log p = log N(z;0,I) + log|det|``.
    """
    y = nf_networks.sa_encoder_net.apply(params['sa_encoder'], state, action)
    g = _encode_goal(nf_networks, params, goal)
    return nf_networks.flow_net.apply(
        params['nf_flow'], g, y, return_forward=True)


def nf_inverse(nf_networks: NFDensityNetworks, params, state, action, z):
    """RealNVP inverse: latent ``z`` → normalized flow-space ``x``.

    Requires ``goal_enc_size=0`` (the goal encoder is not invertible).
    ``state`` / ``action`` are the conditioning inputs (action ignored if
    ``state_only``).  Un-normalize ``x`` with the train-time mean/std.
    """
    if nf_networks.goal_encoder_net is not None:
        raise ValueError(
            'nf_inverse requires goal_enc_size=0 '
            '(goal encoder is not invertible)')
    y = nf_networks.sa_encoder_net.apply(params['sa_encoder'], state, action)
    return nf_networks.flow_net.apply(
        params['nf_flow'], z, y, reverse=True)


def nf_sample(nf_networks: NFDensityNetworks, params, state, action, key):
    """Draw normalized flow-space samples: ``x = f^{-1}(z|cond), z~N(0,I)``."""
    batch = int(state.shape[0])
    z = jax.random.normal(
        key, (batch, int(nf_networks.flow_dim)), dtype=state.dtype)
    return nf_inverse(nf_networks, params, state, action, z)


def _l2_ball_perturb(x, key, prob, eps):
  """Independent per-row L2-ball jitter: with prob p, add δ with ‖δ‖₂ ≤ eps."""
  key, k_dir, k_rad, k_mask = jax.random.split(key, 4)
  direction = jax.random.normal(k_dir, x.shape, dtype=x.dtype)
  direction = direction / jnp.maximum(
      jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
  radius = jax.random.uniform(
      k_rad, (x.shape[0], 1), dtype=x.dtype) * eps
  noise = direction * radius
  do_pert = jax.random.bernoulli(
      k_mask, prob, shape=(x.shape[0],)).astype(x.dtype)
  noise = noise * do_pert[:, None]
  frac = jnp.mean(do_pert)
  mean_norm = jnp.mean(jnp.linalg.norm(noise, axis=-1))
  return x + noise, frac, mean_norm


def _l2_ball_perturb_slice(x, key, prob, eps, lo, hi):
  """L2-ball jitter on ``x[:, lo:hi]`` when that is a proper prefix; else all of x.

  BuilderBench: ``lo=0, hi=n_cubes*3`` leaves the trailing select dim unchanged.
  Other envs: ``lo=0, hi=-1`` / full last-axis (all of s).
  """
  d = int(x.shape[-1])
  lo_i = int(lo)
  hi_i = int(hi) if int(hi) >= 0 else d
  if 0 <= lo_i < hi_i < d:
    sl, frac, mean_norm = _l2_ball_perturb(x[:, lo_i:hi_i], key, prob, eps)
    x = x.at[:, lo_i:hi_i].set(sl)
    return x, frac, mean_norm
  return _l2_ball_perturb(x, key, prob, eps)


# ---------------------------------------------------------------------------
# 3.  Loss
# ---------------------------------------------------------------------------

def make_nf_density_update_fn(
    nf_networks: NFDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    noise_std: float = 0.0,
    mask_prob: float = 0.0,
    s_pert_prob: float = 0.0,
    s_pert_eps: float = 1e-2,
    s_pert_lo: int = 0,
    s_pert_hi: int = -1,
    grad_reg_c: float = 100.0,
    grad_reg_coef: float = 0.0,
    lam_lr: float = 0.0,
    task_goal_frac: float = 0.0,
    task_goal: Optional[np.ndarray] = None,
    time_reg_eta: float = 0.0,
    policy_network_apply=None,
    sample_fn=None,
    policy_goal: Optional[np.ndarray] = None,
):
    """Jitted update with separate grads for SA encoder, goal encoder, and NF flow.

    noise_std > 0 adds Gaussian noise to goals during training (regularisation).
    Noise is applied after normalization but before goal encoding.

    mask_prob > 0: with that probability, zero the online (s,a) so the flow
    also learns the marginal p(g|MASK). Same mechanism as TD-NF; default 0
    (off) for standard NLL. Reward still uses unmasked (s,a).

    s_pert_prob / s_pert_eps: with probability p, add L2-ball noise
    ‖δ‖₂ ≤ eps to raw state s before the SA encoder (same as CRL s perturb).
    s_pert_lo / s_pert_hi: if ``0 <= lo < hi < obs_dim``, jitter only that
    slice (BuilderBench xyz, leaving select alone).  ``hi < 0`` = all of s.

    grad_reg_coef > 0 enables the Lagrangian gradient regularizer.
    lam_lr > 0 switches to dual (primal-dual) optimization:
      θ step (minimize): NLL + λ · E[‖∇_s log p‖]
      λ step (maximize): λ ← clip(λ + lam_lr · (gnorm_mean − c), λ_min, 1)
    lam_lr = 0 keeps the old behaviour (λ injected from outside each iter).
    ∇_s log p diagnostics are always computed and logged; coef=0 keeps
    them out of the loss.

    task_goal_frac in (0, 1]: for that fraction of the batch, the ∇_s log p
    regularizer / diagnostics use the env task goal instead of replay g.
    The NLL is never mixed.  task_goal is raw (unnormalized) goal-space.

    goal_mean / goal_std are per-dim running stats passed at call time to
    normalise goals to approximately zero mean / unit variance before the flow.
    Both are 1-D arrays of length goal_dim.  Pass zeros/ones to disable.

    time_reg_eta > 0 adds η E[(log p_θ(g|s,a) − log p_old(g|s',a'))²]
    with a' ∼ π(·|s', g_task) (stopgrad π and a').  ``g`` is the same
    NLL goal; ``s'`` is ``batch['next_obs']``.  ``params_old`` is
    stopgrad.  eta=0 skips the second forward.  Requires
    policy_network_apply, sample_fn, and policy_goal.
    """
    _s_pert_prob = float(s_pert_prob)
    _s_pert_eps = float(s_pert_eps)
    _s_pert_lo = int(s_pert_lo)
    _s_pert_hi = int(s_pert_hi)
    _do_s_pert = _s_pert_prob > 0.0 and _s_pert_eps > 0.0
    _mask_prob = float(mask_prob)
    _do_mask = _mask_prob > 0.0
    _grad_reg_c = float(grad_reg_c)
    _grad_reg_coef = float(grad_reg_coef)
    _lam_lr = float(lam_lr)
    _lam_min = 1e-8
    _lam_max = 1.0
    _log_grad_reg = obs_dim > 0
    _do_grad_reg = _log_grad_reg and _grad_reg_coef > 0.0
    _dual = _do_grad_reg and _lam_lr > 0.0
    _task_goal_frac = float(task_goal_frac)
    if not (0.0 <= _task_goal_frac <= 1.0):
        raise ValueError(
            'task_goal_frac must be in [0, 1], '
            f'got {_task_goal_frac}')
    _do_task_g_mix = _log_grad_reg and _task_goal_frac > 0.0
    if _do_task_g_mix and task_goal is None:
        raise ValueError(
            'task_goal_frac>0 requires task_goal (raw env task goal)')
    _task_goal = (
        None if task_goal is None
        else jnp.asarray(task_goal, dtype=jnp.float32).reshape(-1))
    _init_lam = jnp.array(
        max(_grad_reg_coef, 5e-4) if _do_grad_reg else 0.0,
        dtype=jnp.float32)
    _time_reg_eta = float(time_reg_eta)
    if _time_reg_eta < 0.0:
        raise ValueError(
            f'time_reg_eta must be >= 0, got {_time_reg_eta}')
    _do_time_reg = _time_reg_eta > 0.0
    _policy_apply = policy_network_apply
    _policy_sample = sample_fn
    _policy_goal_j = None
    if _do_time_reg:
        if (_policy_apply is None or _policy_sample is None
                or policy_goal is None):
            raise ValueError(
                'time_reg_eta>0 requires policy_network_apply, sample_fn, '
                'and policy_goal so a\' ∼ π(·|s\', g_task)')
        _policy_goal_j = jnp.asarray(
            policy_goal, dtype=jnp.float32).reshape(-1)

    def _loss(params, batch, key, goal_mean, goal_std, lam_val,
              params_old, policy_params):
        obs    = batch['obs']
        action = batch['action']
        state  = obs[:, :obs_dim]
        goal   = obs[:, obs_dim:]
        s_pert_frac = jnp.array(0.0, dtype=state.dtype)
        s_pert_norm = jnp.array(0.0, dtype=state.dtype)
        mask_frac = jnp.array(0.0, dtype=state.dtype)
        if _do_time_reg:
            key, k_s, k_g, k_mask, k_act = jax.random.split(key, 5)
        else:
            key, k_s, k_g, k_mask = jax.random.split(key, 4)

        if _do_s_pert:
            state, s_pert_frac, s_pert_norm = _l2_ball_perturb_slice(
                state, k_s, _s_pert_prob, _s_pert_eps,
                _s_pert_lo, _s_pert_hi)

        if _do_mask:
            mask = jax.random.bernoulli(
                k_mask, _mask_prob, shape=(state.shape[0], 1)).astype(
                    state.dtype)
            state = state * (1.0 - mask)
            action = action * (1.0 - mask)
            mask_frac = jnp.mean(mask)

        # Normalise goals to ~N(0,1) using caller-supplied running stats.
        goal = (goal - goal_mean) / (goal_std + 1e-8)

        goal_noise = None
        if noise_std > 0.0:
            goal_noise = noise_std * jax.random.normal(k_g, goal.shape)
            goal = goal + goal_noise

        y = nf_networks.sa_encoder_net.apply(params['sa_encoder'], state, action)
        g = _encode_goal(nf_networks, params, goal)
        log_p, s_raw_mean, s_mean = nf_networks.flow_net.apply(
            params['nf_flow'], g, y, return_stats=True)
        nll = -jnp.mean(log_p)

        zero = jnp.array(0.0, dtype=nll.dtype)
        gnorm_mean = zero
        gnorm_max = zero
        gnorm_frac_above = zero
        grad_reg_raw = zero
        grad_reg_lam = zero
        grad_reg = zero
        task_g_frac = zero
        if _log_grad_reg:
            def _one_logp(s, a, g_one):
                # Flow/encoder expect a batch dim; vmap supplies unbatched rows.
                y_one = nf_networks.sa_encoder_net.apply(
                    params['sa_encoder'], s[None], a[None])
                g_enc = _encode_goal(nf_networks, params, g_one[None])
                lp = nf_networks.flow_net.apply(
                    params['nf_flow'], g_enc, y_one)
                return jnp.reshape(lp, ())

            def _gnorm(s, a, g_one):
                gs = jax.grad(_one_logp, argnums=0)(s, a, g_one)
                # jnp.linalg.norm(0) has NaN VJP (0/0); that poisons ∇_θ
                # when the hinge is in the loss. CRL's φ·ψ grads are rarely 0.
                return optax.safe_norm(gs, 1e-8)

            # NLL keeps replay g. Mix task g into the score regularizer only.
            goal_reg = goal
            if _do_task_g_mix:
                g_task_n = (_task_goal - goal_mean) / (goal_std + 1e-8)
                g_task_b = jnp.broadcast_to(g_task_n[None, :], goal.shape)
                if goal_noise is not None:
                    g_task_b = g_task_b + goal_noise
                n_b = int(goal.shape[0])
                k_task = int(round(_task_goal_frac * n_b))
                k_task = max(0, min(n_b, k_task))
                if k_task == n_b:
                    goal_reg = g_task_b
                    task_g_frac = jnp.array(1.0, dtype=nll.dtype)
                elif k_task > 0:
                    key, k_mix = jax.random.split(key)
                    perm = jax.random.permutation(k_mix, n_b)
                    is_task = jnp.zeros((n_b,), dtype=goal.dtype).at[
                        perm[:k_task]].set(1.0)
                    goal_reg = (
                        goal * (1.0 - is_task[:, None])
                        + g_task_b * is_task[:, None])
                    task_g_frac = jnp.mean(is_task)

            gnorms = jax.vmap(_gnorm)(state, action, goal_reg)
            gnorm_mean = jnp.mean(gnorms)
            gnorm_max = jnp.max(gnorms)
            gnorm_frac_above = jnp.mean(
                (gnorms > _grad_reg_c).astype(nll.dtype))
            grad_reg_raw = jnp.mean(jnp.maximum(gnorms - _grad_reg_c, 0.0))
            if _do_grad_reg:
                grad_reg_lam = lam_val.astype(nll.dtype)
                # Lagrangian penalty: λ · E[‖∇_s log p‖] (dual and fixed-λ).
                grad_reg = grad_reg_lam * gnorm_mean

        time_reg_raw = zero
        time_reg = zero
        if _do_time_reg:
            next_s = batch['next_obs'][:, :obs_dim]
            pol = jax.lax.stop_gradient(policy_params)
            env_goal = jnp.broadcast_to(
                _policy_goal_j[None, :],
                (next_s.shape[0], _policy_goal_j.shape[0]))
            next_policy_obs = jnp.concatenate([next_s, env_goal], axis=1)
            next_dist = _policy_apply(pol, next_policy_obs)
            next_a = jax.lax.stop_gradient(_policy_sample(next_dist, k_act))
            log_p_old = nf_log_prob(
                nf_networks, params_old, next_s, next_a, goal)
            log_p_old = jax.lax.stop_gradient(log_p_old)
            time_reg_raw = jnp.mean((log_p - log_p_old) ** 2)
            time_reg = jnp.asarray(_time_reg_eta, dtype=nll.dtype) * time_reg_raw

        loss = nll + grad_reg + time_reg
        metrics = {
            'density_loss': nll,
            'nf_total_loss': loss,
            'log_p_mean':   jnp.mean(log_p),
            'log_p_min':    jnp.min(log_p),
            'log_p_max':    jnp.max(log_p),
            'repr_norm':    jnp.mean(jnp.linalg.norm(y, axis=-1)),
            'nf_s_perturb_frac': s_pert_frac,
            'nf_s_perturb_norm': s_pert_norm,
            'nf_mask_frac': mask_frac,
            'nf_grad_reg': grad_reg,
            'nf_grad_reg_raw': grad_reg_raw,
            'nf_grad_reg_lam': grad_reg_lam,
            'nf_logp_grad_s_norm_mean': gnorm_mean,
            'nf_logp_grad_s_norm_max': gnorm_max,
            'nf_logp_grad_s_frac_above_c': gnorm_frac_above,
            's_raw_mean': s_raw_mean,
            's_mean': s_mean,
        }
        if _do_time_reg:
            metrics['nf_time_reg'] = time_reg
            metrics['nf_time_reg_raw'] = time_reg_raw
            metrics['nf_time_reg_nll_ratio'] = time_reg / (
                jnp.abs(nll) + jnp.asarray(1e-8, dtype=nll.dtype))
        if _do_task_g_mix:
            metrics['nf_grad_reg_task_g_frac'] = task_g_frac
        return loss, metrics

    grad_fn = jax.value_and_grad(_loss, has_aux=True)

    def update(params, opt_state, batch, key, goal_mean, goal_std,
               lam_val=None, params_old=None, policy_params=None):
        _lam = _init_lam if lam_val is None else lam_val
        _p_old = params if params_old is None else params_old
        _pol = params if policy_params is None else policy_params
        (_, metrics), grads = grad_fn(
            params, batch, key, goal_mean, goal_std, _lam, _p_old, _pol)
        metrics = dict(metrics)
        metrics['encoder_grad_norm'] = _tree_l2_norm(grads['sa_encoder'])
        metrics['flow_grad_norm']    = _tree_l2_norm(grads['nf_flow'])
        if 'goal_encoder' in grads:
            metrics['goal_enc_grad_norm'] = _tree_l2_norm(grads['goal_encoder'])

        grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads))))
        loss_finite = jnp.isfinite(metrics['density_loss'])
        do_update   = jnp.logical_and(grads_finite, loss_finite)

        def _apply(_):
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_opt_state

        def _skip(_):
            return params, opt_state

        new_params, new_opt_state = jax.lax.cond(
            do_update, _apply, _skip, operand=None)
        metrics = dict(metrics)
        metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)

        # Dual λ step (primal-dual / Lagrangian):
        #   λ ← clip(λ + lr_λ · (gnorm_mean − c),  λ_min,  λ_max)
        # stop_gradient so λ does not create a second autodiff path through gnorm.
        if _dual:
            gnorm_sg = jax.lax.stop_gradient(
                metrics['nf_logp_grad_s_norm_mean'])
            lam_new = jnp.clip(
                _lam + _lam_lr * (gnorm_sg - _grad_reg_c),
                _lam_min, _lam_max)
            metrics['nf_grad_reg_lam'] = lam_new
        else:
            lam_new = _lam

        return new_params, new_opt_state, lam_new, metrics

    return jax.jit(update)


def make_scan_nf_update_fn(
    nf_networks: NFDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    noise_std: float = 0.0,
    repr_tau: float = 0.0,
    mask_prob: float = 0.0,
    s_pert_prob: float = 0.0,
    s_pert_eps: float = 1e-2,
    s_pert_lo: int = 0,
    s_pert_hi: int = -1,
    grad_reg_c: float = 100.0,
    grad_reg_coef: float = 0.0,
    lam_lr: float = 0.0,
    task_goal_frac: float = 0.0,
    task_goal: Optional[np.ndarray] = None,
    time_reg_eta: float = 0.0,
    policy_network_apply=None,
    sample_fn=None,
    policy_goal: Optional[np.ndarray] = None,
):
  """Scan-based NF updater: N density steps in one JIT call.

  Caller pre-samples all N batches in NumPy, stacks them into
  ``(N, B, dim)`` arrays, and transfers to GPU once.  Matches the CRL
  ``make_scan_crl_update_fn`` pattern.  Reward-param EMA
  (``0 < repr_tau < 1``) is applied inside the scan.

  When lam_lr > 0 the dual λ step runs inside each scan step so λ is
  updated every NF gradient step (not just between PPO iters).

  time_reg_eta > 0: each step uses the same ``params_old`` (caller
  snapshot / slow EMA) and the same stopgrad policy for
  a' ∼ π(·|s', g_task).  The time-reg EMA itself is mixed by the
  caller once per PPO iter, not inside this scan.

  Returns:
    ``multi_update(params, opt_state, params_ema, batches, key,
                   goal_mean, goal_std, lam_val=None, params_old=None,
                   policy_params=None)``
    → ``(new_params, new_opt_state, new_params_ema, new_key,
         new_lam, mean_metrics)``
  """
  raw_update = make_nf_density_update_fn(
      nf_networks, optimizer, obs_dim=obs_dim, noise_std=noise_std,
      mask_prob=mask_prob,
      s_pert_prob=s_pert_prob, s_pert_eps=s_pert_eps,
      s_pert_lo=s_pert_lo, s_pert_hi=s_pert_hi,
      grad_reg_c=grad_reg_c, grad_reg_coef=grad_reg_coef,
      lam_lr=lam_lr,
      task_goal_frac=task_goal_frac, task_goal=task_goal,
      time_reg_eta=time_reg_eta,
      policy_network_apply=policy_network_apply,
      sample_fn=sample_fn,
      policy_goal=policy_goal)
  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)
  _init_lam_py = max(grad_reg_coef, 5e-4) if grad_reg_coef > 0.0 else 0.0
  _do_time_reg = float(time_reg_eta) > 0.0

  @jax.jit
  def multi_update(
      params, opt_state, params_ema, batches, key, goal_mean, goal_std,
      lam_val=None, params_old=None, policy_params=None):
    lam = (jnp.array(_init_lam_py, dtype=jnp.float32)
           if lam_val is None else lam_val)
    p_old = params if params_old is None else params_old
    pol = params if policy_params is None else policy_params
    if _do_time_reg:
      p_old = jax.lax.stop_gradient(p_old)
      pol = jax.lax.stop_gradient(pol)

    def scan_step(carry, batch):
      p, opt, ema, k, lam = carry
      k, k_u = jax.random.split(k)
      if _do_time_reg:
        p, opt, lam, m = raw_update(
            p, opt, batch, k_u, goal_mean, goal_std, lam, p_old, pol)
      else:
        p, opt, lam, m = raw_update(
            p, opt, batch, k_u, goal_mean, goal_std, lam)
      if use_ema:
        ema = jax.tree_util.tree_map(
            lambda t, o: _tau * t + (1.0 - _tau) * o, ema, p)
      else:
        ema = p
      return (p, opt, ema, k, lam), m

    (params, opt_state, params_ema, key, lam), metrics = jax.lax.scan(
        scan_step, (params, opt_state, params_ema, key, lam), batches)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return params, opt_state, params_ema, key, lam, metrics

  return multi_update


# ---------------------------------------------------------------------------
# 3b.  TD-NF loss  (one-step + importance-weighted bootstrap)
# ---------------------------------------------------------------------------

def make_nf_td_density_update_fn(
    nf_networks: NFDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    discount: float,
    policy_network_apply,
    sample_fn,
    policy_goal: np.ndarray,
    target_tau: float = 0.995,
    noise_std: float = 0.0,
    mask_prob: float = 0.2,
    start_index: int = 0,
    end_index: int = -1,
    goal_state_indices=None,
    ratio_clip: float = 20.0,
    s_pert_prob: float = 0.0,
    s_pert_eps: float = 1e-2,
    s_pert_lo: int = 0,
    s_pert_hi: int = -1,
):
  """TD normalizing-flow density update.

  Maximises::

      (1-γ) log p_θ(g'|s,a)
        + γ · stopgrad( p_{θ⁻}(g_j|s',a') / p_{θ⁻}(g_j) )
              · log p_θ(g_j|s,a)

  where ``g' = obs_to_goal(s')``, ``g_j`` is a rolled other-row next-state
  goal (empirical next-state marginal), ``a' ∼ π(·|s', g_task)`` is sampled
  fresh from the current PPO policy (no grad into π), and the target
  marginal ``p_{θ⁻}(g_j)`` is the same NF with ``(s,a)`` replaced by zeros.

  With probability ``mask_prob``, the *online* conditioning ``(s,a)`` is
  also zero-masked so the flow learns the marginal pathway used in the
  denominator.  Target params follow EMA
  ``θ⁻ ← target_tau·θ⁻ + (1-target_tau)·θ``.
  """
  gamma = float(discount)
  _target_tau = float(target_tau)
  _mask_prob = float(mask_prob)
  _noise_std = float(noise_std)
  _ratio_clip = float(ratio_clip)
  _s_pert_prob = float(s_pert_prob)
  _s_pert_eps = float(s_pert_eps)
  _s_pert_lo = int(s_pert_lo)
  _s_pert_hi = int(s_pert_hi)
  _do_s_pert = _s_pert_prob > 0.0 and _s_pert_eps > 0.0
  si = int(start_index)
  ei = int(end_index)
  _gidx = None if goal_state_indices is None else jnp.asarray(
      goal_state_indices, dtype=jnp.int32)
  _policy_goal = jnp.asarray(policy_goal, dtype=jnp.float32).reshape(-1)

  def _state_as_goal(state: jnp.ndarray) -> jnp.ndarray:
    if _gidx is not None:
      return state[:, _gidx]
    if ei == -1:
      return state[:, si:]
    return state[:, si:ei]

  def _norm_goal(goal, goal_mean, goal_std):
    return (goal - goal_mean) / (goal_std + 1e-8)

  def _loss(params, target_params, policy_params, batch, key,
            goal_mean, goal_std):
    obs = batch['obs']
    action = batch['action']
    next_obs = batch['next_obs']
    batch_size = obs.shape[0]

    s = obs[:, :obs_dim]
    next_s = next_obs[:, :obs_dim]
    s_pert_frac = jnp.array(0.0, dtype=s.dtype)
    s_pert_norm = jnp.array(0.0, dtype=s.dtype)

    g_prime = _norm_goal(_state_as_goal(next_s), goal_mean, goal_std)
    key, k_noise, k_mask, k_act, k_s = jax.random.split(key, 5)
    if _do_s_pert:
      s, s_pert_frac, s_pert_norm = _l2_ball_perturb_slice(
          s, k_s, _s_pert_prob, _s_pert_eps, _s_pert_lo, _s_pert_hi)
    if _noise_std > 0.0:
      g_prime = g_prime + _noise_std * jax.random.normal(
          k_noise, g_prime.shape)
    # sj = next-state goals of other batch rows (marginal sample).
    g_j = jnp.roll(g_prime, shift=1, axis=0)

    # a' ~ π(· | s', g_task) — fresh policy sample (mirrors TD-InfoNCE).
    env_goal = jnp.broadcast_to(
        _policy_goal[None, :], (batch_size, _policy_goal.shape[0]))
    next_policy_obs = jnp.concatenate([next_s, env_goal], axis=1)
    next_dist = policy_network_apply(policy_params, next_policy_obs)
    next_action = sample_fn(next_dist, k_act)

    # Online conditioning: with mask_prob, replace (s,a) by zeros so the
    # flow also learns p(g | MASK) ≈ marginal.
    mask = jax.random.bernoulli(
        k_mask, _mask_prob, shape=(batch_size, 1)).astype(s.dtype)
    s_online = s * (1.0 - mask)
    a_online = action * (1.0 - mask)

    log_p_next = nf_log_prob(
        nf_networks, params, s_online, a_online, g_prime)
    log_p_sj = nf_log_prob(
        nf_networks, params, s_online, a_online, g_j)

    # Target weights (stop-grad): p(g_j|s',a') / p(g_j|MASK).
    log_p_tgt_cond = nf_log_prob(
        nf_networks, target_params, next_s, next_action, g_j)
    zeros_s = jnp.zeros_like(next_s)
    zeros_a = jnp.zeros_like(next_action)
    log_p_tgt_marg = nf_log_prob(
        nf_networks, target_params, zeros_s, zeros_a, g_j)
    log_ratio = jnp.clip(
        log_p_tgt_cond - log_p_tgt_marg, -_ratio_clip, _ratio_clip)
    w = jax.lax.stop_gradient(jnp.exp(log_ratio))

    term1 = (1.0 - gamma) * log_p_next
    term2 = gamma * w * log_p_sj
    objective = term1 + term2
    loss = -jnp.mean(objective)

    metrics = {
        'density_loss': loss,
        'td_term1': jnp.mean(term1),
        'td_term2': jnp.mean(term2),
        'log_p_next_mean': jnp.mean(log_p_next),
        'log_p_sj_mean': jnp.mean(log_p_sj),
        'log_p_tgt_cond_mean': jnp.mean(log_p_tgt_cond),
        'log_p_tgt_marg_mean': jnp.mean(log_p_tgt_marg),
        'td_w_mean': jnp.mean(w),
        'td_w_std': jnp.std(w),
        'td_mask_frac': jnp.mean(mask),
        'a_prime_mean': jnp.mean(next_action),
        'a_prime_std': jnp.std(next_action),
        'log_p_mean': jnp.mean(log_p_next),
        'log_p_min': jnp.min(log_p_next),
        'log_p_max': jnp.max(log_p_next),
        'nf_s_perturb_frac': s_pert_frac,
        'nf_s_perturb_norm': s_pert_norm,
    }
    return loss, metrics

  grad_fn = jax.value_and_grad(_loss, has_aux=True)

  def update(params, opt_state, target_params, policy_params, batch, key,
             goal_mean, goal_std):
    (_, metrics), grads = grad_fn(
        params, target_params, policy_params, batch, key,
        goal_mean, goal_std)
    metrics = dict(metrics)
    metrics['encoder_grad_norm'] = _tree_l2_norm(grads['sa_encoder'])
    metrics['flow_grad_norm'] = _tree_l2_norm(grads['nf_flow'])
    if 'goal_encoder' in grads:
      metrics['goal_enc_grad_norm'] = _tree_l2_norm(grads['goal_encoder'])

    grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads))))
    loss_finite = jnp.isfinite(metrics['density_loss'])
    do_update = jnp.logical_and(grads_finite, loss_finite)

    def _apply(_):
      updates, new_opt_state = optimizer.update(grads, opt_state, params)
      return optax.apply_updates(params, updates), new_opt_state

    def _skip(_):
      return params, opt_state

    new_params, new_opt_state = jax.lax.cond(
        do_update, _apply, _skip, operand=None)
    new_target = jax.tree_util.tree_map(
        lambda t, o: _target_tau * t + (1.0 - _target_tau) * o,
        target_params, new_params)
    metrics = dict(metrics)
    metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
    return new_params, new_opt_state, new_target, metrics

  return jax.jit(update)


def make_scan_nf_td_update_fn(
    nf_networks: NFDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    discount: float,
    policy_network_apply,
    sample_fn,
    policy_goal: np.ndarray,
    target_tau: float = 0.995,
    noise_std: float = 0.0,
    mask_prob: float = 0.2,
    start_index: int = 0,
    end_index: int = -1,
    goal_state_indices=None,
    repr_tau: float = 0.0,
    ratio_clip: float = 20.0,
    s_pert_prob: float = 0.0,
    s_pert_eps: float = 1e-2,
    s_pert_lo: int = 0,
    s_pert_hi: int = -1,
):
  """Scan-based TD-NF updater: N density steps in one JIT call.

  Returns:
    ``multi_update(params, opt_state, target_params, params_ema, batches,
                   key, goal_mean, goal_std, policy_params)``
    → ``(new_params, new_opt, new_target, new_ema, new_key, mean_metrics)``
  """
  raw_update = make_nf_td_density_update_fn(
      nf_networks, optimizer,
      obs_dim=obs_dim,
      discount=discount,
      policy_network_apply=policy_network_apply,
      sample_fn=sample_fn,
      policy_goal=policy_goal,
      target_tau=target_tau,
      noise_std=noise_std,
      mask_prob=mask_prob,
      start_index=start_index,
      end_index=end_index,
      goal_state_indices=goal_state_indices,
      ratio_clip=ratio_clip,
      s_pert_prob=s_pert_prob,
      s_pert_eps=s_pert_eps,
      s_pert_lo=s_pert_lo,
      s_pert_hi=s_pert_hi,
  )
  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)

  @jax.jit
  def multi_update(
      params, opt_state, target_params, params_ema, batches, key,
      goal_mean, goal_std, policy_params):
    def scan_step(carry, batch):
      p, opt, tgt, ema, k = carry
      k, k_u = jax.random.split(k)
      p, opt, tgt, m = raw_update(
          p, opt, tgt, policy_params, batch, k_u, goal_mean, goal_std)
      if use_ema:
        ema = jax.tree_util.tree_map(
            lambda t, o: _tau * t + (1.0 - _tau) * o, ema, p)
      else:
        ema = p
      return (p, opt, tgt, ema, k), m

    (params, opt_state, target_params, params_ema, key), metrics = (
        jax.lax.scan(
            scan_step,
            (params, opt_state, target_params, params_ema, key),
            batches))
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return params, opt_state, target_params, params_ema, key, metrics

  return multi_update


# ---------------------------------------------------------------------------
# 4.  Reward
# ---------------------------------------------------------------------------

NF_REWARD_MODES = ('forward', 'reverse', 'chi_squared')
# float32 exp overflows near |x|~88; clip only reverse / chi-squared maps.
NF_REWARD_LOGP_CLIP = 80.0


def normalize_nf_reward_mode(reward_mode: str) -> str:
    mode = (reward_mode or 'forward').strip().lower()
    if mode not in NF_REWARD_MODES:
        raise ValueError(
            f'Unknown nf_reward_mode={reward_mode!r}; '
            f'expected one of {NF_REWARD_MODES}')
    return mode


def make_nf_reward_fn(nf_networks: NFDensityNetworks, obs_dim: int,
                      tanh_scale: float = 0.0,
                      reward_mode: str = 'forward'):
    """Jitted reward: (params, obs, action, goal_mean, goal_std) → r_NF.

    ``reward_mode`` (density NLL is always raw log p, never these maps):
      forward     — r = log p_NF(g|s,a)   (default; current behaviour)
      reverse     — r = exp(clip(log p, ±80))           = p(g|s,a)
      chi_squared — r = −exp(−clip(log p, ±80))         = −1/p(g|s,a)

    With state_only nets: r(s) uses log p_NF(g|s) (action is ignored).
    goal_mean / goal_std must match those used during training so that the
    flow sees the same normalized input distribution.

    If tanh_scale > 0, returns ``tanh_scale * tanh(r / tanh_scale)``.  Only
    valid with ``reward_mode='forward'``.
    """
    mode = normalize_nf_reward_mode(reward_mode)
    _tanh_scale = float(tanh_scale)
    if _tanh_scale > 0.0 and mode != 'forward':
        raise ValueError(
            f'nf_reward_tanh is only valid with nf_reward_mode=forward '
            f'(got {mode!r})')
    _clip = float(NF_REWARD_LOGP_CLIP)

    @jax.jit
    def reward_fn(nf_params, obs: jnp.ndarray, action: jnp.ndarray,
                  goal_mean: jnp.ndarray, goal_std: jnp.ndarray):
        state = obs[:, :obs_dim]
        goal  = obs[:, obs_dim:]
        goal  = (goal - goal_mean) / (goal_std + 1e-8)
        log_p = nf_log_prob(nf_networks, nf_params, state, action, goal)
        if mode == 'reverse':
            r = jnp.exp(jnp.clip(log_p, -_clip, _clip))
        elif mode == 'chi_squared':
            r = -jnp.exp(-jnp.clip(log_p, -_clip, _clip))
        else:
            r = log_p
        if _tanh_scale > 0.0:
            return _tanh_scale * jnp.tanh(r / _tanh_scale)
        return r

    return reward_fn
