"""Haiku modules that output tfd.Distributions."""

from typing import Any, List, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability
hk_init = hk.initializers
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

_MIN_SCALE = 1e-4
Initializer = hk.initializers.Initializer


class CategoricalHead(hk.Module):
  """Module that produces a categorical distribution with the given number of values."""

  def __init__(
      self,
      num_values: Union[int, List[int]],
      dtype: Optional[Any] = jnp.int32,
      w_init: Optional[Initializer] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._dtype = dtype
    self._logit_shape = num_values
    self._linear = hk.Linear(np.prod(num_values), w_init=w_init)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    logits = self._linear(inputs)
    if not isinstance(self._logit_shape, int):
      logits = hk.Reshape(self._logit_shape)(logits)
    return tfd.Categorical(logits=logits, dtype=self._dtype)


class GaussianMixture(hk.Module):
  """Module that outputs a Gaussian Mixture Distribution."""

  def __init__(self,
               num_dimensions: int,
               num_components: int,
               multivariate: bool,
               init_scale: Optional[float] = None,
               append_singleton_event_dim: bool = False,
               reinterpreted_batch_ndims: Optional[int] = None,
               name: str = 'GaussianMixture'):
    """Initialization.

    Args:
      num_dimensions: dimensionality of the output distribution
      num_components: number of mixture components.
      multivariate: whether the resulting distribution is multivariate or not.
      init_scale: the initial scale for the Gaussian mixture components.
      append_singleton_event_dim: (univariate only) Whether to add an extra
        singleton dimension to the event shape.
      reinterpreted_batch_ndims: (univariate only) Number of batch dimensions to
        reinterpret as event dimensions.
      name: name of the module passed to snt.Module parent class.
    """
    super().__init__(name=name)

    self._num_dimensions = num_dimensions
    self._num_components = num_components
    self._multivariate = multivariate
    self._append_singleton_event_dim = append_singleton_event_dim
    self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

    if init_scale is not None:
      self._scale_factor = init_scale / jax.nn.softplus(0.)
    else:
      self._scale_factor = 1.0  # Corresponds to init_scale = softplus(0).

  def __call__(self,
               inputs: jnp.ndarray,
               low_noise_policy: bool = False) -> tfd.Distribution:
    """Run the networks through inputs.

    Args:
      inputs: hidden activations of the policy network body.
      low_noise_policy: whether to set vanishingly small scales for each
        component. If this flag is set to True, the policy is effectively run
        without Gaussian noise.

    Returns:
      Mixture Gaussian distribution.
    """

    # Define the weight initializer.
    w_init = hk.initializers.VarianceScaling(scale=1e-5)

    # Create a layer that outputs the unnormalized log-weights.
    if self._multivariate:
      logits_size = self._num_components
    else:
      logits_size = self._num_dimensions * self._num_components
    logit_layer = hk.Linear(logits_size, w_init=w_init)

    # Create two layers that outputs a location and a scale, respectively, for
    # each dimension and each component.
    loc_layer = hk.Linear(
        self._num_dimensions * self._num_components, w_init=w_init)
    scale_layer = hk.Linear(
        self._num_dimensions * self._num_components, w_init=w_init)

    # Compute logits, locs, and scales if necessary.
    logits = logit_layer(inputs)
    locs = loc_layer(inputs)

    # When a low_noise_policy is requested, set the scales to its minimum value.
    if low_noise_policy:
      scales = jnp.full(locs.shape, _MIN_SCALE)
    else:
      scales = scale_layer(inputs)
      scales = self._scale_factor * jax.nn.softplus(scales) + _MIN_SCALE

    if self._multivariate:
      components_class = tfd.MultivariateNormalDiag
      shape = [-1, self._num_components, self._num_dimensions]  # [B, C, D]
      # In this case, no need to reshape logits as they are in the correct shape
      # already, namely [batch_size, num_components].
    else:
      components_class = tfd.Normal
      shape = [-1, self._num_dimensions, self._num_components]  # [B, D, C]
      if self._append_singleton_event_dim:
        shape.insert(2, 1)  # [B, D, 1, C]
      logits = logits.reshape(shape)

    # Reshape the mixture's location and scale parameters appropriately.
    locs = locs.reshape(shape)
    scales = scales.reshape(shape)

    # Create the mixture distribution.
    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=components_class(loc=locs, scale=scales))

    if not self._multivariate:
      distribution = tfd.Independent(
          distribution,
          reinterpreted_batch_ndims=self._reinterpreted_batch_ndims)

    return distribution


class TanhTransformedDistribution(tfd.TransformedDistribution):
  """Distribution followed by tanh."""

  def __init__(self, distribution, threshold=.999, validate_args=False):
    """Initialize the distribution.

    Args:
      distribution: The distribution to transform.
      threshold: Clipping value of the action when computing the logprob.
      validate_args: Passed to super class.
    """
    super().__init__(
        distribution=distribution,
        bijector=tfp.bijectors.Tanh(),
        validate_args=validate_args)
    # Computes the log of the average probability distribution outside the
    # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
    # log_prob_left and [atanh(threshold), inf] for log_prob_right.
    self._threshold = threshold
    inverse_threshold = self.bijector.inverse(threshold)
    # average(pdf) = p/epsilon
    # So log(average(pdf)) = log(p) - log(epsilon)
    log_epsilon = jnp.log(1. - threshold)
    # Those 2 values are differentiable w.r.t. model parameters, such that the
    # gradient is defined everywhere.
    self._log_prob_left = self.distribution.log_cdf(
        -inverse_threshold) - log_epsilon
    self._log_prob_right = self.distribution.log_survival_function(
        inverse_threshold) - log_epsilon

  def log_prob(self, event):
    # Without this clip there would be NaNs in the inner tf.where and that
    # causes issues for some reasons.
    event = jnp.clip(event, -self._threshold, self._threshold)
    # The inverse image of {threshold} is the interval [atanh(threshold), inf]
    # which has a probability of "log_prob_right" under the given distribution.
    return jnp.where(
        event <= -self._threshold, self._log_prob_left,
        jnp.where(event >= self._threshold, self._log_prob_right,
                  super().log_prob(event)))

  def mode(self):
    return self.bijector.forward(self.distribution.mode())

  def entropy(self, seed=None):
    # We return an estimation using a single sample of the log_det_jacobian.
    # We can still do some backpropagation with this estimate.
    return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
        self.distribution.sample(seed=seed), event_ndims=0)

  @classmethod
  def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
    td_properties = super()._parameter_properties(dtype,
                                                  num_classes=num_classes)
    del td_properties['bijector']
    return td_properties


class NormalTanhDistribution(hk.Module):
  """Module that produces a TanhTransformedDistribution distribution."""

  def __init__(self,
               num_dimensions: int,
               min_scale: float = 1e-3,
               w_init: hk_init.Initializer = hk_init.VarianceScaling(
                   1.0, 'fan_in', 'uniform'),
               b_init: hk_init.Initializer = hk_init.Constant(0.)):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of a distribution.
      min_scale: Minimum standard deviation.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    super().__init__(name='Normal')
    self._min_scale = min_scale
    self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._bijector=tfp.bijectors.Tanh()

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    loc = self._loc_layer(inputs)
    loc = 10 * self._bijector.forward(loc / 10)
    scale = self._scale_layer(inputs)
    scale = jax.nn.softplus(scale) + self._min_scale
    distribution = tfd.Normal(loc=loc, scale=scale)
    return tfd.Independent(
        TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1)


def select_cube_centers(num_cubes: int) -> jnp.ndarray:
  """Canonical PD ``select_action`` values for cube ids ``0 .. num_cubes-1``.

  Matches BuilderBench ``PDWrapper.get_action`` encoding so env digitize
  recovers the intended cube id.
  """
  ids = jnp.arange(num_cubes, dtype=jnp.float32)
  return (((2.0 * ids + 1.0) * jnp.pi / num_cubes) - jnp.pi) / jnp.pi


def select_action_to_cube_id(select: jnp.ndarray, num_cubes: int) -> jnp.ndarray:
  """Invert continuous select ∈ [-1, 1] → discrete cube id (BuilderBench digitize)."""
  bins = jnp.arange(1, num_cubes + 1) * (2.0 * jnp.pi / float(num_cubes))
  cube_id = jnp.digitize(jnp.pi * select + jnp.pi, bins)
  return jnp.clip(cube_id, 0, num_cubes - 1)


class HybridSelectDistribution:
  """Joint policy: tanh-Gaussian on continuous dims + Categorical select.

  Samples / modes are packed as continuous actions ``[..., :-1]`` concatenated
  with the PD-encoded select center for the chosen cube class.  ``log_prob``
  digitizes the stored select dim back to a class so PPO can train against
  continuous env actions without changing the env API.
  """

  hybrid_select: bool = True

  def __init__(
      self,
      continuous_dist: tfd.Distribution,
      categorical_dist: tfd.Distribution,
      num_select_classes: int,
  ):
    self.continuous_dist = continuous_dist
    self.categorical_dist = categorical_dist
    self.num_select_classes = int(num_select_classes)
    self._centers = select_cube_centers(self.num_select_classes)

  def _select_from_class(self, cube_id: jnp.ndarray) -> jnp.ndarray:
    select = self._centers[cube_id]
    return select[..., None]

  def sample(self, seed=None, sample_shape=()):
    key_cont, key_cat = jax.random.split(seed)
    cont = self.continuous_dist.sample(seed=key_cont, sample_shape=sample_shape)
    cube_id = self.categorical_dist.sample(seed=key_cat, sample_shape=sample_shape)
    return jnp.concatenate([cont, self._select_from_class(cube_id)], axis=-1)

  def mode(self):
    cont = self.continuous_dist.mode()
    cube_id = self.categorical_dist.mode()
    return jnp.concatenate([cont, self._select_from_class(cube_id)], axis=-1)

  def log_prob(self, actions: jnp.ndarray) -> jnp.ndarray:
    cont_lp = self.continuous_dist.log_prob(actions[..., :-1])
    cube_id = select_action_to_cube_id(actions[..., -1], self.num_select_classes)
    cat_lp = self.categorical_dist.log_prob(cube_id)
    return cont_lp + cat_lp

  def entropy(self, seed=None):
    # TanhTransformedDistribution.entropy may need a seed; pass through if set.
    try:
      cont_h = self.continuous_dist.entropy(seed=seed)
    except TypeError:
      cont_h = self.continuous_dist.entropy()
    return cont_h + self.categorical_dist.entropy()


class NormalTanhCategoricalSelect(hk.Module):
  """Shared-trunk hybrid actor head: Gaussian continuous + categorical select.

  Continuous dims use the same mean/std heads as ``NormalTanhDistribution``.
  Select uses a linear logits head with ``num_select_classes`` outputs (one
  class per cube, indices ``0 .. n-1``).
  """

  def __init__(
      self,
      num_continuous: int,
      num_select_classes: int,
      min_scale: float = 1e-3,
      w_init: hk_init.Initializer = hk_init.VarianceScaling(
          1.0, 'fan_in', 'uniform'),
      b_init: hk_init.Initializer = hk_init.Constant(0.),
      name: Optional[str] = None,
  ):
    super().__init__(name=name or 'NormalTanhCategoricalSelect')
    self._num_continuous = int(num_continuous)
    self._num_select_classes = int(num_select_classes)
    self._min_scale = min_scale
    self._cont_head = NormalTanhDistribution(
        self._num_continuous, min_scale=min_scale, w_init=w_init, b_init=b_init)
    self._select_logits = hk.Linear(
        self._num_select_classes, w_init=w_init, b_init=b_init,
        name='select_logits')

  def __call__(self, inputs: jnp.ndarray) -> HybridSelectDistribution:
    continuous_dist = self._cont_head(inputs)
    logits = self._select_logits(inputs)
    categorical_dist = tfd.Categorical(logits=logits)
    return HybridSelectDistribution(
        continuous_dist, categorical_dist, self._num_select_classes)


class HybridSelectWaypointDistribution:
  """Categorical select + per-cube 3D waypoint + shared yaw.

  Env action packing stays ``[xyz(3), yaw(1), select(1)]``.  Select is a
  categorical over ``n`` cubes; each cube has its own tanh-Gaussian waypoint
  head (``n × 3``).  Sampling / mode / log_prob use the waypoint head of the
  chosen cube.  Yaw uses one shared 1D tanh-Gaussian head.

  Entropy is ``H(cat) + E_{k~cat}[H(wp_k)] + H(yaw)``.
  """

  hybrid_select: bool = True
  hybrid_select_waypoint: bool = True

  def __init__(
      self,
      waypoint_loc: jnp.ndarray,
      waypoint_scale: jnp.ndarray,
      yaw_dist: tfd.Distribution,
      categorical_dist: tfd.Distribution,
      num_select_classes: int,
  ):
    # waypoint_loc / scale: [B, n, 3] pre-tanh Normal params.
    self.waypoint_loc = waypoint_loc
    self.waypoint_scale = waypoint_scale
    self.yaw_dist = yaw_dist
    self.categorical_dist = categorical_dist
    self.num_select_classes = int(num_select_classes)
    self._centers = select_cube_centers(self.num_select_classes)
    # Diagnostics / det-select helpers: continuous Gaussian for the mode cube.
    self.continuous_dist = self._packed_continuous_dist(
        self.categorical_dist.mode())

  def _select_from_class(self, cube_id: jnp.ndarray) -> jnp.ndarray:
    return self._centers[cube_id][..., None]

  def _gather_wp_params(self, cube_id: jnp.ndarray):
    idx = cube_id.astype(jnp.int32)[..., None, None]  # [B, 1, 1]
    loc = jnp.take_along_axis(self.waypoint_loc, idx, axis=1).squeeze(1)
    scale = jnp.take_along_axis(self.waypoint_scale, idx, axis=1).squeeze(1)
    return loc, scale

  def _wp_dist(self, cube_id: jnp.ndarray) -> tfd.Distribution:
    loc, scale = self._gather_wp_params(cube_id)
    return tfd.Independent(
        TanhTransformedDistribution(tfd.Normal(loc=loc, scale=scale)),
        reinterpreted_batch_ndims=1)

  def _packed_continuous_dist(self, cube_id: jnp.ndarray) -> tfd.Distribution:
    """Tanh-Gaussian over ``[xyz, yaw]`` for diagnostics / continuous_dist."""
    wp_loc, wp_scale = self._gather_wp_params(cube_id)
    yaw_td = self.yaw_dist.distribution
    yaw_normal = yaw_td.distribution
    loc = jnp.concatenate([wp_loc, yaw_normal.loc], axis=-1)
    scale = jnp.concatenate([wp_scale, yaw_normal.scale], axis=-1)
    threshold = float(getattr(yaw_td, '_threshold', 0.999))
    return tfd.Independent(
        TanhTransformedDistribution(
            tfd.Normal(loc=loc, scale=scale), threshold=threshold),
        reinterpreted_batch_ndims=1)

  def sample(self, seed=None, sample_shape=()):
    key_wp, key_yaw, key_cat = jax.random.split(seed, 3)
    cube_id = self.categorical_dist.sample(
        seed=key_cat, sample_shape=sample_shape)
    wp = self._wp_dist(cube_id).sample(
        seed=key_wp, sample_shape=sample_shape)
    yaw = self.yaw_dist.sample(seed=key_yaw, sample_shape=sample_shape)
    return jnp.concatenate(
        [wp, yaw, self._select_from_class(cube_id)], axis=-1)

  def mode(self):
    cube_id = self.categorical_dist.mode()
    wp = self._wp_dist(cube_id).mode()
    yaw = self.yaw_dist.mode()
    return jnp.concatenate(
        [wp, yaw, self._select_from_class(cube_id)], axis=-1)

  def log_prob(self, actions: jnp.ndarray) -> jnp.ndarray:
    cube_id = select_action_to_cube_id(
        actions[..., -1], self.num_select_classes)
    wp_lp = self._wp_dist(cube_id).log_prob(actions[..., :3])
    yaw_lp = self.yaw_dist.log_prob(actions[..., 3:4])
    cat_lp = self.categorical_dist.log_prob(cube_id)
    return wp_lp + yaw_lp + cat_lp

  def entropy(self, seed=None):
    # H(cat) + Σ_k π(k) H(wp_k) + H(yaw)
    probs = self.categorical_dist.probs_parameter()  # [B, n]
    bsz, n_cls, wp_dim = self.waypoint_loc.shape
    flat_loc = self.waypoint_loc.reshape((bsz * n_cls, wp_dim))
    flat_scale = self.waypoint_scale.reshape((bsz * n_cls, wp_dim))
    flat_wp = tfd.Independent(
        TanhTransformedDistribution(
            tfd.Normal(loc=flat_loc, scale=flat_scale)),
        reinterpreted_batch_ndims=1)
    if seed is not None:
      key_wp, key_yaw = jax.random.split(seed)
      try:
        wp_h_flat = flat_wp.entropy(seed=key_wp)
      except TypeError:
        wp_h_flat = flat_wp.entropy()
      try:
        yaw_h = self.yaw_dist.entropy(seed=key_yaw)
      except TypeError:
        yaw_h = self.yaw_dist.entropy()
    else:
      try:
        wp_h_flat = flat_wp.entropy()
      except TypeError:
        # TanhTransformedDistribution.entropy requires a seed; invent one.
        wp_h_flat = flat_wp.entropy(seed=jax.random.PRNGKey(0))
      try:
        yaw_h = self.yaw_dist.entropy()
      except TypeError:
        yaw_h = self.yaw_dist.entropy(seed=jax.random.PRNGKey(0))
    wp_h = wp_h_flat.reshape((bsz, n_cls))  # [B, n]
    wp_h_exp = jnp.sum(probs * wp_h, axis=-1)
    return self.categorical_dist.entropy() + wp_h_exp + yaw_h


class NormalTanhCategoricalSelectWaypoint(hk.Module):
  """Hybrid actor: n-way categorical select + n×3 waypoint heads + shared yaw.

  Shared trunk features feed:
    * ``select_logits`` → Categorical over ``num_select_classes`` cubes
    * ``waypoint_{loc,scale}`` → ``[n, 3]`` tanh-Gaussian params per cube
    * shared 1D yaw tanh-Gaussian (PD action dim 3)
  """

  def __init__(
      self,
      num_select_classes: int,
      waypoint_dim: int = 3,
      min_scale: float = 1e-3,
      w_init: hk_init.Initializer = hk_init.VarianceScaling(
          1.0, 'fan_in', 'uniform'),
      b_init: hk_init.Initializer = hk_init.Constant(0.),
      name: Optional[str] = None,
  ):
    super().__init__(name=name or 'NormalTanhCategoricalSelectWaypoint')
    self._num_select_classes = int(num_select_classes)
    self._waypoint_dim = int(waypoint_dim)
    self._min_scale = min_scale
    n3 = self._num_select_classes * self._waypoint_dim
    self._wp_loc = hk.Linear(n3, w_init=w_init, b_init=b_init,
                             name='waypoint_loc')
    self._wp_scale = hk.Linear(n3, w_init=w_init, b_init=b_init,
                               name='waypoint_scale')
    self._yaw_head = NormalTanhDistribution(
        1, min_scale=min_scale, w_init=w_init, b_init=b_init)
    self._select_logits = hk.Linear(
        self._num_select_classes, w_init=w_init, b_init=b_init,
        name='select_logits')
    self._bijector = tfp.bijectors.Tanh()

  def __call__(self, inputs: jnp.ndarray) -> HybridSelectWaypointDistribution:
    bsz = inputs.shape[0]
    n = self._num_select_classes
    d = self._waypoint_dim
    loc = self._wp_loc(inputs)
    # Match NormalTanhDistribution pre-squash of means.
    loc = 10.0 * self._bijector.forward(loc / 10.0)
    scale = jax.nn.softplus(self._wp_scale(inputs)) + self._min_scale
    loc = loc.reshape((bsz, n, d))
    scale = scale.reshape((bsz, n, d))
    yaw_dist = self._yaw_head(inputs)
    logits = self._select_logits(inputs)
    categorical_dist = tfd.Categorical(logits=logits)
    return HybridSelectWaypointDistribution(
        loc, scale, yaw_dist, categorical_dist, self._num_select_classes)


class MultivariateNormalDiagHead(hk.Module):
  """Module that produces a tfd.MultivariateNormalDiag distribution."""

  def __init__(self,
               num_dimensions: int,
               init_scale: float = 0.3,
               min_scale: float = 1e-6,
               w_init: hk_init.Initializer = hk_init.VarianceScaling(1e-4),
               b_init: hk_init.Initializer = hk_init.Constant(0.)):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of MVN distribution.
      init_scale: Initial standard deviation.
      min_scale: Minimum standard deviation.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    super().__init__(name='MultivariateNormalDiagHead')
    self._min_scale = min_scale
    self._init_scale = init_scale
    self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    loc = self._loc_layer(inputs)
    scale = jax.nn.softplus(self._scale_layer(inputs))
    scale *= self._init_scale / jax.nn.softplus(0.)
    scale += self._min_scale
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)


class CategoricalValueHead(hk.Module):
  """Network head that produces a categorical distribution and value."""

  def __init__(
      self,
      num_values: int,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._logit_layer = hk.Linear(num_values)
    self._value_layer = hk.Linear(1)

  def __call__(self, inputs: jnp.ndarray):
    logits = self._logit_layer(inputs)
    value = jnp.squeeze(self._value_layer(inputs), axis=-1)
    return (tfd.Categorical(logits=logits), value)


class DiscreteValued(hk.Module):
  """C51-style head.

  For each action, it produces the logits for a discrete distribution over
  atoms. Therefore, the returned logits represents several distributions, one
  for each action.
  """

  def __init__(
      self,
      num_actions: int,
      head_units: int = 512,
      num_atoms: int = 51,
      v_min: float = -1.0,
      v_max: float = 1.0,
  ):
    super().__init__('DiscreteValued')
    self._num_actions = num_actions
    self._num_atoms = num_atoms
    self._atoms = jnp.linspace(v_min, v_max, self._num_atoms)
    self._network = hk.nets.MLP([head_units, num_actions * num_atoms])

  def __call__(self, inputs: jnp.ndarray):
    q_logits = self._network(inputs)
    q_logits = jnp.reshape(q_logits, (-1, self._num_actions, self._num_atoms))
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * self._atoms, axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return q_values, q_logits, self._atoms
