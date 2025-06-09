# mid_goal_selector.py
import jax
import jax.numpy as jnp
import numpy as np
from acme.agents.jax import actors

from typing import Generic, Optional

from acme import adders
from acme import core
from acme import types
from acme.agents.jax import actor_core
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
import dm_env
import jax
import jax.numpy as jnp
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
from acme import wrappers
import dm_env
import logging, pathlib

log_dir = pathlib.Path("logs/debug")
log_dir.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler(log_dir / "midpoint.log", mode='a')
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"))
logger = logging.getLogger('midpoint')
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)  

class MidpointGoalSelector:
  """Pick w = argmin_s ||ψ(s) - ½(ψ(s0)+ψ(g))|| among free cells."""

  def __init__(self, env, networks, variable_client):
    self._env = env
    self._nets = networks
    self._vc   = variable_client
    ys, xs = np.where(env.walls == 0)
    self._free_cells = np.stack([ys + 0.5, xs + 0.5], axis=1).astype(np.float32)

    # convenience: zero action tensor
    self._zero_act = np.zeros((1, np.prod(env.action_space.shape)), np.float32)

  # ------- helpers ----------------------------------------------------------
  def _psi_state(self, params_q, s):
    """ψ(s) from sa_repr with a zero action."""
    obs = np.concatenate([s, s*0], axis=-1)[None]   # [1, obs_dim*2]
    _, sa_repr, _ = self._nets.q_network.apply(params_q, obs, self._zero_act)
    return sa_repr[0]                              # [repr_dim]

  def _psi_goal(self, params_q, g):
    """ψ(g) from g_repr; state part is dummy."""
    obs = np.concatenate([g*0, g], axis=-1)[None]
    _, _, g_repr = self._nets.q_network.apply(params_q, obs, self._zero_act)
    return g_repr[0]
  # --------------------------------------------------------------------------
  def pick_goal(self, s0, g):
      # pull freshest weights
      self._vc.update_and_wait()
      params_q = self._vc.params[1]

      # compute representations
      psi_s0 = self._psi_state(params_q, s0)   # shape [repr_dim]
      psi_g  = self._psi_goal (params_q, g)    # shape [repr_dim]

      # draw a random interpolation factor α in [0,1]
      alpha = np.random.uniform(0.0, 1.0)
      #alpha = 0.5

      # form the interpolated psi target
      psi_w = (1 - alpha) * psi_s0 + alpha * psi_g

      # encode all free cells as goals (batched obs → cells)
      obs_cells = np.hstack([
          np.zeros_like(self._free_cells),  # dummy state part
          self._free_cells                  # goal coords
      ])
      _, _, psi_cells = self._nets.q_network.apply(
          params_q, obs_cells, np.zeros_like(self._free_cells)
      )

      # find the free cell whose psi is closest to our psi_w
      dists = jnp.linalg.norm(psi_cells - psi_w, axis=1)
      idx   = int(jnp.argmin(dists))

      return self._free_cells[idx]




class MidpointEnvWrapper(wrappers.EnvironmentWrapper):
    def __init__(self, base_env, networks, variable_client):
        super().__init__(base_env)
        self._gs = MidpointGoalSelector(
            self._environment, networks, variable_client)

    def reset(self) -> dm_env.TimeStep:
        # 1) call the base env reset
        ts = self._environment.reset()  

        # 2) extract original obs dtype so we know what to cast back to
        obs_dtype = ts.observation.dtype

        # 3) pick new midpoint goal
        s0, g = ts.observation[:2], ts.observation[2:]
        new_goal = self._gs.pick_goal(s0, g)
        self._environment.set_goal(new_goal)

        # 4) rebuild the obs and cast it
        #    self._environment.state is float64 by default, so upcast happens
        raw_state = self._environment.state
        new_obs = np.concatenate([raw_state, new_goal], axis=-1)
        new_obs = new_obs.astype(obs_dtype)       # <<< cast back to float32

        # 5) return a TimeStep with the corrected dtype
        return ts._replace(observation=new_obs)

    def step(self, action) -> dm_env.TimeStep:
        return self._environment.step(action)





# class MidpointActor(actors.GenericActor):

#   def __init__(self, actor_core, rng, variable_client, adder,
#                env, goal_selector):
#     super().__init__(actor_core, rng, variable_client, adder, backend='cpu')
#     self._env = env
#     self._gs  = goal_selector
#     self._logger = logger

#   def observe_first(self, timestep):
#     s0, g = timestep.observation[:2], timestep.observation[2:]
#     new_goal = self._gs.pick_goal(s0, g)     # ψ‑midpoint trick
#     self._env.set_goal(new_goal)
#     print("[MidpointActor] new_goal=", new_goal, flush = True)

#     new_obs = np.concatenate([s0, new_goal], dtype=np.float32)
#     super().observe_first(timestep._replace(observation=new_obs))

  
#   def select_action(self,
#                     observation):
#     policy_params = self._params[0]
#     if (policy_params['mlp/~/linear_0']['b'] == 0).all():
#       shape = policy_params['Normal/~/linear']['b'].shape
#       rng, self._state = jax.random.split(self._state)
#       action = jax.random.uniform(key=rng, shape=shape,
#                                   minval=-1.0, maxval=1.0)
#     else:
#       action, self._state = self._policy(policy_params, observation,
#                                          self._state)
#     return utils.to_numpy(action)
#     # ---------------------------------------------------------------------

