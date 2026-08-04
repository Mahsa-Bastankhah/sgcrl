"""Controlled 3D (x,y,z) random-walk density-estimation probe.

Trains CRL / NF / TD3 estimators from ``contrastive/`` on data from a
JAX vectorized additive-noise env with a uniform random policy.
"""

from xyz_density.env import JaxXYZVecEnv

__all__ = ['JaxXYZVecEnv']
