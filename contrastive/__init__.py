"""Contrastive RL agent."""

# Launchpad `local_mp` workers import this package before any script main;
# they never execute lp_contrastive.py, so patch JAX for older acme here.
import sgcrl_jax_acme_compat  # noqa: F401

from contrastive.config import ContrastiveConfig
from contrastive.config import target_entropy_from_env_spec
from contrastive.networks import apply_policy_and_sample
from contrastive.networks import ContrastiveNetworks
from contrastive.networks import make_networks

# Heavy / launchpad-dependent modules are lazy so PPO+BuilderBench does not
# require dm-launchpad or metaworld at import time.
_LAZY = {
    'DistributedContrastive': ('contrastive.agents', 'DistributedContrastive'),
    'ContrastiveBuilder': ('contrastive.builder', 'ContrastiveBuilder'),
    'ContrastiveLearner': ('contrastive.learning', 'ContrastiveLearner'),
}


def __getattr__(name: str):
  if name in _LAZY:
    module, attr = _LAZY[name]
    import importlib
    return getattr(importlib.import_module(module), attr)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
  return sorted(list(globals().keys()) + list(_LAZY.keys()))
