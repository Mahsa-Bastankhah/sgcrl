"""Contrastive RL agent."""

# Launchpad `local_mp` workers import this package before any script main;
# they never execute lp_contrastive.py, so patch JAX for older acme here.
import sgcrl_jax_acme_compat  # noqa: F401

from contrastive.agents import DistributedContrastive
from contrastive.builder import ContrastiveBuilder
from contrastive.config import ContrastiveConfig
from contrastive.config import target_entropy_from_env_spec
from contrastive.learning import ContrastiveLearner
from contrastive.networks import apply_policy_and_sample
from contrastive.networks import ContrastiveNetworks
from contrastive.networks import make_networks
