"""Default logger."""

import dataclasses
import logging
from typing import Any, Callable, Dict, Mapping, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal


# Metrics from evaluator/actor loops that contain non-scalar or internal values
# we don't need to clutter the wandb dashboard with.
_WANDB_SKIP_KEYS = frozenset({'steps', 'walltime'})


class WandbLogger(base.Logger):
  """Logger that writes metrics to Weights & Biases.

  Lazily initialises wandb on the first call so that importing this module
  does not require wandb to be installed when use_wandb=False.

  All metrics are prefixed with ``label`` (e.g. "train/critic_loss",
  "eval/success") to keep the wandb dashboard organised.  A single wandb
  run is shared across all loggers in the same process — the first
  WandbLogger to be constructed calls ``wandb.init``; subsequent ones
  reuse the existing run.
  """

  def __init__(
      self,
      label: str = '',
      config: Optional[Dict[str, Any]] = None,
      project: str = 'sgcrl',
      entity: str = '',
      name: str = '',
      group: str = '',
  ):
    import wandb  # pylint: disable=import-outside-toplevel
    self._wandb = wandb
    # Map human-readable labels to cleaner wandb prefixes.
    _prefix_map = {'learner': 'train', 'evaluator': 'eval', 'actor': 'actor'}
    self._prefix = _prefix_map.get(label, label)

    if wandb.run is None:
      init_kwargs: Dict[str, Any] = dict(project=project, config=config or {})
      if entity:
        init_kwargs['entity'] = entity
      if name:
        init_kwargs['name'] = name
      if group:
        init_kwargs['group'] = group
      wandb.init(**init_kwargs)

  def write(self, data: Mapping[str, Any]) -> None:
    payload = {}
    for k, v in data.items():
      if k in _WANDB_SKIP_KEYS:
        continue
      try:
        payload[f'{self._prefix}/{k}'] = float(v)
      except (TypeError, ValueError):
        pass  # skip non-scalar values
    if payload:
      self._wandb.log(payload)

  def close(self) -> None:
    pass  # keep the run open; let the process exit close it naturally


def make_default_logger(
    label: str,
    save_data: bool = True,
    save_dir: str = 'logs',
    add_uid: bool = True,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = 'steps',
    use_wandb: bool = False,
    wandb_kwargs: Optional[Dict[str, Any]] = None,
) -> base.Logger:
  """Makes a default Acme logger.

  Args:
    label: Name to give to the logger.
    save_data: Whether to persist data.
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: Ignored.
    use_wandb: If True, also log to Weights & Biases.
    wandb_kwargs: Extra keyword arguments forwarded to WandbLogger
      (project, entity, name, group, config).

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  del steps_key
  if not print_fn:
    print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers = [terminal_logger]

  if save_data:
    loggers.append(csv.CSVLogger(label=label, directory_or_file=save_dir, add_uid=add_uid))

  if use_wandb:
    loggers.append(WandbLogger(label=label, **(wandb_kwargs or {})))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.TimeFilter(logger, time_delta)

  return logger
