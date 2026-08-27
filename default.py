"""Default logger."""

import logging
from typing import Any, Callable, Mapping, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal


class WandbLogger(base.Logger):
  """Acme Logger that forwards write() calls to an already-open wandb run.

  Keys are prefixed with `{label}/` (mirroring
  scripts/csv_runs_to_wandb.py's `f"{prefix}/{k}"` convention) so the
  'learner' and 'eval' loggers -- both of which currently emit a
  'learner_steps' field -- never collide on the same run. The wandb step
  is taken explicitly from data['learner_steps'] rather than relying on
  wandb's implicit auto-increment, because 'learner' writes every PPO
  iteration while 'eval' writes only every 10th; auto-increment would
  desync the two loggers' step axes on one run.
  """

  def __init__(self, label: str, run: Any = None):
    self._label = label
    self._run = run

  def write(self, data: Mapping[str, Any]) -> None:
    if self._run is None:
      return
    step = data.get('learner_steps')
    payload = {}
    for k, v in data.items():
      if isinstance(v, bool) or isinstance(v, (int, float)) or hasattr(v, 'item'):
        payload[f'{self._label}/{k}'] = v
    if payload:
      self._run.log(payload, step=int(step) if step is not None else None)

  def close(self) -> None:
    pass  # run.finish() is owned by the training entrypoint, not this logger.


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
    wandb_run: Optional[Any] = None,
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
    wandb_run: If given (an active `wandb.init()` return value), a
      WandbLogger is added so every write() also logs to that run.

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

  if wandb_run is not None:
    loggers.append(WandbLogger(label=label, run=wandb_run))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.TimeFilter(logger, time_delta)

  return logger
