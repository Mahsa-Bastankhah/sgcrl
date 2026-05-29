"""Default logger."""

import logging
from typing import Any, Callable, Mapping, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal
import numpy as np


def _format_key(key: str) -> str:
  return key.replace('_', ' ').title()


def _format_value(value: Any, precision: int = 3) -> str:
  """Format a scalar for terminal logging with configurable precision."""
  value = base.to_numpy(value)
  if isinstance(value, (float, np.number)):
    v = float(value)
    if np.isnan(v):
      return 'nan'
    if np.isinf(v):
      return 'inf' if v > 0 else '-inf'
    if v != 0.0 and (abs(v) < 10 ** (-precision) or abs(v) >= 10 ** (precision + 2)):
      return f'{v:.{precision}e}'
    return f'{v:.{precision}f}'
  return f'{value}'


def make_serialize_fn(precision: int = 3):
  """Build a terminal serialize function with *precision* decimal digits."""
  def serialize(values: base.LoggingData) -> str:
    return ' | '.join(
        f'{_format_key(k)} = {_format_value(v, precision=precision)}'
        for k, v in sorted(values.items()))
  return serialize


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
    float_precision: int = 3,
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
    float_precision: Decimal digits for floats in terminal output.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  del steps_key
  if not print_fn:
    print_fn = logging.info
  terminal_serialize = make_serialize_fn(precision=float_precision)
  terminal_logger = terminal.TerminalLogger(
      label=label, print_fn=print_fn, serialize_fn=terminal_serialize)

  loggers = [terminal_logger]

  if save_data:
    loggers.append(csv.CSVLogger(label=label, directory_or_file=save_dir, add_uid=add_uid))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.TimeFilter(logger, time_delta)

  return logger
