"""CSV logger that preserves an existing header/schema on resume.

Acme's ``CSVLogger`` infers columns from the first ``write()`` after restart
and appends rows in sorted-key order, which misaligns with an existing header
when the first post-resume dict omits keys (e.g. flow-dense / CRL columns).
"""
from __future__ import annotations

import csv
import os
import time
from typing import List, Optional, TextIO, Union

from absl import logging

from acme.utils import paths
from acme.utils.loggers import base


class ResumableCSVLogger(base.Logger):
  """Append-only CSV logger compatible with checkpoint resume."""

  _open = open

  def __init__(
      self,
      directory_or_file: Union[str, TextIO] = '~/acme',
      label: str = '',
      time_delta: float = 0.0,
      add_uid: bool = True,
      flush_every: int = 1,
  ):
    # Default flush_every=1: sparse loggers (e.g. eval every 200 PPO iters)
    # otherwise buffer for hours and plots/CSV readers see only the first row.
    if flush_every <= 0:
      raise ValueError(
          f'`flush_every` must be a positive integer (got {flush_every}).')

    self._last_log_time = time.time() - time_delta
    self._time_delta = time_delta
    self._flush_every = flush_every
    self._add_uid = add_uid
    self._writer: Optional[csv.DictWriter] = None
    self._fieldnames: Optional[List[str]] = None
    self._file_owner = False
    self._file = self._create_file(directory_or_file, label)
    self._writes = 0
    logging.info('Logging to %s', self.file_path)

  def _create_file(
      self,
      directory_or_file: Union[str, TextIO],
      label: str,
  ) -> TextIO:
    if isinstance(directory_or_file, str):
      directory = paths.process_path(
          directory_or_file, 'logs', label, add_uid=self._add_uid)
      file_path = os.path.join(directory, 'logs.csv')
      self._file_owner = True
      return self._open(file_path, mode='a', newline='')

    file = directory_or_file
    if label:
      logging.info(
          'File, not directory, passed to ResumableCSVLogger; label not used.')
    if not file.mode.startswith('a'):
      raise ValueError('File must be open in append mode.')
    return file

  def _load_existing_fieldnames(self) -> Optional[List[str]]:
    if not self._file.tell():
      return None
    path = self.file_path
    try:
      with self._open(path, 'r', newline='') as fh:
        first = fh.readline().strip()
    except OSError:
      return None
    if not first:
      return None
    return [c.strip() for c in first.split(',') if c.strip()]

  def _ensure_writer(self, data: base.LoggingData) -> None:
    if self._writer is not None:
      return
    existing = self._load_existing_fieldnames()
    if existing:
      self._fieldnames = existing
    else:
      self._fieldnames = sorted(data.keys())
    self._writer = csv.DictWriter(
        self._file, fieldnames=self._fieldnames, extrasaction='ignore')
    if not self._file.tell():
      self._writer.writeheader()

  def write(self, data: base.LoggingData):
    now = time.time()
    elapsed = now - self._last_log_time
    if elapsed < self._time_delta:
      return
    self._last_log_time = now

    data = base.to_numpy(data)
    self._ensure_writer(data)
    assert self._writer is not None
    assert self._fieldnames is not None

    row = {k: float('nan') for k in self._fieldnames}
    for k, v in data.items():
      if k in row:
        row[k] = v
    self._writer.writerow(row)

    if self._writes % self._flush_every == 0:
      self.flush()
    self._writes += 1

  def close(self):
    self.flush()
    if self._file_owner:
      self._file.close()

  def flush(self):
    self._file.flush()

  @property
  def file_path(self) -> str:
    return self._file.name
