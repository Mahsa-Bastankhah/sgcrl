#!/usr/bin/env python3
"""Copy stdin to stdout, failing once output exceeds a byte limit."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--max-mib', type=float, default=8.0)
  args = parser.parse_args()

  limit = max(1, int(args.max_mib * 1024 * 1024))
  written = 0
  source = sys.stdin.buffer
  sink = sys.stdout.buffer

  while True:
    chunk = source.read(64 * 1024)
    if not chunk:
      sink.flush()
      return 0
    remaining = limit - written
    if len(chunk) <= remaining:
      sink.write(chunk)
      sink.flush()
      written += len(chunk)
      continue

    if remaining > 0:
      sink.write(chunk[:remaining])
    sink.write(
        b'\n[cap_log_stream] ERROR: output exceeded configured limit; '
        b'aborting producer to prevent a runaway Slurm log.\n')
    sink.flush()
    return 86


if __name__ == '__main__':
  raise SystemExit(main())
