#!/usr/bin/env python3
"""Regenerate offline NF figures from metrics without loading JAX or CUDA."""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

from scripts.offline_nf.core import replot_output  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--domain_label", required=True)
  parser.add_argument("--batch_size", type=int, default=256)
  args = parser.parse_args()
  for path in replot_output(
      args.output_dir, domain_label=args.domain_label,
      batch_size=args.batch_size):
    print(path)


if __name__ == "__main__":
  main()
