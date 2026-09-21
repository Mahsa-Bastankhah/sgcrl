#!/usr/bin/env python3
"""Train the shared offline NF diagnostic on BuilderBench PD datasets."""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

from scripts.offline_nf.cuda_init import initialize_torch_first  # noqa: E402

initialize_torch_first()

from scripts.offline_nf.core import make_train_parser, run_training  # noqa: E402


def main() -> None:
  parser = make_train_parser(
      "Train compact conditional NFs on fixed BuilderBench PD datasets.")
  args = parser.parse_args()
  run_training(
      args, domain_label="BuilderBench creative-3-task1",
      expected_dims=(10, 5, 9), task_goal_stats_fraction=0.0)


if __name__ == "__main__":
  main()
