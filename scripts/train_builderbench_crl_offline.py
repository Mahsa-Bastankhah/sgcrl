#!/usr/bin/env python3
"""Train production PPO CRL on fixed BuilderBench PD episode datasets."""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

from scripts.offline_nf.cuda_init import initialize_torch_first  # noqa: E402

initialize_torch_first()

from scripts.offline_nf.crl import (  # noqa: E402
    make_crl_train_parser,
    run_crl_training,
)


def main() -> None:
  parser = make_crl_train_parser(
      "Train production PPO CRL on fixed BuilderBench PD datasets.")
  args = parser.parse_args()
  run_crl_training(
      args, domain_label="BuilderBench creative-3-task1",
      expected_dims=(10, 5, 9))


if __name__ == "__main__":
  main()
