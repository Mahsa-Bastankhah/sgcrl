import sys
from pathlib import Path
import tensorflow as tf   # works with TF‑V2 checkpoints (Acme / JAX uses this format)
import pandas as pd
import matplotlib.pyplot as plt


import sys, pickle, pprint, tensorflow as tf

import numpy as np
import tensorflow as tf
from pathlib import Path

import hashlib, pickle, numpy as np
import tensorflow as tf
from pathlib import Path

import functools, pickle, hashlib, numpy as np
import tensorflow as tf
from pathlib import Path


import os
import re
import glob
import pickle
import numpy as np
import tensorflow as tf


def load_ckpt(path: str, *, first: bool = True):
    # --------------------------------------------------------
    # 1) Resolve prefix  (dir → ckpt-0 … or latest)
    # --------------------------------------------------------
    if os.path.isdir(path):
        if first:
            idx_files = sorted(
                glob.glob(os.path.join(path, "ckpt-*.index")),
                key=lambda p: int(re.search(r"ckpt-(\d+)\.index", p).group(1)),
            )
            if not idx_files:
                raise FileNotFoundError(f"No ckpt-*.index files inside {path}")
            ckpt_prefix = idx_files[0][:-6]  # strip ".index"
        else:
            ckpt_prefix = tf.train.latest_checkpoint(path)
            if ckpt_prefix is None:
                raise FileNotFoundError(f"No checkpoints inside {path}")
    else:
        ckpt_prefix = path

    reader = tf.train.load_checkpoint(ckpt_prefix)
    blob   = reader.get_tensor("learner/.ATTRIBUTES/py_state")
    state  = pickle.loads(blob)
    print(f"\n=== decoded state from {ckpt_prefix}=== policy params")
    print(state.policy_params)

    return state.policy_params, state.q_params



def compare_ckpts(addr1: str, addr2: str):
    """
    Compare policy_params and q_params between two checkpoints.

    Args:
      addr1: either a checkpoint-prefix ('.../ckpt-123') or a directory containing ckpt-*.index.
      addr2: same as addr1, for the second model.
      ckpt_number: if provided *and* addr1/addr2 are dirs, picks 'ckpt-{ckpt_number}'.

    Returns:
      {
        "policy_same": bool,
        "policy_diffs": [keys…],
        "q_same": bool,
        "q_diffs": [keys…],
      }
    """


    p1, q1 = load_ckpt(addr1, first=True)
    p2, q2 = load_ckpt(addr2, first=True)

    



def inspect_counter(counter_dir: str, ckpt_num: int):
    """
    Print episode‑ and evaluator‑step counters from a TensorFlow checkpoint.

    Args
    ----
    counter_dir : path that contains ckpt‑N.index / ckpt‑N.data‑… files.
    ckpt_num    : integer N (e.g. 2 for ckpt‑2.*)
    """
    ckpt_prefix = str(Path(counter_dir) / f"ckpt-{ckpt_num}")
    reader      = tf.train.load_checkpoint(ckpt_prefix)



    for name, shape in reader.get_variable_to_shape_map().items():
        print(f"{name:<60} {shape}")

    print(f"\nVariables in {ckpt_prefix} that look like episode / evaluator steps:")
    for name, shape in reader.get_variable_to_shape_map().items():
        low = name.lower()
        if any(word in low for word in ("episode", "evaluator", "step")):
            val = reader.get_tensor(name)
            print(f"{name:<50} {val}")

def show_counter_state(ckpt_prefix: str):
    reader = tf.train.load_checkpoint(ckpt_prefix)

    # Name of the PythonState tensor that holds the dict:
    key = "counter/.ATTRIBUTES/py_state"
    if key not in reader.get_variable_to_shape_map():
        print(f"❌ '{key}' not found in {ckpt_prefix}")
        return

    blob = reader.get_tensor(key)           # dtype=string scalar → bytes
    state = pickle.loads(blob)              # usually a dict: {'episode': int, ...}

    print(f"\n=== decoded counter state from {ckpt_prefix} ===")
    pprint.pprint(state, width=80)

def load_saved_params(save_dir: str, name: str = "initial_params.pkl"):
    """
    Returns the dict you originally saved.
    """
    path = Path(save_dir) / name
    with open(path, "rb") as f:
        print(f"Loading saved params from {path}")
        saved = pickle.load(f)
        policy_params = saved["policy_params"]
        q_params      = saved["q_params"]
        print(policy_params, flush= True)
        print(q_params, flush= True)  


seed = 4000
#34, 36
ckpts = [1 , 2]
for ckpt in ckpts:
    show_counter_state(f"logs/contrastive_cpc_point_Wall11x11_{seed}/checkpoints/counter/ckpt-{ckpt}")

#show_counter_state("logs/contrastive_cpc_point_Impossible_42/checkpoints/counter/ckpt-240")

# load_saved_params("logs/contrastive_cpc_point_Wall11x11_4/checkpoints/learner", "initial_params.pkl")
# load_saved_params("logs/contrastive_cpc_point_Wall11x11_4/checkpoints/learner", "initial_params_old.pkl")
