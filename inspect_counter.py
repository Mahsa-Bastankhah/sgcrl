import sys
from pathlib import Path
import tensorflow as tf   # works with TF‑V2 checkpoints (Acme / JAX uses this format)
import pandas as pd
import matplotlib.pyplot as plt


import sys, pickle, pprint, tensorflow as tf
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

show_counter_state("logs/contrastive_cpc_sawyer_bin_42/checkpoints/counter/ckpt-15")
#show_counter_state("logs/contrastive_cpc_point_Impossible_42/checkpoints/counter/ckpt-240")
