import pandas as pd
import matplotlib.pyplot as plt


import sys, pickle, pprint, tensorflow as tf


import matplotlib.pyplot as plt
def plot_sawyer_bin():
    # Function to load actor_episodes and last column from a CSV
    def load_actor_and_success(filepath):
        actor_episodes = []
        successes = []
        with open(filepath, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue  # Skip broken lines
                try:
                    actor = float(parts[0])
                    success = float(parts[-1])
                    actor_episodes.append(actor)
                    successes.append(success)
                except ValueError:
                    # Skip rows that can't be parsed
                    continue
        return actor_episodes, successes

    # Load both files
    x1, y1 = load_actor_and_success("./logs/contrastive_cpc_sawyer_bin_64/logs/evaluator/logs.csv")
    x2, y2 = load_actor_and_success("./logs/contrastive_cpc_sawyer_bin_52/logs/evaluator/logs.csv")
    x3, y3 = load_actor_and_success("./logs/contrastive_cpc_sawyer_bin_32/logs/evaluator/logs.csv")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label='max Q', marker='o')
    plt.plot(x2, y2, label='Actor, seed 52', marker='x')
    plt.plot(x3, y3, label='Actor, seed 32', marker='x')

    plt.xlabel('Actor Episodes')
    plt.ylabel('Success (last column)')
    plt.title('Success vs Actor Episodes, Sawyer bin')

    plt.xlim(15000, 50000)  # <-- Restrict x-axis to [15000, 50000]

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("success_vs_episodes_sawyer_bin.png")
    print("✅ Plot saved as success_vs_episodes_sawyer_bin.png")


def plot_four_rooms_compare(seed = 66):
    # Load both CSVs
    if seed == 64:
        df1 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_64/logs/evaluator/logs.csv")
        df2 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_64/b6085d8e-2520-11f0-8691-d85ed38c5779/logs/evaluator/logs.csv")
        df3 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_64/5dd72662-2788-11f0-b183-d85ed38c5779/logs/evaluator/logs.csv")
        df4 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_64/aa47da3c-2788-11f0-a9e0-d85ed38c5779/logs/evaluator/logs.csv")
    elif seed == 65:
        df1 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_65/b9610d32-27aa-11f0-b48b-d85ed38c5779/logs/evaluator/logs.csv")
        df2 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_65/ff937952-27aa-11f0-9da1-d85ed38c5779/logs/evaluator/logs.csv")
        df3 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_65/d604b0b0-27aa-11f0-8e77-d85ed38c5779/logs/evaluator/logs.csv")
        df4 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_65/effc5540-27aa-11f0-90e0-d85ed38c5779/logs/evaluator/logs.csv")
    elif seed == 66:
        df1 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_66/479c9a86-27af-11f0-8932-d85ed38c5779/logs/evaluator/logs.csv")
        df2 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_66/62ae7d26-27af-11f0-9f2d-d85ed38c5779/logs/evaluator/logs.csv")
        df3 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_66/4dbe6494-27af-11f0-a2a1-d85ed38c5779/logs/evaluator/logs.csv")
        df4 = pd.read_csv("./logs/contrastive_cpc_point_FourRooms_66/581f333c-27af-11f0-9ae5-d85ed38c5779/logs/evaluator/logs.csv")
    # Use the last column name from each file (handles changes in number of columns)
    last_col_df1 = df1.columns[-1]
    last_col_df2 = df2.columns[-1]
    last_col_df3 = df3.columns[-1]
    last_col_df4 = df4.columns[-1]

    # Just make sure actor_episodes is numeric (important for plotting)
    df1['evaluator_episodes'] = pd.to_numeric(df1['evaluator_episodes'], errors='coerce')
    df2['evaluator_episodes'] = pd.to_numeric(df2['evaluator_episodes'], errors='coerce')
    df3['evaluator_episodes'] = pd.to_numeric(df3['evaluator_episodes'], errors='coerce')
    df4['evaluator_episodes'] = pd.to_numeric(df4['evaluator_episodes'], errors='coerce')

    # Also make sure success values are numeric (sometimes parsing creates strings)
    df1[last_col_df1] = pd.to_numeric(df1[last_col_df1], errors='coerce')
    df2[last_col_df2] = pd.to_numeric(df2[last_col_df2], errors='coerce')
    df3[last_col_df3] = pd.to_numeric(df3[last_col_df3], errors='coerce')
    df4[last_col_df4] = pd.to_numeric(df4[last_col_df4], errors='coerce')

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(df1['evaluator_episodes'], df1[last_col_df1], label='SGCRL, max Q', marker='o')
    plt.plot(df2['evaluator_episodes'], df2[last_col_df2], label='SGCRL, Actor', marker='x')
    plt.plot(df3['evaluator_episodes'], df3[last_col_df3], label='Sample goals, max Q', marker='o')
    plt.plot(df4['evaluator_episodes'], df4[last_col_df4], label='Sample goals, Actor', marker='x')

    plt.xlabel('Evaluator Episodes')
    plt.ylabel('Success@1000')
    plt.title('Success@1000 vs Actor Episodes, FourRooms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    plt.savefig(f"success_vs_episodes_all_{seed}.png")  # You can change to .pdf, .svg, etc.
    print("Plot saved as success_vs_episodes.png")


def plot_stochastic_four_rooms(seed = 73):
        # Load both CSVs

    df1 = pd.read_csv(f"./logs/contrastive_cpc_stochastic_point_FourRooms_{seed}/logs/evaluator/logs.csv")

   
    # Use the last column name from each file (handles changes in number of columns)
    last_col_df1 = df1.columns[-1]


    # Just make sure actor_episodes is numeric (important for plotting)
    df1['evaluator_episodes'] = pd.to_numeric(df1['evaluator_episodes'][:5000], errors='coerce')
    
    # Also make sure success values are numeric (sometimes parsing creates strings)
    df1[last_col_df1] = pd.to_numeric(df1[last_col_df1][:5000], errors='coerce')
    

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(df1['evaluator_episodes'], df1[last_col_df1], label='SGCRL, teleport to goal', marker='o')
    
    plt.xlabel('Evaluator Episodes')
    plt.ylabel('Success@1000')
    plt.title('Success@1000 vs Actor Episodes, teleport FourRooms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    plt.savefig(f"stochastic_four_rooms_{seed}.png")  # You can change to .pdf, .svg, etc.




def plot_four_rooms(seed = 20):
        # Load both CSVs

    df1 = pd.read_csv(f"./logs/contrastive_cpc_point_FourRooms_{seed}/logs/evaluator/logs.csv")

   
    # Use the last column name from each file (handles changes in number of columns)
    last_col_df1 = df1.columns[-1]


    # Just make sure actor_episodes is numeric (important for plotting)
    df1['evaluator_episodes'] = pd.to_numeric(df1['evaluator_episodes'], errors='coerce')
    
    # Also make sure success values are numeric (sometimes parsing creates strings)
    df1[last_col_df1] = pd.to_numeric(df1[last_col_df1], errors='coerce')
    

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(df1['evaluator_episodes'], df1[last_col_df1], label='SGCRL', marker='o')
    
    plt.xlabel('Evaluator Episodes')
    plt.ylabel('Success@1000')
    plt.title('Success@1000 vs Actor Episodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    plt.savefig(f"four_rooms_{seed}.png")  # You can change to .pdf, .svg, etc.



#!/usr/bin/env python
# file: inspect_counter.py
#
# Usage:
#   python inspect_counter.py /home/mahsa/sgcrl/logs/contrastive_cpc_stochastic_point_FourRooms_70/checkpoints/counter 2
#
# Prints the episode / evaluator counters stored in ckpt-2.*

import sys
from pathlib import Path
import tensorflow as tf   # works with TF‑V2 checkpoints (Acme / JAX uses this format)

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


#plot_four_rooms(seed= 20)
plot_stochastic_four_rooms(seed = 71)
#seed = 70
#inspect_counter("/home/mahsa/sgcrl/logs/contrastive_cpc_stochastic_point_FourRooms_70/checkpoints/counter", 2)
#show_counter_state(f"/home/mahsa/sgcrl/logs/contrastive_cpc_stochastic_point_FourRooms_{seed}/checkpoints/counter/ckpt-40")