import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import os


def _last_non_nan_column(df, max_nan=100):
    """Return the right‑most column that has **≤ max_nan** NaNs after coercion."""
    for col in reversed(df.columns):
        series = pd.to_numeric(df[col], errors="coerce")
        nan_count = series.isna().sum()
        if nan_count <= max_nan:
            return col
    raise ValueError("No column with acceptable NaN count found in evaluator log.")


def plot_eval_compare(
    seed_list,
    env_name,
    labels,
    mode="separate",
    success_col=None,
):
    """
    Plot success curves for multiple seeds, either separately or averaged by group.

    When *success_col* is None, the last column that is **not all NaN** (after
    numeric coercion) is chosen automatically.
    """

    data = []  # {seed, label, eps, vals}
    for seed, label in zip(seed_list, labels):
        folder = f"./logs/contrastive_cpc_point_{env_name}_{seed}/logs/evaluator"
        eval_path = os.path.join(folder, "logs.csv")
        df = pd.read_csv(eval_path)

        # pick success column
        col = success_col or _last_non_nan_column(df)
        print(seed, label)
        print("Using column", col)
        print(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # episodes
        if "actor_episodes" in df:
            eps = pd.to_numeric(df["actor_episodes"], errors="coerce").values
        else:
            actor_path = os.path.join(folder.replace("evaluator", "actor"), "logs.csv")
            actor_df = pd.read_csv(actor_path)
            eps = pd.to_numeric(actor_df["actor_episodes"], errors="coerce").values

        vals = df[col].values
        # align lengths if mismatch
        if len(eps) != len(vals):
            eps = np.interp(np.linspace(0, 1, len(vals)), np.linspace(0, 1, len(eps)), eps)

        data.append({"seed": seed, "label": label, "eps": eps, "vals": vals})

    plt.figure(figsize=(10, 6))

    if mode == "separate":
        for d in data:
            plt.plot(d["eps"], d["vals"], label=f"Seed {d['seed']}, {d['label']}")
    elif mode == "average":
        groups = {}
        for d in data:
            groups.setdefault(d["label"], []).append(d)

        for label, runs in groups.items():
            common_max = min(r["eps"].max() for r in runs)
            common_eps = np.linspace(0, common_max, 100)
            aligned = [np.interp(common_eps, r["eps"], r["vals"]) for r in runs]
            mat = np.vstack(aligned)
            mean, std = np.nanmean(mat, 0), np.nanstd(mat, 0)
            plt.plot(common_eps, mean, label=f"{label} (mean±std)")
            plt.fill_between(common_eps, mean - std, mean + std, alpha=0.3)
    else:
        raise ValueError("Mode must be 'separate' or 'average'")

    plt.xlabel("Actor Episodes")
    plt.ylabel(success_col or "Success")
    plt.title(f"{success_col or 'Success'} vs Actor Episodes ({env_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fname = f"eval_compare_{mode}_{env_name}_{'_'.join(map(str, seed_list))}.png"
    plt.savefig(fname)
    print("Saved plot to", fname)

env_name = "Maze11x11"
env_name = "Impossible"
seed_list = [13, 12 ,42,  20 , 21, 30, 31, 40, 41]
seed_list = [13, 12 ,  20 , 21, 30, 31, 40, 41]
labels = ["max Q, sgcrl", "max Q, sgcrl","actor, sgcrl", "actor, subgoal", "actor, sgcrl", "actor, subgoal", "actor, sgcrl", "actor, subgoal"]
#seed_list = [13, 12 , 20 , 21, 30, 31]
#labels = ["max Q, sgcrl",  "max Q, sgcrl","actor, sgcrl", "actor, subgoal", "actor, sgcrl", "actor, subgoal"]
plot_eval_compare(seed_list, env_name, labels, mode = 'average')





# import pandas as pd
# import os

# import pandas as pd

# seed  = 31
# env   = "Impossible"
# path  = f"./logs/contrastive_cpc_point_{env}_{seed}/logs/evaluator/logs.csv"

# df = pd.read_csv(path)

# # show dtypes first (optional)
# print("Column dtypes:\n", df.dtypes, "\n")

# # print the first 30 rows of every column
# print("First 30 rows:\n")
# print(df.head(100).to_string(index=False))

