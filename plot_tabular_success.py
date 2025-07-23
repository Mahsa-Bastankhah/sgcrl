import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

RUNS_DIR = "runs_many"
WINDOW = 1  # Moving average window
GROUP_SIZE = 10  # Group size for plotting

def moving_average(x, w=WINDOW):
    if len(x) < w:
        return np.full_like(x, np.mean(x))  # fallback: flat average if too short
    return np.convolve(x, np.ones(w) / w, mode='valid')


# UID → list of success curves across seeds
train_curves = defaultdict(list)
eval_curves  = defaultdict(list)
config_map   = {}

print("🔍 Scanning run directories...")

# --- Step 1: Load and organize data ---
for seed_dir in Path(RUNS_DIR).glob("seed*/"):
    for run_dir in seed_dir.iterdir():
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        train_path = run_dir / "success_list.npy"
        eval_path = run_dir / "eval_success_list.npy"

        if not (config_path.exists() and train_path.exists() and eval_path.exists()):
            print(f"⚠️ Missing files in {run_dir}. Skipping.")
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            uid = config["uid"]

            train_curve = np.load(train_path)
            
            eval_curve  = np.load(eval_path)

            if uid not in config_map:
                config_map[uid] = {
                    k: config[k] for k in ["rep_dim", "lr_phi_psi", "alpha"]
                }
            if uid == "0bc6d1b448":
                print("🔍 Found config for UID 0bc6d1b448: train ")
                print(train_curve)
                print("eval ")
                print(eval_curve)

            min_len = min(len(train_curve), len(eval_curve))
            train_curves[uid].append(train_curve[:min_len])
            eval_curves[uid].append(eval_curve[:min_len])

        except Exception as e:
            print(f"❌ Error loading from {run_dir}: {e}")

# --- Group and Plot in chunks of GROUP_SIZE ---
uids = list(config_map.keys())
total_groups = (len(uids) + GROUP_SIZE - 1) // GROUP_SIZE

print(f"\n📊 Total configs: {len(uids)}. Plotting in {total_groups} groups of {GROUP_SIZE}...")

for g_idx in range(total_groups):
    group_uids = uids[g_idx * GROUP_SIZE : (g_idx + 1) * GROUP_SIZE]
    print(f"\n➡️ Plotting group {g_idx + 1}/{total_groups}: UIDs {group_uids}")

    # --- Train plot ---
    plt.figure(figsize=(10, 5))
    for uid in group_uids:
        curves = train_curves.get(uid, [])
        if not curves:
            print(f"⚠️ No training data for UID {uid}")
            continue
        smoothed = [moving_average(c) for c in curves]
        min_len = min(len(s) for s in smoothed)
        avg_curve = np.mean([s[:min_len] for s in smoothed], axis=0)
        label = f"{uid} " + str(config_map[uid])
        plt.plot(avg_curve, label=label, linewidth=2)

    plt.xlabel("plot point (after smoothing)")
    plt.ylabel("train success rate (100-step avg)")
    plt.title(f"Train Success (Group {g_idx + 1})")
    plt.ylim(0, 1.05)
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    fname = f"train_success_group{g_idx + 1}"
    plt.savefig(f"{fname}.png", dpi=150)
    plt.close()

    # --- Eval plot ---
    plt.figure(figsize=(10, 5))
    for uid in group_uids:
        curves = eval_curves.get(uid, [])
        if not curves:
            print(f"⚠️ No evaluation data for UID {uid}")
            continue
        smoothed = [moving_average(c) for c in curves]
        min_len = min(len(s) for s in smoothed)
        avg_curve = np.mean([s[:min_len] for s in smoothed], axis=0)
        label = f"{uid} " + str(config_map[uid])
        plt.plot(avg_curve, label=label, linestyle="--", linewidth=2)

    plt.xlabel("plot point (after smoothing)")
    plt.ylabel("eval success rate (100-step avg)")
    plt.title(f"Eval Success (Group {g_idx + 1})")
    plt.ylim(0, 1.05)
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    fname = f"eval_success_group{g_idx + 1}"
    plt.savefig(f"{fname}.png", dpi=150)
    plt.close()

print("\n✅ Plotting complete! Check the output files for each group.")
