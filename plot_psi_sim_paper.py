#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def fourrooms_map():
    point_map = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    ])
    axes_lims = {0: (0, 11), 1: (0, 11)}
    ct = 11
    return point_map, axes_lims, ct

def build_npz_path(env_name, seed, ckpt, action_mode, project, uid):
    base_dir = os.path.join("experiments/psi_data", f"{env_name}_{seed}")
    if uid is not None:
        base_dir = os.path.join(base_dir, uid)
    base = "psi_projected" if project else f"psi_{action_mode}"
    return os.path.join(base_dir, f"{base}_{ckpt}.npz"), base

def plot_one(ax, data, point_map, axes_lims, title=None):
    max_y = point_map.shape[0]  # 11 for FourRooms
    goal_locations = data["goal_locations"]
    psi_similarity = data["psi_similarity"]
    pos_arr       = data["pos_arr"]
    starts_arr    = data["starts_arr"]
    goals_arr     = data["goals_arr"]
    episode_length = int(data["episode_length"])
    num_episodes   = int(data["num_episodes"])
    original_start = pos_arr[0]  # first step of first rollout

    # Plot map
    ax.imshow(point_map, cmap='binary', extent=[0, 11, 0, 11])

    # Heatmap (NO per-axes colorbar here)
    vmin, vmax = float(np.min(psi_similarity)), float(np.max(psi_similarity))
    norm = plt.Normalize(vmin, vmax)
    heat = ax.scatter(
        goal_locations[:, 1], max_y - goal_locations[:, 0],
        c=psi_similarity, s=10, alpha=0.2,
        cmap='RdYlGn_r', norm=norm
    )

    # Trajectories
    for epi in range(num_episodes):
        sl = slice(epi * episode_length, (epi + 1) * episode_length)
        ax.plot(pos_arr[sl, 1], max_y - pos_arr[sl, 0], alpha=0.7)

    # Markers
    # start_scatter = ax.scatter(
    #     starts_arr[0, 1], max_y - starts_arr[0, 0],
    #     c='red', marker='D', s=80, edgecolors='black', linewidth=2, label='Perturbed Start'
    # )
    goal_scatter = ax.scatter(
        goals_arr[0, 1], max_y - goals_arr[0, 0],
        c='red', marker='*', s=300, edgecolors='black', linewidth=2, label='Goal'
    )
    original_scatter = ax.scatter(
        original_start[1], max_y - original_start[0],
        c='lime', marker='o', s=200, edgecolors='black', linewidth=2, label='Start'
    )

    # Axis aesthetics (no axis-level titles)
    ax.set_xlim(axes_lims[0])
    ax.set_ylim(axes_lims[1])
    ax.set_xticks([])
    ax.set_yticks([])

    # Return heat + local range so the caller can build a shared colorbar
    return [goal_scatter, original_scatter], heat, vmin, vmax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[220, 331])
    parser.add_argument("--ckpt", type=int, default=7)
    parser.add_argument("--action_mode", type=str, default="actor_sample",
                        choices=["actor_max", "actor_sample", "q_max"])
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    env_name = "point_FourRooms"
    point_map, axes_lims, _ = fourrooms_map()

    # Figure with 1x2 panels
    #fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    # OLD
    # fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # NEW — reserve a 3rd, skinny column for the colorbar
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.00)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])  # dedicated colorbar axis
    axes = [ax0, ax1]


    all_handles = None

    heats = []
    vmins, vmaxs = [], []

    # Plot each seed in its own panel (no local colorbars, no local titles)
    for i, seed in enumerate(args.seeds[:2]):  # ensure two panels
        npz_path, base = build_npz_path(env_name, seed, args.ckpt,
                                        args.action_mode, args.project, args.uid)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Could not find NPZ at {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        handles, heat, vmin, vmax = plot_one(axes[i], data, point_map, axes_lims)
        heats.append(heat)
        vmins.append(vmin); vmaxs.append(vmax)
        if all_handles is None:
            all_handles = handles  # use handles from first panel

    # Shared color scale + ONE shared colorbar
    vmin_global, vmax_global = float(np.min(vmins)), float(np.max(vmaxs))
    shared_norm = plt.Normalize(vmin_global, vmax_global)
    for h in heats:
        h.set_norm(shared_norm)

    # cbar = fig.colorbar(heats[0], ax=axes, orientation="vertical",
    #                     fraction=0.046, pad=1)
    # cbar.ax.tick_params(labelsize=18)
    # cbar.set_label(r"$\psi$ Similarity", fontsize=24, labelpad=15)

    # OLD
    # cbar = fig.colorbar(heats[0], ax=axes, orientation="vertical", fraction=0.046, pad=0.04)

    # NEW — draw into the reserved axis so it never overlaps
    cbar = fig.colorbar(heats[0], cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(r"$\psi$ Similarity", fontsize=24, labelpad=15)


    # (Removed suptitle to avoid duplicate "psi similarity" text)
    # fig.suptitle(r"$\psi$ Similarity (shared scale)", fontsize=20)

    # Legend at bottom center (unchanged)
    fig.subplots_adjust(bottom=0.18, wspace=0.00)
    fig.legend(
        handles=all_handles,
        labels=['Goal', 'Start'],
        loc='lower center', ncol=2, fontsize=18
    )

    # Save one PDF
    out_dir = args.out_dir or os.path.join("experiments/plots", f"{env_name}_multi")
    os.makedirs(out_dir, exist_ok=True)

    mode_tag = "psi_projected" if args.project else f"psi_{args.action_mode}"
    out_path = os.path.join(out_dir, f"{mode_tag}_seeds_{args.seeds[0]}_{args.seeds[1]}_ckpt_{args.ckpt}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined PDF to: {out_path}")

if __name__ == "__main__":
    main()
