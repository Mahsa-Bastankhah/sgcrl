#!/usr/bin/env python3
"""
replot_goal_aligned_match.py

Load *_goal_aligned.npz (preferred) or *_goal_aligned.csv from a run folder
and recreate the 3D goal-aligned PCA scatter with the same plotting settings
as the provided `plot_goal_aligned_3d` function.

Adjust RUN_DIR if necessary.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------- Settings that match your function ----------------
#the main figure
RUN_DIR = "runs_final/seed204/31eb61d99e/"   # change if needed
ep_num =120000 
legend = True if ep_num == 40000 else False  # only show legend for episode 40000



# data collection comparision, random policy
#RUN_DIR = "runs_final/seed205/d5ddebdafc/"   # change if needed


# data collection comparision, sgcrl policy
# RUN_DIR = "runs_final/seed205/a25099a065/"   # change if needed

# # specify episode number if needed, or leave as None to use the latest
# ep_num = "09000" # specify episode number if needed, or leave as None to use the latest

# legend = False


OUT_NAME = f"replot_goal_aligned_{ep_num}.pdf"
FIG_DPI = 300
# rcParams per your snippet (no LaTeX)
mpl.rcParams.update({
    "text.usetex": False,
    # "font.family": "serif",
    # "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 26,
    "axes.titlesize": 26,
    "legend.fontsize": 26,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "figure.dpi": FIG_DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Plot params (copied from your function)
FIGSIZE = (12, 10)
DOT_SIZE = 150
RASTERIZE_THRESHOLD = 30000
ALPHA = 0.95
EDGECOLOR = 'k'
EDGE_LINEWIDTH = 0.1
WIRE_U_RES = 60
WIRE_V_RES = 30
GOAL_MARKER_S = 200
TITLE = "3D PCA plot of ψ(s) - FourRooms - Episode {}".format(ep_num)
VIEW_ELEV = 25
VIEW_AZIM = -60
import os
import glob
import re

def find_projection_file(run_dir, ep_num=None):
    """
    Find a *_goal_aligned.npz (preferred) or *_goal_aligned.csv file.
    If ep_num is provided (int or str), prefer a file containing '_{ep_num}_goal_aligned'.
    Returns (path, fmt) where fmt is 'npz' or 'csv'.
    Raises FileNotFoundError if nothing suitable is found.
    """
    # normalize ep_num to string (no spaces)
    ep_str = None if ep_num is None else str(ep_num).strip()

    # helper to search by exact episode token
    def _glob_for(pattern):
        return sorted(glob.glob(os.path.join(run_dir, pattern)))

    # 1) If user specified an episode, try strict matches first
    if ep_str is not None:
        # prefer npz
        npz_pat = f"*_{ep_str}_goal_aligned.npz"
        npz_matches = _glob_for(npz_pat)
        if npz_matches:
            return npz_matches[-1], "npz"
        # fallback to csv with same ep
        csv_pat = f"*_{ep_str}_goal_aligned.csv"
        csv_matches = _glob_for(csv_pat)
        if csv_matches:
            return csv_matches[-1], "csv"
        # try a looser match (in case filename doesn't have underscores exactly)
        npz_loose = _glob_for(f"*{ep_str}*goal_aligned.npz")
        if npz_loose:
            return npz_loose[-1], "npz"
        csv_loose = _glob_for(f"*{ep_str}*goal_aligned.csv")
        if csv_loose:
            return csv_loose[-1], "csv"

        # nothing matched the requested ep_num — raise informative error listing available eps
        # collect all available goal_aligned files and extract episode-like tokens
        all_files = _glob_for("*_goal_aligned.npz") + _glob_for("*_goal_aligned.csv")
        available_eps = set()
        for p in all_files:
            name = os.path.basename(p)
            # extract numbers in filename that are plausible episode tokens
            for m in re.finditer(r"(\d{3,})", name):
                available_eps.add(m.group(1))
        available_sorted = sorted(available_eps, key=lambda x: int(x)) if available_eps else []
        raise FileNotFoundError(
            f"No goal_aligned file for episode '{ep_str}' found in {run_dir}.\n"
            f"Available episode-like tokens found in filenames: {available_sorted}\n"
            "Make sure you passed the correct episode number or inspect the folder."
        )

    # 2) No ep_num specified: fallback to previous behavior (latest file)
    npz_files = sorted(glob.glob(os.path.join(run_dir, "*_goal_aligned.npz")))
    csv_files = sorted(glob.glob(os.path.join(run_dir, "*_goal_aligned.csv")))
    if npz_files:
        return npz_files[-1], "npz"
    if csv_files:
        return csv_files[-1], "csv"
    raise FileNotFoundError(f"No *_goal_aligned.npz or *_goal_aligned.csv in {run_dir}")

def load_projection(path, fmt):
    if fmt == "npz":
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        Y = data["Y"]
        Z = data["Z"]
        c = data["c"]
        room = data["room"].astype(str) if "room" in data else None
        i = data["i"] if "i" in data else None
        j = data["j"] if "j" in data else None
        u1 = data["u1"] if "u1" in data else None
        u2 = data["u2"] if "u2" in data else None
        u3 = data["u3"] if "u3" in data else None
        return {"X": X, "Y": Y, "Z": Z, "c": c, "room": room, "i": i, "j": j, "u1": u1, "u2": u2, "u3": u3}
    else:
        df = pd.read_csv(path)
        X = df["X"].values
        Y = df["Y"].values
        Z = df["Z"].values
        c = df["c"].values
        room = df["room"].values if "room" in df.columns else None
        return {"X": X, "Y": Y, "Z": Z, "c": c, "room": room, "i": None, "j": None, "u1": None, "u2": None, "u3": None}

def reconstruct_rooms_if_needed(proj):
    if proj["room"] is not None:
        return proj["room"].astype(str)
    # try reconstruct using i,j if present
    if (proj["i"] is not None) and (proj["j"] is not None):
        i = np.asarray(proj["i"]).astype(int)
        j = np.asarray(proj["j"]).astype(int)
        mid_i = int(np.median(i))
        mid_j = int(np.median(j))
        room = np.array(["None"] * len(i), dtype=object)
        room[(i < mid_i) & (j < mid_j)] = "Bottom-Left"
        room[(i < mid_i) & (j >= mid_j)] = "Bottom-Right"
        room[(i >= mid_i) & (j < mid_j)] = "Top-Left"
        room[(i >= mid_i) & (j >= mid_j)] = "Top-Right"
        return room
    # fallback: all "None"
    N = proj["X"].shape[0]
    return np.array(["None"] * N, dtype=object)

def main():
    data_path, fmt = find_projection_file(RUN_DIR, ep_num)
    print(f"[info] loading {fmt} file: {data_path}")
    proj = load_projection(data_path, fmt)

    X = np.asarray(proj["X"])
    Y = np.asarray(proj["Y"])
    Z = np.asarray(proj["Z"])
    c = np.asarray(proj["c"])
    room = reconstruct_rooms_if_needed(proj)
    N = X.shape[0]

    rasterize_flag = (N >= RASTERIZE_THRESHOLD)

    # room masks (keep same naming/order)
    m_TL = (room == "Bottom-Left")
    m_TR = (room == "Bottom-Right")
    m_BL = (room == "Top-Left")
    m_BR = (room == "Top-Right")

    groups = [
        ('o', 'Bottom-Left',     m_TL),
        ('^', 'Bottom-Right',    m_TR),
        ('s', 'Top-Left',        m_BL),
        ('D', 'Top-Right',       m_BR),
    ]

    # ----- plotting (match your function) -----
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=c.min() if N else 0.0, vmax=c.max() if N else 1.0)
    cmap = plt.get_cmap("viridis")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for marker, label, m in groups:
        if np.any(m):
            ax.scatter(
                X[m], Y[m], Z[m],
                c=c[m], cmap=cmap, norm=norm,
                s=DOT_SIZE,
                alpha=ALPHA,
                marker=marker,
                label=label,
                edgecolor=EDGECOLOR,
                linewidth=EDGE_LINEWIDTH,
                rasterized=rasterize_flag
            )

    # colorbar exactly as in your function
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
    cb.set_label("ψ-similarity", fontsize=24)

    # unit sphere wireframe (same resolution and styling)
    u = np.linspace(0, 2*np.pi, WIRE_U_RES)
    v = np.linspace(0, np.pi, WIRE_V_RES)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.3, alpha=0.5, zorder=0)

    # mark goal exactly the same
    # ax.scatter([0], [0], [1], c='red', s=GOAL_MARKER_S, marker='*', label='Goal (z)')
    # place the star slightly outside the unit sphere so it can't be occluded
    goal = np.array([0.0, 0.0, 1.0])
    eps = 1.12               # 0.5% outside the unit sphere (tune if you want)
    goal = goal / np.linalg.norm(goal) * eps

    ax.scatter([goal[0]], [goal[1]], [goal[2]],
            c='red',
            s=800,                 # tune bigger if you want
            marker='*',
            edgecolor='k',
            linewidths=0.9,
            zorder=1000,
            rasterized=False,
            depthshade=False)


    # title & fixed axes (exact limits you used)
    ax.set_title(TITLE, fontsize=24)
    # ax.set_xlim(-1.1, 1.1)
    # ax.set_ylim(-1.1, 1.1)
    # ax.set_zlim(-0.15, 1.1)
    # ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    # after plotting everything, before saving:
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)           # make z symmetric to match x/y
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect([1,1,1])   # equal aspect ratio in 3D
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    from matplotlib.ticker import MaxNLocator

    # set ~4 major ticks on each axis (tune 3..6)
    nbins = 6
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

    ax.zaxis.set_tick_params(pad=16)   # try 6..14
    # move the axis *label* as well
    #ax.zaxis.labelpad = 100            # controls "u1 (along goal)" distance

    ## uncomment the legend if needed

    # manually define a smaller star for the legend
    if legend:
        from matplotlib.lines import Line2D
        # grab the existing handles and labels from the scatter plots
        handles, labels = ax.get_legend_handles_labels()

        # define a custom (smaller) star just for the legend
        goal_handle = Line2D([0], [0], marker='*', color='red',
                            markersize=18, markeredgecolor='k',
                            linestyle='None', label='Goal (z)')

        # append the goal handle to the existing legend entries
        handles.append(goal_handle)
        labels.append("Goal (z)")

        leg = ax.legend(handles, labels,
                        loc="upper left",
                        markerscale=1.8,
                        handletextpad=0.6,
                        labelspacing=0.4,
                        borderpad=0.4)




    #legend exactly as your function used
    # if legend:
    #     leg = ax.legend(loc="upper left",
    #                     markerscale=2.5,
    #                     handletextpad=0.6,
    #                     labelspacing=0.4,
    #                     borderpad=0.4)

    fig.tight_layout()
    out_path = os.path.join(RUN_DIR, OUT_NAME)
    #fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.02)
    print(f"[info] saved replot to: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
