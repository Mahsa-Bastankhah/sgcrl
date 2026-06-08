"""Read PPO training CSVs and save mean ± std shaded plots to figs/.

Usage:
    python save_training_figs.py
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

def _fmt_steps(v, _):
    """Format global step counts as e.g. '0', '500K', '1M', '2.5M'."""
    if v == 0:
        return "0"
    if v >= 1e6:
        s = f"{v/1e6:.1f}M"
        return s.replace(".0M", "M")
    if v >= 1e3:
        s = f"{v/1e3:.0f}K"
        return s
    return str(int(v))

LOG_BASE = "/n/fs/mislresearch/sgcrl/logs/ppo"
LOG_BASES = {
    "btn_noanneal":        "/n/fs/mislresearch/sgcrl/logs/ppo_btn_noanneal",
    "btn_noanneal_lowent": "/n/fs/mislresearch/sgcrl/logs/ppo_btn_noanneal_lowent",
}
FIGS_DIR = "/n/fs/mislresearch/sgcrl/figs"
os.makedirs(FIGS_DIR, exist_ok=True)

FLOW_HORIZON = 1500

ENVS = {
    "sawyer_push":              dict(label="Sawyer Push",              metric="success_1000", ylabel="Success ×1000"),
    "sawyer_drawer_open":       dict(label="Sawyer Drawer Open",       metric="success_1000", ylabel="Success ×1000"),
    "sawyer_button_press":      dict(label="Sawyer Button Press",      metric="success_1000", ylabel="Success ×1000"),
    "flow_figureeight":         dict(label="Flow Figure-Eight (1/14)", metric="ep_flow_dense_return_mean", ylabel="Episode Return",  log="learner"),
    "flow_figureeight_7rl":     dict(label="Flow Figure-Eight (7/14)", metric="ep_flow_dense_return_mean", ylabel="Episode Return",  log="learner"),
    "flow_figureeight_14rl":    dict(label="Flow Figure-Eight (14/14)",metric="ep_flow_dense_return_mean", ylabel="Episode Return",  log="learner"),
    "flow_fe7rl_success":       dict(label="Flow 7/14 – Success Rate",  env_dir="flow_figureeight_7rl",  metric="ep_return_mean", scale=1/FLOW_HORIZON, ylabel="Success Rate", log="learner"),
    "flow_fe14rl_success":      dict(label="Flow 14/14 – Success Rate", env_dir="flow_figureeight_14rl", metric="ep_return_mean", scale=1/FLOW_HORIZON, ylabel="Success Rate", log="learner"),
    "flow_figureeight_2v1rl":   dict(label="Flow 2-Car (1/2 RL)",       metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    "flow_figureeight_2v2rl":   dict(label="Flow 2-Car (2/2 RL)",       metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    "flow_figureeight_1v1rl":   dict(label="Flow 1-Car Sanity (1/1 RL)", metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    # scaling experiments
    "flow_figureeight_4v2rl":   dict(label="Flow 4-Car (2/4 RL)",         metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    "flow_figureeight_8v4rl":   dict(label="Flow 8-Car (4/8 RL)",         metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    # button-press ablations
    "btn_noanneal":        dict(label="Button (no LR sched, ent=0.05)",
                                env_dir="sawyer_button_press",
                                metric="success_1000", ylabel="Success ×1000",
                                log_base="btn_noanneal"),
    "btn_noanneal_lowent": dict(label="Button (no LR sched, ent=0.01)",
                                env_dir="sawyer_button_press",
                                metric="success_1000", ylabel="Success ×1000",
                                log_base="btn_noanneal_lowent"),
}

ACCENT_COLORS = ["#4C9BE8", "#E8834C", "#4CE87A", "#E84C6F", "#A84CE8", "#E8D44C", "#4CE8D4", "#E84CA8"]


def read_csv(env, seed, log_type="eval", log_base=None):
    base = log_base if log_base else LOG_BASE
    path = os.path.join(base, f"ppo_{env}_{seed}", "logs", log_type, "logs.csv")
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def coerce(v):
    try:
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return None


def extract(rows, x_col, y_col, scale=1.0):
    pts = []
    for r in rows:
        x = coerce(r.get(x_col, ""))
        y = coerce(r.get(y_col, ""))
        if x is not None and y is not None:
            pts.append((int(x), y * scale))
    return pts


def subsample(pts, max_pts=200):
    if len(pts) <= max_pts:
        return pts
    step = max(1, len(pts) // max_pts)
    sampled = pts[::step]
    if pts[-1] not in sampled:
        sampled = sampled + [pts[-1]]
    return sampled


# ── plot each env ────────────────────────────────────────────────────────────

for i, (env_key, cfg) in enumerate(ENVS.items()):
    log_type     = cfg.get("log", "eval")
    env_dir      = cfg.get("env_dir", env_key)
    scale        = cfg.get("scale", 1.0)
    log_base_key = cfg.get("log_base")
    log_base_dir = LOG_BASES.get(log_base_key) if log_base_key else None
    x_col        = "learner_steps"

    seed_xs, seed_ys = {}, {}
    for s in range(3):
        rows = read_csv(env_dir, s, log_type, log_base=log_base_dir)
        pts  = extract(rows, x_col, cfg["metric"], scale)
        pts  = subsample(pts, max_pts=200)
        if pts:
            seed_xs[s] = [p[0] for p in pts]
            seed_ys[s] = [p[1] for p in pts]

    if not seed_xs:
        print(f"  {env_key}: no data, skipping")
        continue

    # Interpolate all seeds onto a common x grid
    all_x = sorted({x for xs in seed_xs.values() for x in xs})
    interp_ys = []
    for s in seed_xs:
        iy = np.interp(all_x, seed_xs[s], seed_ys[s])
        interp_ys.append(iy)
    interp_ys = np.array(interp_ys)  # (n_seeds, n_x)

    mean = interp_ys.mean(axis=0)
    std  = interp_ys.std(axis=0) if interp_ys.shape[0] > 1 else np.zeros_like(mean)

    x_m = np.array(all_x)
    color = ACCENT_COLORS[i % len(ACCENT_COLORS)]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(x_m, np.maximum(0, mean - std), mean + std,
                    alpha=0.25, color=color, linewidth=0)
    ax.plot(x_m, mean, color=color, linewidth=1.8, label="mean ± std")

    ax.set_xlabel("Global Steps", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
    ax.set_ylabel(cfg["ylabel"], fontsize=11)
    ax.set_title(cfg["label"], fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    patch = mpatches.Patch(color=color, alpha=0.4, label=f"±1 std  (n={len(seed_xs)} seeds)")
    ax.legend(handles=[patch], fontsize=9, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(FIGS_DIR, f"{env_key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  ({len(seed_xs)} seeds, {len(all_x)} x-pts)")

# ── combined figure (all panels in one image) ────────────────────────────────
envs_with_data = []
for env_key, cfg in ENVS.items():
    log_type = cfg.get("log", "eval")
    env_dir  = cfg.get("env_dir", env_key)
    scale    = cfg.get("scale", 1.0)
    seed_xs  = {}
    for s in range(3):
        rows = read_csv(env_dir, s, log_type)
        pts  = subsample(extract(rows, "learner_steps", cfg["metric"], scale), 200)
        if pts:
            seed_xs[s] = pts
    if seed_xs:
        envs_with_data.append((env_key, cfg, seed_xs))

n = len(envs_with_data)
ncols = 2
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows))
axes = np.array(axes).flatten()

for idx, (env_key, cfg, seed_xs_pts) in enumerate(envs_with_data):
    ax = axes[idx]
    scale = cfg.get("scale", 1.0)
    color = ACCENT_COLORS[idx % len(ACCENT_COLORS)]

    seed_xs_arr, seed_ys_arr = {}, {}
    for s, pts in seed_xs_pts.items():
        seed_xs_arr[s] = [p[0] for p in pts]
        seed_ys_arr[s] = [p[1] for p in pts]

    all_x = sorted({x for xs in seed_xs_arr.values() for x in xs})
    interp_ys = np.array([np.interp(all_x, seed_xs_arr[s], seed_ys_arr[s]) for s in seed_xs_arr])
    mean = interp_ys.mean(axis=0)
    std  = interp_ys.std(axis=0) if interp_ys.shape[0] > 1 else np.zeros_like(mean)
    x_m  = np.array(all_x)

    ax.fill_between(x_m, np.maximum(0, mean - std), mean + std,
                    alpha=0.25, color=color, linewidth=0)
    ax.plot(x_m, mean, color=color, linewidth=1.8)
    ax.set_title(cfg["label"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Global Steps", fontsize=8)
    ax.set_ylabel(cfg["ylabel"], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))

for ax in axes[len(envs_with_data):]:
    ax.set_visible(False)

fig.suptitle("PPO Training — mean ± std across seeds", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
combined_path = os.path.join(FIGS_DIR, "all_envs.png")
fig.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Combined figure saved: {combined_path}")
