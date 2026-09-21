"""Replot already-logged CE / CL / TD losses. No training."""

from pathlib import Path

import matplotlib.pyplot as plt

# From the first completed run (seed 0, N=100, d=200). Printed every 5k steps.
ce_steps = [0, 5000, 10000]
ce_losses = [1.609438, 1.581731, 1.581713]

cl_steps = [0, 5000, 10000]
cl_losses = [6.910324, 6.882667, 6.882649]

td_steps = [0, 5000, 10000]
td_losses = [3.301726e-02, 1.634910e-04, 1.060475e-04]

out_dir = Path(__file__).resolve().parent / "outputs"
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
for ax, name, xs, ys, nstep in (
    (axes[0], "CE", ce_steps, ce_losses, 10_000),
    (axes[1], "CL", cl_steps, cl_losses, 10_000),
    (axes[2], "TD", td_steps, td_losses, 10_000),
):
    ax.plot(xs, ys, marker="o", markersize=3)
    ax.set_xlim(0, nstep)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title(f"{name}  ({nstep} steps)")
fig.suptitle("N=100, d=200", y=1.02)
fig.tight_layout()
plot_path = out_dir / "three_losses_train.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"saved plot → {plot_path}")
