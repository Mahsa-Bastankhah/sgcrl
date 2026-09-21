"""||W_t - W*||_F during training. W* = that method's converged W.

Saves traces to outputs/wstar_distance_hist.pt so this does not need a rerun.
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

out_dir = Path(__file__).resolve().parent / "outputs"
ckpt_path = out_dir / "soft_teacher_ce_td.pt"
hist_path = out_dir / "wstar_distance_hist.pt"
_PAPER_STYLE = Path(__file__).resolve().parents[1] / "paper material" / "paper plot pythons"
if str(_PAPER_STYLE) not in sys.path:
    sys.path.insert(0, str(_PAPER_STYLE))


def _steps_for_log(steps):
    s = steps.detach().cpu().double()
    s[s <= 0] = 1.0
    return s.numpy()


def plot_from_saved():
    import numpy as np
    import paper_style as ps

    hist = torch.load(hist_path, weights_only=False, map_location="cpu")
    series = [
        ("CE", hist["ce_steps"], hist["ce_dists"], ps.C["blue"], "-"),
        ("CL, $B=1000$", hist["cl_steps"], hist["cl_dists"], ps.C["green"], ps.DASH),
        ("TD", hist["td_steps"], hist["td_dists"], ps.C["vermillion"], "-"),
    ]
    extra = []
    if "cl_b10_steps" in hist:
        extra.append(("CL, $B=10$", hist["cl_b10_steps"], hist["cl_b10_dists"], ps.C["orange"], (0, (2.5, 2.0))))
    if "cl_b5_steps" in hist:
        extra.append(("CL, $B=5$", hist["cl_b5_steps"], hist["cl_b5_dists"], ps.C["sky"], (0, (1.2, 1.6))))
    if "cl_b2_steps" in hist:
        extra.append(("CL, $B=2$", hist["cl_b2_steps"], hist["cl_b2_dists"], ps.C["purple"], (0, (4.0, 2.0))))
    series[2:2] = extra
    ps.apply()
    fig, ax = ps.figure("single")
    for name, steps, dists, color, ls in series:
        y = dists.cpu().numpy().copy()
        x = steps.cpu().numpy() / 1000.0
        keep = x <= 500.0
        x, y = x[keep], y[keep]
        if x[-1] < 500.0:
            x = np.concatenate([x, [x[-1], 500.0]])
            y = np.concatenate([y, [0.0, 0.0]])
        ax.plot(x, y, color=color, label=name, linewidth=ps.LW, linestyle=ls)
    ax.set_xlim(0.0, 500.0)
    ax.set_ylim(0.0, 0.58)
    ax.set_xlabel("step (k)")
    ax.set_ylabel(r"$\|W^\star - W_t\|_F$")
    ax.set_title("Distance to the learned $W$")
    ps.style_axes(ax)
    ps.nice_legend(ax, loc="upper right")
    stem = out_dir / "wstar_distance"
    ps.savefig(fig, stem, png=True)
    print(f"saved plot → {stem}.pdf")


def train_cl_B(B_cl):
    """Fit CL with a given B on the saved X, P*. Append traces to the hist file."""
    lr = 1e-2
    grad_tol = 1e-8
    log_every = 100
    max_steps = 200_000

    def cl_loss(logits, P, B):
        p_marg = P.mean(dim=0)
        log_Z = torch.logsumexp(logits, dim=-1, keepdim=True)
        log_denom = torch.logaddexp(
            logits,
            math.log(B - 1) + torch.log(p_marg + 1e-12) + log_Z,
        )
        return -(P * (logits - log_denom)).sum(dim=-1).mean()

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = ckpt["X"].to(device)
    P_star = ckpt["P_star"].to(device)
    d, k = ckpt["W_ce"].shape
    print(f"device={device}  CL B={B_cl} until ||grad||<{grad_tol:g}")

    W = torch.zeros(d, k, device=device, requires_grad=True)
    opt = torch.optim.SGD([W], lr=lr)
    snap_step, snap_W = [], []
    for step in range(max_steps):
        opt.zero_grad()
        loss = cl_loss(X @ W, P_star, B_cl)
        loss.backward()
        gnorm = W.grad.detach().norm().item()
        if step % log_every == 0 or gnorm < grad_tol:
            snap_step.append(step)
            snap_W.append(W.detach().clone())
            if step % (log_every * 50) == 0 or gnorm < grad_tol:
                print(
                    f"CL B={B_cl:<4d}  step {step:7d}  loss={loss.item():.6e}  "
                    f"||grad||={gnorm:.3e}  ||W||={W.detach().norm().item():.4f}"
                )
        if gnorm < grad_tol:
            print(f"CL B={B_cl}  stopped: ||grad|| < {grad_tol:g}")
            break
        opt.step()
    else:
        print(f"CL B={B_cl}  DID NOT hit grad_tol by {max_steps}")

    W_star = W.detach()
    dists = [(Wt - W_star).norm().item() for Wt in snap_W]
    hist = torch.load(hist_path, weights_only=False, map_location="cpu")
    hist[f"cl_b{B_cl}_steps"] = torch.tensor(snap_step)
    hist[f"cl_b{B_cl}_dists"] = torch.tensor(dists)
    hist[f"W_star_cl_b{B_cl}"] = W_star.cpu()
    torch.save(hist, hist_path)
    if "W_star_cl" in hist:
        print(
            f"||W_CL1000* - W_CL{B_cl}*||_F = "
            f"{(hist['W_star_cl'] - W_star.cpu()).norm().item():.4e}"
        )
    print(f"saved traces → {hist_path}")


def train_cl_b10():
    train_cl_B(10)


if __name__ == "__main__" and "plot-only" in sys.argv:
    plot_from_saved()
    raise SystemExit

if __name__ == "__main__" and "cl-b10" in sys.argv:
    train_cl_B(10)
    plot_from_saved()
    raise SystemExit

if __name__ == "__main__" and "cl-b5" in sys.argv:
    train_cl_B(5)
    plot_from_saved()
    raise SystemExit

if __name__ == "__main__" and "cl-b2" in sys.argv:
    train_cl_B(2)
    plot_from_saved()
    raise SystemExit
B = 1000
lr = 1e-2
grad_tol = 1e-8
log_every = 100
max_steps_ce = 200_000
max_steps_cl = 200_000
max_steps_td = 1_500_000
w_tol = 1e-3

ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
W_star_ce = ckpt["W_ce"].to(device)
W_star_td = ckpt["W_td"].to(device)
X = ckpt["X"].to(device)
S = ckpt["S"].to(device)
P_star = ckpt["P_star"].to(device)
gamma = float(ckpt["gamma"])
d, k = W_star_ce.shape
print(f"device={device}  logging ||W_t-W*||  → {hist_path}")


def ce_loss(logits, P):
    return -(P * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def td_loss(logits, S, gamma):
    probs = F.softmax(logits, dim=-1)
    target = (1.0 - gamma) * S[:-1] + gamma * probs[1:].detach()
    return ((probs[:-1] - target) ** 2).mean() + ((probs[-1] - S[-1]) ** 2).mean()


def cl_loss(logits, P, B):
    p_marg = P.mean(dim=0)
    log_Z = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_denom = torch.logaddexp(
        logits,
        math.log(B - 1) + torch.log(p_marg + 1e-12) + log_Z,
    )
    return -(P * (logits - log_denom)).sum(dim=-1).mean()


def save_hist(payload):
    cpu = {}
    for k, v in payload.items():
        cpu[k] = v.detach().cpu() if torch.is_tensor(v) else v
    torch.save(cpu, hist_path)


def train_to_ref(name, loss_fn, W_ref, max_steps, stop_on_grad):
    W = torch.zeros_like(W_ref, requires_grad=True)
    opt = torch.optim.SGD([W], lr=lr)
    steps, dists = [], []
    for step in range(max_steps):
        opt.zero_grad()
        loss = loss_fn(W)
        loss.backward()
        gnorm = W.grad.detach().norm().item()
        dist = (W.detach() - W_ref).norm().item()
        if step % log_every == 0 or (stop_on_grad and gnorm < grad_tol) or dist < w_tol:
            steps.append(step)
            dists.append(dist)
            if name == "TD" and step > 0 and step % 50000 == 0:
                save_hist({
                    "ce_steps": torch.tensor(ce_steps),
                    "ce_dists": torch.tensor(ce_dists),
                    "cl_steps": torch.tensor(cl_steps),
                    "cl_dists": torch.tensor(cl_dists),
                    "W_star_cl": W_star_cl,
                    "td_steps": torch.tensor(steps),
                    "td_dists": torch.tensor(dists),
                })
            if step % (log_every * 50) == 0 or (stop_on_grad and gnorm < grad_tol) or dist < w_tol:
                print(
                    f"{name:8s}  step {step:7d}  ||W*-W||={dist:.4e}  "
                    f"||grad||={gnorm:.3e}"
                )
        if stop_on_grad and gnorm < grad_tol:
            print(f"{name:8s}  stopped: ||grad|| < {grad_tol:g}")
            return steps, dists, W.detach()
        if dist < w_tol:
            print(f"{name:8s}  stopped: ||W*-W|| < {w_tol:g}")
            return steps, dists, W.detach()
        opt.step()
    print(f"{name:8s}  max_steps  ||W*-W||={dists[-1]:.4e}")
    return steps, dists, W.detach()


print("\n--- CE ---")
ce_steps, ce_dists, _ = train_to_ref(
    "CE", lambda W: ce_loss(X @ W, P_star), W_star_ce, max_steps_ce, True
)

print("\n--- CL ---")
W_cl = torch.zeros(d, k, device=device, requires_grad=True)
opt = torch.optim.SGD([W_cl], lr=lr)
cl_snap_step, cl_snap_W = [], []
for step in range(max_steps_cl):
    opt.zero_grad()
    loss = cl_loss(X @ W_cl, P_star, B)
    loss.backward()
    gnorm = W_cl.grad.detach().norm().item()
    if step % log_every == 0 or gnorm < grad_tol:
        cl_snap_step.append(step)
        cl_snap_W.append(W_cl.detach().clone())
        if step % (log_every * 50) == 0 or gnorm < grad_tol:
            print(f"CL        step {step:7d}  ||grad||={gnorm:.3e}")
    if gnorm < grad_tol:
        print(f"CL        stopped: ||grad|| < {grad_tol:g}")
        break
    opt.step()
W_star_cl = W_cl.detach()
cl_steps = cl_snap_step
cl_dists = [(W - W_star_cl).norm().item() for W in cl_snap_W]

save_hist({
    "ce_steps": torch.tensor(ce_steps),
    "ce_dists": torch.tensor(ce_dists),
    "cl_steps": torch.tensor(cl_steps),
    "cl_dists": torch.tensor(cl_dists),
    "W_star_cl": W_star_cl,
    "td_steps": torch.tensor([]),
    "td_dists": torch.tensor([]),
})
print(f"saved CE/CL traces → {hist_path}")

print("\n--- TD ---")
td_steps, td_dists, _ = train_to_ref(
    "TD", lambda W: td_loss(X @ W, S, gamma), W_star_td, max_steps_td, False
)

save_hist({
    "ce_steps": torch.tensor(ce_steps),
    "ce_dists": torch.tensor(ce_dists),
    "cl_steps": torch.tensor(cl_steps),
    "cl_dists": torch.tensor(cl_dists),
    "W_star_cl": W_star_cl,
    "td_steps": torch.tensor(td_steps),
    "td_dists": torch.tensor(td_dists),
})

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
for ax, name, xs, ys in (
    (axes[0], "CE", ce_steps, ce_dists),
    (axes[1], "CL", cl_steps, cl_dists),
    (axes[2], "TD", td_steps, td_dists),
):
    ax.plot(xs, ys)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|W^\star - W_t\|_F$")
    ax.set_title(name)
fig.suptitle(r"$W^\star$ = that method's converged $W$" + "    N=100, d=200", y=1.03)
fig.tight_layout()
plot_path = out_dir / "wstar_distance.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"saved plot → {plot_path}")
print(f"saved traces → {hist_path}")
