"""CE vs TD vs CL on the same overparameterized TD-label chain.

Fixed short step counts. Same X, same P*, zero-init W, SGD.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
torch.set_default_dtype(torch.float64)

n_train = 100
d = 200
k = 5
gamma = 0.9
B = 1000

steps_ce = 10_000
steps_cl = 10_000
steps_td = 50_000
lr = 1e-2
log_every = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path(__file__).resolve().parent / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"device={device}  n={n_train}  d={d}  k={k}  gamma={gamma}  B={B}")
print(f"steps CE={steps_ce}  CL={steps_cl}  TD={steps_td}  lr={lr}")


def make_td_targets(S: torch.Tensor, gamma: float) -> torch.Tensor:
    P = torch.zeros_like(S)
    P[-1] = S[-1]
    for i in range(S.shape[0] - 2, -1, -1):
        P[i] = (1.0 - gamma) * S[i] + gamma * P[i + 1]
    return P


def ce_loss(logits: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return -(P * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def td_loss(logits: torch.Tensor, S: torch.Tensor, gamma: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    target = (1.0 - gamma) * S[:-1] + gamma * probs[1:].detach()
    td_mse = ((probs[:-1] - target) ** 2).mean()
    term_mse = ((probs[-1] - S[-1]) ** 2).mean()
    return td_mse + term_mse


def cl_loss(logits: torch.Tensor, P: torch.Tensor, B: int) -> torch.Tensor:
    p_marg = P.mean(dim=0)
    log_Z = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_denom = torch.logaddexp(
        logits,
        math.log(B - 1) + torch.log(p_marg + 1e-12) + log_Z,
    )
    return -(P * (logits - log_denom)).sum(dim=-1).mean()


def train_fixed(name, W, loss_fn, steps, lr, log_every):
    opt = torch.optim.SGD([W], lr=lr)
    hist_step = []
    hist_loss = []
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(W)
        if step % log_every == 0 or step == steps - 1:
            hist_step.append(step)
            hist_loss.append(loss.detach().item())
            if step % (log_every * 50) == 0 or step == steps - 1:
                print(
                    f"{name:8s}  step {step:7d}  loss={loss.item():.6e}  "
                    f"||W||={W.detach().norm().item():.4f}"
                )
        loss.backward()
        opt.step()
    return hist_step, hist_loss


W_star = torch.randn(d, k, device=device) / d**0.5
X = torch.randn(n_train, d, device=device)
S = F.softmax(X @ W_star, dim=-1)
P_star = make_td_targets(S, gamma)

W_ce = torch.zeros(d, k, device=device, requires_grad=True)
W_td = torch.zeros(d, k, device=device, requires_grad=True)
W_cl = torch.zeros(d, k, device=device, requires_grad=True)

print("\n--- CE ---")
ce_steps, ce_losses = train_fixed(
    "CE", W_ce, lambda W: ce_loss(X @ W, P_star), steps_ce, lr, log_every
)
print("\n--- CL ---")
cl_steps, cl_losses = train_fixed(
    "CL", W_cl, lambda W: cl_loss(X @ W, P_star, B), steps_cl, lr, log_every
)
print("\n--- TD ---")
td_steps, td_losses = train_fixed(
    "TD", W_td, lambda W: td_loss(X @ W, S, gamma), steps_td, lr, log_every
)

W_ce_d = W_ce.detach()
W_td_d = W_td.detach()
W_cl_d = W_cl.detach()

print("\n=== ||W||_F ===")
print(f"CE  {W_ce_d.norm().item():.6f}")
print(f"TD  {W_td_d.norm().item():.6f}")
print(f"CL  {W_cl_d.norm().item():.6f}")

print("\n=== pairwise ||W_a - W_b||_F ===")
print(f"CE vs TD  {(W_ce_d - W_td_d).norm().item():.6f}")
print(f"CE vs CL  {(W_ce_d - W_cl_d).norm().item():.6f}")
print(f"TD vs CL  {(W_td_d - W_cl_d).norm().item():.6f}")

with torch.no_grad():
    def tv_mean(p, q):
        return (0.5 * (p - q).abs().sum(-1).mean()).item()
    for name, W in (("CE", W_ce_d), ("CL", W_cl_d), ("TD", W_td_d)):
        q = F.softmax(X @ W, dim=-1)
        print(f"TV(P*, {name}) = {tv_mean(P_star, q):.6e}")

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
for ax, name, xs, ys, nstep in (
    (axes[0], "CE", ce_steps, ce_losses, steps_ce),
    (axes[1], "CL", cl_steps, cl_losses, steps_cl),
    (axes[2], "TD", td_steps, td_losses, steps_td),
):
    ax.plot(xs, ys)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title(f"{name}  ({nstep} steps)")
fig.suptitle(f"N={n_train}, d={d}", y=1.02)
fig.tight_layout()
plot_path = out_dir / "three_losses_train.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nsaved plot → {plot_path}")
