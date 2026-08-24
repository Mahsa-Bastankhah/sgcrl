"""
New CRL: expected InfoNCE with class-marginal * unweighted partition.

  L = - mean_i sum_j P_ij log(
          exp(s_ij) / (exp(s_ij) + (B-1) p^marg_j * sum_l exp(s_il))
      )

Readout: p_W = softmax(x W)  (unlike old CRL's prior-weighted softmax).
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
torch.set_default_dtype(torch.float64)

n_train = 10
d = 15
k = 5

noise_std = 0.2
lr = 1e-2
grad_tol = 1e-8
max_steps = 100_000
crl_Bs = (5, 15, 50, 1000)

device = "cpu"
print(f"device={device}")


def teacher(x: torch.Tensor, W_star: torch.Tensor, eps: Optional[torch.Tensor] = None):
    logits = x @ W_star
    if eps is not None:
        logits = logits + eps
    probs = F.softmax(logits, dim=-1)
    return logits, probs


def new_crl_loss(logits: torch.Tensor, P: torch.Tensor, B: int) -> torch.Tensor:
    """New CRL: negatives = (B-1) * p_j * sum_l exp(s_il).

    L = - mean_i sum_j P_ij log(
            exp(s_ij) / (exp(s_ij) + (B-1) p_j * sum_l exp(s_il))
        )
    """
    assert B >= 2, f"CRL needs B>=2 for B-1 negatives, got B={B}"
    p_marg = P.mean(dim=0)  # (k,)  — p_j
    log_Z = torch.logsumexp(logits, dim=-1, keepdim=True)  # log sum_l exp(s_il)
    # log((B-1) p_j sum_l exp) = log(B-1) + log(p_j) + log_Z
    log_denom = torch.logaddexp(
        logits,
        math.log(B - 1) + torch.log(p_marg + 1e-12) + log_Z,
    )
    log_frac = logits - log_denom
    return -(P * log_frac).sum(dim=-1).mean()


def kl_mean(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum(-1).mean()


def report(name: str, W: torch.Tensor, X: torch.Tensor, P_train: torch.Tensor) -> None:
    Wd = W.detach()
    p_hat = F.softmax(X @ Wd, dim=-1)  # new CRL: p_W = softmax(xW)
    print(f"\n=== {name} ===")
    print(f"||W||_F = {Wd.norm().item():.6f}")
    print(f"KL(P || p_W) = {kl_mean(P_train, p_hat).item():.6e}   "
          f"[p_W = softmax(xW)]")
    print("W =")
    print(Wd.cpu())


def train_until(name, W, loss_fn, lr, grad_tol, max_steps):
    opt = torch.optim.SGD([W], lr=lr)
    loss = gnorm = None
    for step in range(max_steps):
        opt.zero_grad()
        loss = loss_fn(W)
        loss.backward()
        gnorm = W.grad.detach().norm().item()
        if step % 5000 == 0 or gnorm < grad_tol:
            print(
                f"{name:16s}  step {step:6d}  loss={loss.item():.6e}  "
                f"||grad||={gnorm:.3e}  ||W||={W.detach().norm().item():.4f}"
            )
        if gnorm < grad_tol:
            return step, loss.item(), gnorm
        opt.step()
    print(
        f"{name:16s}  DID NOT vanish by {max_steps}  loss={loss.item():.6e}  "
        f"||grad||={gnorm:.3e}"
    )
    return max_steps, loss.item(), gnorm


W_star = torch.randn(d, k, device=device) / d**0.5
X_train = torch.randn(n_train, d, device=device)
eps_train = noise_std * torch.randn(n_train, k, device=device) if noise_std > 0 else None
_, P_train = teacher(X_train, W_star, eps_train)

W_new = {
    B: torch.zeros(d, k, device=device, requires_grad=True) for B in crl_Bs
}

print(f"new CRL only  until ||grad||_F < {grad_tol:g}  (max_steps={max_steps})")
print(f"B values: {crl_Bs}")
print("loss denom: exp(s_ij) + (B-1) * p_j * sum_l exp(s_il)")
print("eval p_W = softmax(x W)")

stops = {}
for B in crl_Bs:
    stops[B] = train_until(
        f"newCRL B={B}", W_new[B],
        lambda W, B=B: new_crl_loss(X_train @ W, P_train, B=B),
        lr, grad_tol, max_steps,
    )

print("\n=== steps to ||grad|| vanish ===")
for B in crl_Bs:
    st, ls, g = stops[B]
    print(f"newCRL B={B:<4d} steps={st}  loss={ls:.6e}  ||grad||={g:.3e}")

for B in crl_Bs:
    report(f"newCRL B={B}", W_new[B], X_train, P_train)

with torch.no_grad():
    print("\n=== pairwise ||W(B) - W(B')||_F ===")
    Bs = list(crl_Bs)
    for i, B1 in enumerate(Bs):
        for B2 in Bs[i + 1:]:
            print(
                f"B={B1} vs B={B2}  "
                f"{(W_new[B1] - W_new[B2]).norm().item():.6f}"
            )
