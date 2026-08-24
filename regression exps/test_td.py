"""
Soft-teacher immediates → TD-unrolled P* → CE vs TD.

1) S_i = softmax(x_i W_star)          # only used as immediate soft labels
2) P*_N = S_N
   P*_i = (1-gamma) S_i + gamma P*_{i+1}   # true target for BOTH CE and TV
3) CE fits P*
4) TD-stop: q_i ≈ (1-g) S_i + g sg(q_{i+1}),  q_N ≈ S_N
"""

from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
torch.set_default_dtype(torch.float64)

n_train = 100
d = 200
k = 5
gamma = 0.9

lr = 1e-2
grad_tol = 1e-8
tv_tol = 1e-3
max_steps_ce = 200_000
max_steps_td = 2_000_000
n_ood = 10

device = "cpu"
out_dir = Path(__file__).resolve().parent / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"device={device}  gamma={gamma}  n={n_train}  d={d}  k={k}")
print(f"TD stop when TV(P*, q) < {tv_tol:g}")


def tv_mean(p: torch.Tensor, q: torch.Tensor) -> float:
    return (0.5 * (p - q).abs().sum(-1).mean()).item()


def make_td_targets(S: torch.Tensor, gamma: float) -> torch.Tensor:
    """P*_N = S_N; P*_i = (1-g) S_i + g P*_{i+1}."""
    n, k = S.shape
    P = torch.zeros_like(S)
    P[-1] = S[-1]
    for i in range(n - 2, -1, -1):
        P[i] = (1.0 - gamma) * S[i] + gamma * P[i + 1]
    return P


def ce_loss(logits: torch.Tensor, P_star: torch.Tensor) -> torch.Tensor:
    return -(P_star * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def td_loss(logits: torch.Tensor, S: torch.Tensor, gamma: float) -> torch.Tensor:
    """Bootstrap on immediates S; terminal q_N ≈ S_N."""
    probs = F.softmax(logits, dim=-1)
    target = (1.0 - gamma) * S[:-1] + gamma * probs[1:].detach()
    td_mse = ((probs[:-1] - target) ** 2).mean()
    term_mse = ((probs[-1] - S[-1]) ** 2).mean()
    return td_mse + term_mse


def train_until(name, W, loss_fn, lr, max_steps, X, P_star,
                grad_tol=None, tv_tol=None, log_every=10000):
    opt = torch.optim.SGD([W], lr=lr)
    loss = gnorm = tv = None
    for step in range(max_steps):
        opt.zero_grad()
        loss = loss_fn(W)
        loss.backward()
        gnorm = W.grad.detach().norm().item()
        tv = tv_mean(P_star, F.softmax(X @ W.detach(), dim=-1))
        if step % log_every == 0 or (grad_tol and gnorm < grad_tol) or (tv_tol and tv < tv_tol):
            print(
                f"{name:12s}  step {step:7d}  loss={loss.item():.6e}  "
                f"||grad||={gnorm:.3e}  ||W||={W.detach().norm().item():.4f}  "
                f"TV(P*,q)={tv:.4e}"
            )
        if grad_tol is not None and gnorm < grad_tol:
            print(f"{name:12s}  stopped: ||grad|| < {grad_tol:g}")
            return step, loss.item(), gnorm, tv
        if tv_tol is not None and tv < tv_tol:
            print(f"{name:12s}  stopped: TV(P*,q) < {tv_tol:g}")
            return step, loss.item(), gnorm, tv
        opt.step()
    print(
        f"{name:12s}  DID NOT hit stop by {max_steps}  "
        f"loss={loss.item():.6e}  ||grad||={gnorm:.3e}  TV={tv:.4e}"
    )
    return max_steps, loss.item(), gnorm, tv


def ood_tv_vs_pstar(W, W_star, mean_val, n_ood, d, gamma):
    """Fresh OOD x; build S then P* on that chain; TV(P*, softmax(xW))."""
    X = mean_val + torch.randn(n_ood, d)
    S = F.softmax(X @ W_star, dim=-1)
    P_star = make_td_targets(S, gamma)
    q = F.softmax(X @ W, dim=-1)
    return tv_mean(P_star, q)


# ---- data ----
W_star = torch.randn(d, k, device=device) / d**0.5
X = torch.randn(n_train, d, device=device)
S = F.softmax(X @ W_star, dim=-1)          # immediates only
P_star = make_td_targets(S, gamma)         # actual probabilities for CE / eval

print(f"||W_star||_F = {W_star.norm().item():.6f}")
print(f"mean H(S)     = {(-(S * (S + 1e-10).log()).sum(-1).mean()).item():.4f}")
print(f"mean H(P*)    = {(-(P_star * (P_star + 1e-10).log()).sum(-1).mean()).item():.4f}")
print(f"mean TV(S,P*) = {tv_mean(S, P_star):.4e}  (immediates vs TD-unrolled)")

W_ce = torch.zeros(d, k, device=device, requires_grad=True)
W_td = torch.zeros(d, k, device=device, requires_grad=True)

print(f"\n--- CE on P* (stop ||grad|| < {grad_tol:g}) ---")
s_ce, l_ce, g_ce, tv_ce = train_until(
    "CE", W_ce,
    lambda W: ce_loss(X @ W, P_star),
    lr, max_steps_ce, X, P_star,
    grad_tol=grad_tol, tv_tol=None,
)

print(f"\n--- TD-stop on S (stop TV(P*,q) < {tv_tol:g}) ---")
s_td, l_td, g_td, tv_td = train_until(
    "TD-stop", W_td,
    lambda W: td_loss(X @ W, S, gamma),
    lr, max_steps_td, X, P_star,
    grad_tol=None, tv_tol=tv_tol,
)

print("\n=== final (vs P*) ===")
print(f"CE       steps={s_ce}  ||W||={W_ce.detach().norm():.4f}  TV={tv_ce:.6e}")
print(f"TD-stop  steps={s_td}  ||W||={W_td.detach().norm():.4f}  TV={tv_td:.6e}")
print(f"||W_ce - W_td||_F = {(W_ce - W_td).norm().item():.6f}")

print("\n=== OOD mean TV vs P* on that OOD chain (n=10) ===")
torch.manual_seed(1)
for mean_val, label in [(0.0, "N(0,I)"), (1.0, "N(1,I)"), (10.0, "N(10,I)")]:
    ce_tv = ood_tv_vs_pstar(W_ce.detach(), W_star, mean_val, n_ood, d, gamma)
    td_tv = ood_tv_vs_pstar(W_td.detach(), W_star, mean_val, n_ood, d, gamma)
    print(f"{label:<10}  CE={ce_tv:.6e}  TD={td_tv:.6e}")

save_path = out_dir / "soft_teacher_ce_td.pt"
torch.save(
    {
        "W_star": W_star.cpu(),
        "W_ce": W_ce.detach().cpu(),
        "W_td": W_td.detach().cpu(),
        "X": X.cpu(),
        "S": S.cpu(),
        "P_star": P_star.cpu(),
        "gamma": gamma,
        "n_train": n_train,
        "d": d,
        "k": k,
        "tv_tol": tv_tol,
        "seed": 0,
        "steps_ce": s_ce,
        "steps_td": s_td,
        "tv_ce": tv_ce,
        "tv_td": tv_td,
    },
    save_path,
)
print(f"\nsaved → {save_path}")
