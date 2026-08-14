"""
MSE vs soft-label CE for an overparameterized linear model.

Teacher: y = x @ W_star + eps, p = softmax(y)
Student: f_W(x) = x @ W, d >> n

Both losses can interpolate the train set in many ways; GD's implicit bias
(and what each loss identifies) may yield different OOD behavior.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ------------------------------------------------
# Parameters
# ------------------------------------------------

n_train = 100
n_test = 5000

d = 500
k = 5
r_subspace = 50  # training support for subspace OOD

noise_std = 0.2  # training-label logit noise; test against noiseless teacher

steps = 1000
lr = 1e-2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")

out_dir = Path(__file__).resolve().parent / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def center_logits(logits: torch.Tensor) -> torch.Tensor:
    """Remove softmax-invariant additive offset (mean over classes)."""
    return logits - logits.mean(dim=-1, keepdim=True)


def teacher(x: torch.Tensor, W_star: torch.Tensor, eps: Optional[torch.Tensor] = None):
    logits = x @ W_star
    if eps is not None:
        logits = logits + eps
    probs = F.softmax(logits, dim=-1)
    return logits, probs


@torch.no_grad()
def evaluate(X: torch.Tensor, W_mse: torch.Tensor, W_ce: torch.Tensor, W_star: torch.Tensor):
    true_logits, true_probs = teacher(X, W_star)
    true_c = center_logits(true_logits)

    mse_logits = X @ W_mse
    ce_logits = X @ W_ce

    mse_probs = F.softmax(mse_logits, dim=-1)
    ce_probs = F.softmax(ce_logits, dim=-1)

    mse_c = center_logits(mse_logits)
    ce_c = center_logits(ce_logits)

    def kl(p, q):
        return (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum(-1).mean()

    true_class = true_probs.argmax(-1)

    return {
        "KL_MSE": kl(true_probs, mse_probs).item(),
        "KL_CE": kl(true_probs, ce_probs).item(),
        "L1_MSE": (true_probs - mse_probs).abs().sum(-1).mean().item(),
        "L1_CE": (true_probs - ce_probs).abs().sum(-1).mean().item(),
        "ACC_MSE": (mse_probs.argmax(-1) == true_class).float().mean().item(),
        "ACC_CE": (ce_probs.argmax(-1) == true_class).float().mean().item(),
        # centered logit recovery (fairer than raw logit MSE for CE)
        "LOGIT_MSE": ((mse_c - true_c) ** 2).mean().item(),
        "LOGIT_CE": ((ce_c - true_c) ** 2).mean().item(),
    }


def fmt(result: dict) -> str:
    return (
        f"KL  MSE={result['KL_MSE']:.4e}  CE={result['KL_CE']:.4e} | "
        f"L1  MSE={result['L1_MSE']:.4e}  CE={result['L1_CE']:.4e} | "
        f"ACC MSE={result['ACC_MSE']:.3f}  CE={result['ACC_CE']:.3f} | "
        f"cLogit MSE={result['LOGIT_MSE']:.4e}  CE={result['LOGIT_CE']:.4e}"
    )


# ------------------------------------------------
# Teacher
# ------------------------------------------------

W_star = torch.randn(d, k, device=device) / d**0.5

# ------------------------------------------------
# Training data (full-space Gaussian)
# ------------------------------------------------

X_train = torch.randn(n_train, d, device=device)
eps_train = None
if noise_std > 0:
    eps_train = noise_std * torch.randn(n_train, k, device=device)

Y_train, P_train = teacher(X_train, W_star, eps_train)

# ------------------------------------------------
# Models (zero init → GD min-norm bias for MSE)
# ------------------------------------------------

W_mse = torch.zeros(d, k, device=device, requires_grad=True)
W_ce = torch.zeros(d, k, device=device, requires_grad=True)

opt_mse = torch.optim.SGD([W_mse], lr=lr)
opt_ce = torch.optim.SGD([W_ce], lr=lr)

# ------------------------------------------------
# Training
# ------------------------------------------------

print("training...")
for step in range(steps):
    # ---------- MSE on target logits ----------
    pred_logits = X_train @ W_mse
    loss_mse = ((pred_logits - Y_train) ** 2).mean()

    opt_mse.zero_grad()
    loss_mse.backward()
    opt_mse.step()

    # ---------- Soft-label CE on softmax probs ----------
    pred_logits = X_train @ W_ce
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss_ce = -(P_train * log_probs).sum(dim=-1).mean()

    opt_ce.zero_grad()
    loss_ce.backward()
    opt_ce.step()

    if step % 500 == 0 or step == steps - 1:
        print(f"step {step:4d}  MSE={loss_mse.item():.4e}  CE={loss_ce.item():.4e}")

# Train interpolation (same metrics as test)
print("\nTRAIN (interpolation)")
print(fmt(evaluate(X_train, W_mse, W_ce, W_star)))

with torch.no_grad():
    # How much of W lives in the row-span of X vs orthogonal complement
    # For Gaussian X, span(X) is n-dimensional; min-norm solutions put mass only there.
    # Measure ||W||_F and alignment with W_star (after centering columns).
    def col_center(W):
        return W - W.mean(dim=1, keepdim=True)

    Ws = col_center(W_star)
    Wm = col_center(W_mse.detach())
    Wc = col_center(W_ce.detach())

    def cos(A, B):
        return (A * B).sum() / (A.norm() * B.norm() + 1e-12)

    print(
        f"||W||_F  star={Ws.norm():.4f}  mse={Wm.norm():.4f}  ce={Wc.norm():.4f} | "
        f"cos(W,W*) mse={cos(Wm, Ws):.4f}  ce={cos(Wc, Ws):.4f}"
    )

# ------------------------------------------------
# 1) IID
# ------------------------------------------------

X_test = torch.randn(n_test, d, device=device)
print("\nIID")
print(fmt(evaluate(X_test, W_mse, W_ce, W_star)))

# ------------------------------------------------
# 2) Radial OOD: x_test = c * N(0,I)
# ------------------------------------------------

scales = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
radial_rows = []

print("\nRADIAL OOD")
for scale in scales:
    X = scale * torch.randn(n_test, d, device=device)
    result = evaluate(X, W_mse, W_ce, W_star)
    radial_rows.append(result)
    print(f"scale={scale:g}  {fmt(result)}")

# ------------------------------------------------
# 3) Subspace OOD
# Train models again on x living in first r coords; test with energy in unseen coords.
# (Separate W so the full-space radial experiment above stays clean.)
# ------------------------------------------------

print(f"\nSUBSPACE OOD  (train support r={r_subspace}, test mixes unseen dirs)")

X_sub = torch.zeros(n_train, d, device=device)
X_sub[:, :r_subspace] = torch.randn(n_train, r_subspace, device=device)
Y_sub, P_sub = teacher(X_sub, W_star)

W_mse_s = torch.zeros(d, k, device=device, requires_grad=True)
W_ce_s = torch.zeros(d, k, device=device, requires_grad=True)
opt_mse_s = torch.optim.SGD([W_mse_s], lr=lr)
opt_ce_s = torch.optim.SGD([W_ce_s], lr=lr)

for step in range(steps):
    pred = X_sub @ W_mse_s
    loss = ((pred - Y_sub) ** 2).mean()
    opt_mse_s.zero_grad()
    loss.backward()
    opt_mse_s.step()

    pred = X_sub @ W_ce_s
    loss = -(P_sub * F.log_softmax(pred, dim=-1)).sum(-1).mean()
    opt_ce_s.zero_grad()
    loss.backward()
    opt_ce_s.step()

# In-support test (only first r coords nonzero)
X_in = torch.zeros(n_test, d, device=device)
X_in[:, :r_subspace] = torch.randn(n_test, r_subspace, device=device)
print("in-support")
print(fmt(evaluate(X_in, W_mse_s, W_ce_s, W_star)))

# Unseen directions only
X_out = torch.zeros(n_test, d, device=device)
X_out[:, r_subspace:] = torch.randn(n_test, d - r_subspace, device=device)
print("unseen-only")
print(fmt(evaluate(X_out, W_mse_s, W_ce_s, W_star)))

# Mix: same in-support + unseen energy
X_mix = X_in.clone()
X_mix[:, r_subspace:] = torch.randn(n_test, d - r_subspace, device=device)
print("mix (in + unseen)")
print(fmt(evaluate(X_mix, W_mse_s, W_ce_s, W_star)))

# ------------------------------------------------
# Plot radial KL
# ------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

axes[0].plot(scales, [r["KL_MSE"] for r in radial_rows], marker="o", label="MSE train")
axes[0].plot(scales, [r["KL_CE"] for r in radial_rows], marker="o", label="CE train")
axes[0].set_xlabel("test input scale c")
axes[0].set_ylabel("KL(true || pred)")
axes[0].set_title("Radial OOD — KL")
axes[0].legend()

axes[1].plot(scales, [r["L1_MSE"] for r in radial_rows], marker="o", label="MSE train")
axes[1].plot(scales, [r["L1_CE"] for r in radial_rows], marker="o", label="CE train")
axes[1].set_xlabel("test input scale c")
axes[1].set_ylabel("L1(prob)")
axes[1].set_title("Radial OOD — L1")
axes[1].legend()

axes[2].plot(scales, [r["LOGIT_MSE"] for r in radial_rows], marker="o", label="MSE train")
axes[2].plot(scales, [r["LOGIT_CE"] for r in radial_rows], marker="o", label="CE train")
axes[2].set_xlabel("test input scale c")
axes[2].set_ylabel("MSE(centered logits)")
axes[2].set_title("Radial OOD — centered logits")
axes[2].legend()

fig.tight_layout()
plot_path = out_dir / "mse_vs_ce_radial.png"
fig.savefig(plot_path, dpi=150)
print(f"\nsaved plot → {plot_path}")
