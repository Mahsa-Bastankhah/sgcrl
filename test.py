import numpy as np
import matplotlib.pyplot as plt
import os

# === Create plot directory if needed ===
os.makedirs("plots", exist_ok=True)
save_path = "plots/depth_function_map_new.png"

a_vals = np.linspace(-1, 1, 500)

# === Function 1: Composite nonlinearity ===
def f1(a):
    return np.clip(0.5 * np.sin(4 * a) + 0.3 * np.tanh(3 * a) + 0.2 * np.exp(-a**2), 0.0, 1.0)



# def f2(a):
#     # Always zero when action is negative
#     if a < 0:
#         return 0.0

#     # Define the unpredictable spike at e/π ≈ 0.865
#     spike_center = 0.9
#     spike_width = 0.1  # very narrow spike window

#     # Sharp spike to 1
#     if np.abs(a - spike_center) < spike_width:
#         return 1.0

#     # Smooth function starting at 0, rising slowly, bounded below 0.5
#     base = (
#           # damped sine
#         0.8 * (1 - np.exp(-5 * a))                  # smooth rise to ~0.2
#     )

#     return np.clip(base, 0.0, 1)

def f2(a):
    # Always zero when action is negative
    if a < 0:
        return 0.0

    # Define bin width and how many bins (from 0.0 to 1.0)
    bin_width = 0.05
    num_bins = int(1.0 / bin_width)  # 20 bins
    bin_index = int(min(a, 0.999) // bin_width)  # floor into bin

    # Predefined values for each bin (bounded between 0 and 0.85)
    bin_values = [
        0.05, 0.12, 0.08, 0.25, 0.3,
        0.1,  0.15, 0.22, 0.35, 0.4,
        0.85,  # 🔥 bin 10 is the spike → will be overridden to 1.0
        0.45, 0.38, 0.2, 0.28,
        0.33, 0.18, 0.1, 0.05, 0.0
    ]

    spike_bin_index = 10

    if bin_index == spike_bin_index:
        return 1.0
    else:
        return bin_values[bin_index]


f2_vec = np.vectorize(f2)

# === Function 3: Polynomial + cosine ===
def f3(a):
    return np.clip((a ** 5) + 0.5 * np.cos(3 * a), 0.0, 1.0)

# Evaluate each function over the range of action[2] values
z1 = f1(a_vals)
z2 = f2_vec(a_vals)
z3 = f3(a_vals)

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(a_vals, z2, label="Thresholded chaotic", color="orange")

plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

plt.title("state[2] as a function of action[2]")
plt.xlabel("action[2]")
plt.ylabel("state[2] (depth)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved depth function plot to: {save_path}")
