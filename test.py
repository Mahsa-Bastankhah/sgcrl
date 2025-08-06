import numpy as np
import math
import matplotlib.pyplot as plt
# Set random seed for reproducibility

def single_batch():
    np.random.seed(10)

    # Parameters
    n = 100          # number of samples
    d = 500        # dimension of each vector
    c0 = 0.3        # initial parallel component magnitude
    num_steps = 100 # number of update steps
    lr = 0.1        # learning rate

    norm = False

    # # Define the common direction vector z (unit norm)
    z = np.random.randn(d)
    z = z / np.linalg.norm(z)


    # # Covariance for orthogonal Gaussian noise
    # sigma = np.sqrt(1 - c0**2) / math.sqrt(d)

    # # Initialize u and v: u_i = c0*z + zeta_i, v_i = c0*z + kappa_i
    # zeta = np.random.randn(n, d) * sigma
    # kappa = np.random.randn(n, d) * sigma
    # u = c0 * z + zeta
    # v = c0 * z + kappa

    # --- replace the scalar c0 & sigma with per-sample values ---

    # Per-sample z-coefficients (edit these however you like)
    c0_u = np.random.uniform(0.7, 0.7, size=n)  # shape (n,)
    c0_v = np.random.uniform(0.7, 0.7, size=n)  # shape (n,)

    # Per-sample noise scales
    sigma_u = np.sqrt(1.0 - c0_u**2) / math.sqrt(d)  # shape (n,)
    sigma_v = np.sqrt(1.0 - c0_v**2) / math.sqrt(d)  # shape (n,)

    # Initialize u and v: u_i = c0_u[i]*z + zeta_i, v_i = c0_v[i]*z + kappa_i
    zeta  = np.random.randn(n, d) * sigma_u[:, None]
    kappa = np.random.randn(n, d) * sigma_v[:, None]

    # (Optional) make the noise orthogonal to z exactly:
    # zeta  -= (zeta  @ z)[:, None] * z
    # kappa -= (kappa @ z)[:, None] * z

    u = c0_u[:, None] * z + zeta
    v = c0_v[:, None] * z + kappa


    # Normalize each vector to lie on the unit sphere
    def normalize_rows(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    u = normalize_rows(u)
    v = normalize_rows(v)


    dot_same = []
    dot_diff = []
    c_u_history = []
    c_v_history = []
    # Decompose along z: u_i = c_u(i)*z + zeta_i, etc.
    c_u = np.dot(u, z)
    c_v = np.dot(v, z)
    print("Initial c_u:", c_u)
    print("Initial c_v:", c_v)
    c_u_history.append(c_u)
    c_v_history.append(c_v)
    cos_grad_u_history = []
    u_diff = []
    cos_grad_v_history = []
    v_diff = []
    dot_same_var = []                  # population variance
    dot_diff_var = []
    u_off_mean = []
    off_u_vals = []
    u_off_var = []

    v_off_mean = []
    v_off_var = []
    v_norms = []
    u_norms = []
    u_norms = []
    v_norms = []
    u_norms_std = []
    v_norms_std = []
    c_u_cos_history = []
    c_v_cos_history = []


    # Save trajectories of dot products and coefficients


    for step in range(num_steps):
        # Compute logits and p_ij
        logits = np.dot(u, v.T)
        exp_logits = np.exp(logits)
        p = exp_logits / exp_logits.sum(axis=0, keepdims=True)

            # Save p_ii values (matching pair probabilities)
        p_diag = np.diag(p)  # shape (n,)
        if step == 0:
            p_diag_history = []
        p_diag_history.append(p_diag.mean())  # store mean p_ii at this step


        # Compute gradients
        delta = np.eye(n)
        grad_u = (delta - p) @ v
        grad_v = (delta - p).T @ u
        

        # Cosine between grad_u[i] and u[i] for each i
        cos_grad_u = np.sum(grad_u * u, axis=1) / (
            np.linalg.norm(grad_u, axis=1) * np.linalg.norm(u, axis=1)
        )
        cos_grad_u_history.append(cos_grad_u)

        cos_grad_v = np.sum(grad_v * v, axis=1) / (
            np.linalg.norm(grad_v, axis=1) * np.linalg.norm(v, axis=1)
        )
        cos_grad_v_history.append(cos_grad_v)



        # Gradient descent update
        u_before = u.copy()
        v_before = v.copy()
        u = u + lr * grad_u
        v = v + lr * grad_v
        # Mean and std of norms
        u_norm_vals = np.linalg.norm(u, axis=1)
        v_norm_vals = np.linalg.norm(v, axis=1)

        u_norms.append(np.mean(u_norm_vals))
        v_norms.append(np.mean(v_norm_vals))

        u_norms_std.append(np.std(u_norm_vals))
        v_norms_std.append(np.std(v_norm_vals))




        # Normalize after update
        if norm == True:
            u = normalize_rows(u)
            v = normalize_rows(v)

        u_difff = np.linalg.norm(u_before - u , axis =1)
        u_diff.append(u_difff)
        v_difff = np.linalg.norm(v_before - v , axis =1)
        v_diff.append(v_difff)





        u_norm = normalize_rows(u)
        v_norm = normalize_rows(v)
        # Decompose along z: u_i = c_u(i)*z + zeta_i, etc.
        c_u = np.dot(u, z)
        c_v = np.dot(v, z)
        c_u_history.append(c_u)
        c_v_history.append(c_v)
        c_u_cos = np.dot(u_norm, z)
        c_v_cos = np.dot(v_norm, z)
        c_u_cos_history.append(c_u_cos)
        c_v_cos_history.append(c_v_cos)

        # Mean/variance of inner products at this timestep
        M = u_norm @ v_norm.T
        diag_vals = np.diag(M)                                  # matching pairs: u_i · v_i
        off_vals = M[~np.eye(n, dtype=bool)]                    # non-matching: all off-diagonals

        dot_same.append(diag_vals.mean())
        dot_diff.append(off_vals.mean())
        dot_same_var.append(diag_vals.var())                    # population variance
        dot_diff_var.append(off_vals.var())


        # Inner products
        # same_dot = np.sum(u * v, axis=1).mean()
        # diff_dot = (np.sum(u @ v.T) - np.sum(u * v)) / (n * (n - 1))
        # dot_same.append(same_dot)
        # dot_diff.append(diff_dot)
        # ----- inside the for-step loop, just after you assemble M = u @ v.T -----

        # 1) inner products among _u_ vectors
        Mu = u_norm @ u_norm.T
        off_u_vals = Mu[~np.eye(n, dtype=bool)]
        u_off_mean.append(off_u_vals.mean())
        u_off_var.append(off_u_vals.var())

        # 2) inner products among _v_ vectors
        Mv = v_norm @ v_norm.T
        off_v_vals = Mv[~np.eye(n, dtype=bool)]
        v_off_mean.append(off_v_vals.mean())
        v_off_var.append(off_v_vals.var())



    import pandas as pd

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 for 3D projection

    # Convert c_u_history and c_v_history to arrays
    c_u_array = np.array(c_u_history[:-1])
    c_v_array = np.array(c_v_history[:-1])
    dot_same_array = np.array(dot_same)
    dot_diff_array = np.array(dot_diff)

    # Plot mean and std of c_u and c_v over time
    plt.figure(figsize=(12, 6))
    timesteps = np.arange(len(c_u_array))

    mean_c_u = np.mean(c_u_array, axis=1)
    std_c_u = np.std(c_u_array, axis=1)
    mean_c_v = np.mean(c_v_array, axis=1)
    std_c_v = np.std(c_v_array, axis=1)

    plt.fill_between(timesteps, mean_c_u - std_c_u, mean_c_u + std_c_u, alpha=0.2, label='c_u std')
    plt.plot(timesteps, mean_c_u, label='mean c_u')

    plt.fill_between(timesteps, mean_c_v - std_c_v, mean_c_v + std_c_v, alpha=0.2, label='c_v std')
    plt.plot(timesteps, mean_c_v, label='mean c_v')

    plt.xlabel('Timestep')
    plt.ylabel('Projection on z (c values)')
    plt.title('Evolution of Parallel Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_plots/parallel_components_over_time_{norm}.png')



    c_u_array = np.array(c_u_cos_history[:-1])
    c_v_array = np.array(c_v_cos_history[:-1])
    dot_same_array = np.array(dot_same)
    dot_diff_array = np.array(dot_diff)

    # Plot mean and std of c_u and c_v over time
    plt.figure(figsize=(12, 6))
    timesteps = np.arange(len(c_u_array))

    mean_c_u = np.mean(c_u_array, axis=1)
    std_c_u = np.std(c_u_array, axis=1)
    mean_c_v = np.mean(c_v_array, axis=1)
    std_c_v = np.std(c_v_array, axis=1)

    plt.fill_between(timesteps, mean_c_u - std_c_u, mean_c_u + std_c_u, alpha=0.2, label='c_u std')
    plt.plot(timesteps, mean_c_u, label='mean c_u')

    plt.fill_between(timesteps, mean_c_v - std_c_v, mean_c_v + std_c_v, alpha=0.2, label='c_v std')
    plt.plot(timesteps, mean_c_v, label='mean c_v')

    plt.xlabel('Timestep')
    plt.ylabel('cosine value of u , v and z ')
    plt.title('Evolution of Parallel Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_plots/parallel_components_cos_{norm}.png')

    # Plot mean (line) and ±1σ (shade) for matching and non-matching inner products
    plt.figure(figsize=(12, 6))
    timesteps = np.arange(len(dot_same))  # align with histories

    dot_same_array = np.array(dot_same)
    dot_diff_array = np.array(dot_diff)
    dot_same_std = np.sqrt(np.array(dot_same_var))
    dot_diff_std = np.sqrt(np.array(dot_diff_var))

    # matching (u_i · v_i)
    plt.plot(timesteps, dot_same_array, label='matching mean ⟨u_i, v_i⟩')
    plt.fill_between(timesteps,
                    dot_same_array - dot_same_std,
                    dot_same_array + dot_same_std,
                    alpha=0.2, label='matching ±1σ')

    # non-matching (off-diagonals of u @ v.T)
    plt.plot(timesteps, dot_diff_array, label='non-matching mean (off-diag)')
    plt.fill_between(timesteps,
                    dot_diff_array - dot_diff_std,
                    dot_diff_array + dot_diff_std,
                    alpha=0.2, label='non-matching ±1σ')

    plt.xlabel('Timestep')
    plt.ylabel('Dot Product')
    plt.title('Matching vs Non-matching Inner Products (mean ± 1σ)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_plots/dot_products_over_time_{norm}.png')


    cos_grad_u_array = np.array(cos_grad_u_history)
    mean_cos_grad_u = np.mean(cos_grad_u_array, axis=1)
    std_cos_grad_u = np.std(cos_grad_u_array, axis=1)
    mean_cos_grad_v = np.mean(cos_grad_v_history, axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_cos_grad_u, label='mean cos(grad_u, u)')
    #plt.plot(timesteps, u_diff, linestyle='--', color='orange', label='u diff')
    plt.plot(timesteps, mean_cos_grad_v, label='mean cos(grad_v, v)')
    #plt.plot(timesteps, v_diff, linestyle='--', color='orange', label='v diff')
    plt.fill_between(timesteps,
                    mean_cos_grad_u - std_cos_grad_u,
                    mean_cos_grad_u + std_cos_grad_u,
                    alpha=0.2,
                    label='std cos(grad_u, u)')

    plt.xlabel('Timestep')
    plt.ylabel('Cosine')
    plt.title('Cosine Between grad_u and u Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_plots/cosine_grad_u_vs_u_{norm}.png')



    # ----- after training, create mean ± 1σ plot for u·u and v·v -----
    u_off_mean = np.array(u_off_mean)
    v_off_mean = np.array(v_off_mean)
    u_off_std  = np.sqrt(np.array(u_off_var))
    v_off_std  = np.sqrt(np.array(v_off_var))
    timesteps  = np.arange(len(u_off_mean))

    plt.figure(figsize=(12, 6))

    # non-matching u_i · u_j
    plt.plot(timesteps, u_off_mean, label='non-match ⟨u_i, u_j⟩ mean')
    plt.fill_between(timesteps,
                    u_off_mean - u_off_std,
                    u_off_mean + u_off_std,
                    alpha=0.2, label='u off-diag ±1σ')

    # non-matching v_i · v_j
    plt.plot(timesteps, v_off_mean, label='non-match ⟨v_i, v_j⟩ mean')
    plt.fill_between(timesteps,
                    v_off_mean - v_off_std,
                    v_off_mean + v_off_std,
                    alpha=0.2, label='v off-diag ±1σ')

    plt.xlabel('Timestep')
    plt.ylabel('Dot Product')
    plt.title('Non-matching Inner Products: u·u vs v·v (mean ± 1σ)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_plots/uu_vv_offdiag_over_time_{norm}.png')

    u_norms = np.array(u_norms)
    v_norms = np.array(v_norms)
    u_norms_std = np.array(u_norms_std)
    v_norms_std = np.array(v_norms_std)
    timesteps = np.arange(len(u_norms))

    plt.figure(figsize=(12, 6))

    # u norms (mean ± std)
    plt.plot(timesteps, u_norms, label='mean ||u||')
    plt.fill_between(timesteps, u_norms - u_norms_std, u_norms + u_norms_std, alpha=0.2)

    # v norms (mean ± std)
    plt.plot(timesteps, v_norms, label='mean ||v||')
    plt.fill_between(timesteps, v_norms - v_norms_std, v_norms + v_norms_std, alpha=0.2)

    plt.xlabel('Timestep')
    plt.ylabel('Norm')
    plt.title('Mean ± 1σ Norms of u and v Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_plots/u_v_norms_over_time_{norm}.png')
    plt.show()

    p_diag_history = np.array(p_diag_history)
    timesteps = np.arange(len(p_diag_history))

    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, p_diag_history, label='mean p_ii')
    plt.xlabel('Timestep')
    plt.ylabel('Mean p_ii')
    plt.title('Mean Matching Probability $p_{ii}$ Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_plots/p_diag_over_time_{norm}.png')
    plt.show()




def multi_batch():
    print("Running multi-batch simulation...")

        # -------- params for two phases --------
    n1 = 100               # first batch size
    n2 = 80                # second batch size
    d  = 1024
    c0_1 = 0.9            # c0 for batch 1
    c0_2 = 0.9            # c0 for batch 2
    num_steps_phase1 = 100
    num_steps_phase2 = 1000
    lr = 0.1

    # z direction
    z = np.random.randn(d)
    z = z / np.linalg.norm(z)

    def normalize_rows(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def init_batch(n, c0, d, z):
        sigma = np.sqrt(1.0 - c0**2) / math.sqrt(d)
        noise_u = np.random.randn(n, d) * sigma
        noise_v = np.random.randn(n, d) * sigma
        # (optional) make noise exactly orthogonal to z
        # noise_u -= (noise_u @ z)[:, None] * z
        # noise_v -= (noise_v @ z)[:, None] * z
        u = c0 * z + noise_u
        v = c0 * z + noise_v
        return normalize_rows(u), normalize_rows(v)

    # ---------- Phase 1: train first batch ----------
    u1, v1 = init_batch(n1, c0_1, d, z)

    c1_u_hist_phase1 = []   # per-step, the vector of c for batch1 (shape (n1,))
    c1_v_hist_phase1 = []

    for step in range(num_steps_phase1):
        logits = u1 @ v1.T
        exp_logits = np.exp(logits)                   # (n1, n1)
        p = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        delta = np.eye(n1)
        grad_u1 = (delta - p) @ v1                    # (n1, d)
        grad_v1 = (delta - p).T @ u1                  # (n1, d)

        u1 = normalize_rows(u1 + lr * grad_u1)
        v1 = normalize_rows(v1 + lr * grad_v1)

        # track c for batch 1 in phase 1
        c1_u_hist_phase1.append(u1 @ z)              # length n1
        c1_v_hist_phase1.append(v1 @ z)

    # ---------- Phase 2: add second batch, train all together ----------
    u2, v2 = init_batch(n2, c0_2, d, z)

    # concatenate
    u = np.vstack([u1, u2])      # (n1+n2, d)
    v = np.vstack([v1, v2])      # (n1+n2, d)
    N = n1 + n2

    # histories during phase 2, tracked per batch
    c1_u_hist_phase2 = []   # batch1 u c’s per step (shape (n1,))
    c2_u_hist_phase2 = []   # batch2 u c’s per step (shape (n2,))
    c1_v_hist_phase2 = []
    c2_v_hist_phase2 = []

    for step in range(num_steps_phase2):
        logits = u @ v.T                      # (N, N)
        exp_logits = np.exp(logits)
        p = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        delta = np.eye(N)
        grad_u = (delta - p) @ v             # (N, d)
        grad_v = (delta - p).T @ u           # (N, d)

        u = normalize_rows(u + lr * grad_u)
        v = normalize_rows(v + lr * grad_v)

        # track c for each batch separately
        c_all_u = u @ z
        c_all_v = v @ z
        c1_u_hist_phase2.append(c_all_u[:n1])    # first batch slice
        c2_u_hist_phase2.append(c_all_u[n1:])    # second batch slice
        c1_v_hist_phase2.append(c_all_v[:n1])
        c2_v_hist_phase2.append(c_all_v[n1:])

    # ---------- Plot: how c changed for batch 1 and batch 2 ----------
    # Convert to arrays
    c1_u_phase1 = np.array(c1_u_hist_phase1)      # (T1, n1)
    c1_u_phase2 = np.array(c1_u_hist_phase2)      # (T2, n1)
    c2_u_phase2 = np.array(c2_u_hist_phase2)      # (T2, n2)

    # means/stds over items in each batch
    mean_c1_phase1 = c1_u_phase1.mean(axis=1)
    std_c1_phase1  = c1_u_phase1.std(axis=1)
    mean_c1_phase2 = c1_u_phase2.mean(axis=1)
    std_c1_phase2  = c1_u_phase2.std(axis=1)
    mean_c2_phase2 = c2_u_phase2.mean(axis=1)
    std_c2_phase2  = c2_u_phase2.std(axis=1)

    # Build a single timeline where phase 2 continues after phase 1
    t1 = np.arange(num_steps_phase1)
    t2 = np.arange(num_steps_phase1, num_steps_phase1 + num_steps_phase2)

    plt.figure(figsize=(12, 6))

    # Batch 1: show both phases back-to-back
    plt.plot(t1, mean_c1_phase1, label='Batch 1 (phase 1) mean c(u)')
    plt.fill_between(t1, mean_c1_phase1 - std_c1_phase1, mean_c1_phase1 + std_c1_phase1, alpha=0.2)

    plt.plot(t2, mean_c1_phase2, label='Batch 1 (phase 2) mean c(u)')
    plt.fill_between(t2, mean_c1_phase2 - std_c1_phase2, mean_c1_phase2 + std_c1_phase2, alpha=0.2)

    # Batch 2: appears only in phase 2 (starts at t = num_steps_phase1)
    plt.plot(t2, mean_c2_phase2, label='Batch 2 (phase 2) mean c(u)')
    plt.fill_between(t2, mean_c2_phase2 - std_c2_phase2, mean_c2_phase2 + std_c2_phase2, alpha=0.2)

    plt.xlabel('Step')
    plt.ylabel('Projection on z  (c = ⟨u, z⟩)')
    plt.title('Evolution of c for Batch 1 (both phases) and Batch 2 (phase 2)')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_plots/c_evolution_two_batches_u.png')

    # (Optional) same plot for v’s
    c1_v_phase1 = np.array(c1_v_hist_phase1)
    c1_v_phase2 = np.array(c1_v_hist_phase2)
    c2_v_phase2 = np.array(c2_v_hist_phase2)

    mean_c1v_phase1 = c1_v_phase1.mean(axis=1); std_c1v_phase1 = c1_v_phase1.std(axis=1)
    mean_c1v_phase2 = c1_v_phase2.mean(axis=1); std_c1v_phase2 = c1_v_phase2.std(axis=1)
    mean_c2v_phase2 = c2_v_phase2.mean(axis=1); std_c2v_phase2 = c2_v_phase2.std(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(t1, mean_c1v_phase1, label='Batch 1 (phase 1) mean c(v)')
    plt.fill_between(t1, mean_c1v_phase1 - std_c1v_phase1, mean_c1v_phase1 + std_c1v_phase1, alpha=0.2)
    plt.plot(t2, mean_c1v_phase2, label='Batch 1 (phase 2) mean c(v)')
    plt.fill_between(t2, mean_c1v_phase2 - std_c1v_phase2, mean_c1v_phase2 + std_c1v_phase2, alpha=0.2)
    plt.plot(t2, mean_c2v_phase2, label='Batch 2 (phase 2) mean c(v)')
    plt.fill_between(t2, mean_c2v_phase2 - std_c2v_phase2, mean_c2v_phase2 + std_c2v_phase2, alpha=0.2)
    plt.xlabel('Step')
    plt.ylabel('Projection on z  (c = ⟨v, z⟩)')
    plt.title('Evolution of c for v’s: Batch 1 and Batch 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_plots/c_evolution_two_batches_v.png')


multi_batch()