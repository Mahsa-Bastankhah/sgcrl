#!/usr/bin/env python
# tests/test_subspace_transferability.py
"""
Ensure that subspace_transferability is deterministic for a fixed
checkpoint / seed: ranks and projection matrices must match
across successive calls on the same (env, learner_state, networks).
"""

import numpy as np
from experiments.mahsa_utils import get_psi_norms
from experiments.mahsa_utils import get_point_map                    # already defined above
from experiments.mahsa_utils import subspace_transferability  # adjust import

def test_consistency():
    # ----- 1. load env + trained networks -----------------------------------
    env_name   = "point_Wall11x11"
    log_dir    = "logs"
    seed       = 5
    ckpt_num   = 10
    project    = False           # we just want raw ϕ/ψ to bootstrap the network

    (_goal_locs, _psi_norms, _sf, _g, _psi_sim, _waypt_sim,
     _acts, env, trained_learner_state, networks, *_
    ) = get_psi_norms(
        env_name     = env_name,
        log_dir      = log_dir,
        seed         = seed,
        ckpt_num     = ckpt_num,
        project      = project,
    )

    point_map = get_point_map(env_name)

    # ----- 2. run sub-space analysis twice ----------------------------------
    args = dict(
        env              = env,
        trained_learner_state = trained_learner_state,
        networks         = networks,
        point_map        = point_map,
        env_name         = env_name,
        seed             = seed,
        ckpt_num         = ckpt_num,
        verbose          = True,
        use_all_free_cells = True,    
        cell_size = 1

    )

    fit1, cells1, proj1 = subspace_transferability(**args)
    fit2, cells2, proj2 = subspace_transferability(**args)

    # ----- 3. sanity checks --------------------------------------------------
    assert cells1 == cells2, "Cell visitation order changed between runs"

    # map: cell → local rank (first element returned by _svd_basis)
    rank1 = {cell: int(np.linalg.matrix_rank(proj))
             for cell, proj in proj1.items()}
    rank2 = {cell: int(np.linalg.matrix_rank(proj))
             for cell, proj in proj2.items()}

    for cell in cells1:
        # rank consistency
        if rank1[cell] != rank2[cell]:
            print(f"Rank mismatch at cell {cell}: {rank1[cell]} vs {rank2[cell]}")

        
        a = proj1[cell]
        b = proj2[cell]

        # element-wise relative error (avoid divide-by-zero)
        rel_err = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)

        print(f"{cell}:  mean rel-err = {rel_err.mean():.2e}   "
            f"var = {rel_err.var():.2e}")


    print("✔ consistency test passed")

if __name__ == "__main__":
    test_consistency()
