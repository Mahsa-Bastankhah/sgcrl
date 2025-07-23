import numpy as np

def eigenvalues_J(P: np.ndarray) -> np.ndarray:
    """
    Compute the eigenvalues of
    
        J = (1/2) * [[-P,  I,  P, -I],
                     [ P, -P, -I,  I],
                     [ I, -I,  0,  0],
                     [-I,  I,  0,  0]]
                     
    where P = v vᵀ (shape m×m) and I is the m×m identity.
    
    Parameters
    ----------
    P : (m, m) ndarray
        Rank-1 projector v vᵀ (or any symmetric m×m matrix you want to try).
    
    Returns
    -------
    eigvals : (4m,) ndarray
        The complex eigenvalues of J.
    """
    m = P.shape[0]
    I = np.eye(m)
    Z = np.zeros_like(P)

    # build the 4 × 4 block matrix, each block is m×m
    M = np.block([
        [ -P,  I,  P, -I],
        [  P, -P, -I,  I],
        [  I, -I,  Z,  Z],
        [ -I,  I,  Z,  Z]
    ])

    J = 0.5 * M
    return np.linalg.eigvals(J)


# --- quick demo -------------------------------------------------
if __name__ == "__main__":
    # sample vector v ∈ ℝ^3  →  P = v vᵀ
    v = np.array([0 , 1])
    P = np.outer(v, v)           # shape (3,3)

    lambdas = eigenvalues_J(P)
    print("Eigenvalues of J:", np.sort(lambdas.real))  # sort for readability
