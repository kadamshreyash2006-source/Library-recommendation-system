"""
Implicit-feedback ALS (Hu, Koren & Volinsky 2008) in NumPy/SciPy.

Written out rather than pulled from the `implicit` package so the project has
no extra dependency and the maths is inspectable for the report.

Model: each user u and item i gets a latent vector. Observed interactions are
treated as binary preference p=1 with confidence c = 1 + alpha, unobserved as
p=0 with confidence 1. Alternating least squares solves one side at a time;
each solve is a k x k system, so cost scales with interactions, not with the
full user x item matrix.
"""

import numpy as np
import scipy.sparse as sp


def _solve_side(Cui, X, Y, reg, alpha):
    """Update X (one row per user) given fixed Y (one row per item)."""
    n, k = X.shape
    YtY = Y.T @ Y                      # precomputed once: the p=0 term
    base = YtY + reg * np.eye(k, dtype=np.float64)
    Cui = Cui.tocsr()

    for u in range(n):
        start, end = Cui.indptr[u], Cui.indptr[u + 1]
        idx = Cui.indices[start:end]
        if len(idx) == 0:
            X[u] = 0
            continue
        Yu = Y[idx]                    # (n_u, k)
        # confidence above the baseline of 1
        Cu = alpha * Cui.data[start:end]
        # A = YtY + Yu^T (Cu) Yu + reg I ; b = Yu^T (1 + Cu) * p   with p = 1
        A = base + Yu.T @ (Yu * Cu[:, None])
        b = Yu.T @ (1.0 + Cu)
        X[u] = np.linalg.solve(A, b)
    return X


def train_als(Cui, factors=48, reg=0.05, alpha=40.0, iterations=15, seed=0,
              verbose=True):
    """Cui: sparse user x item matrix of interaction counts (CSR).

    Returns (user_factors, item_factors).
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = Cui.shape
    X = rng.normal(0, 0.01, (n_users, factors))
    Y = rng.normal(0, 0.01, (n_items, factors))
    Ciu = Cui.T.tocsr()

    for it in range(iterations):
        X = _solve_side(Cui, X, Y, reg, alpha)
        Y = _solve_side(Ciu, Y, X, reg, alpha)
        if verbose:
            print(f'  ALS iteration {it + 1}/{iterations}', flush=True)
    return X, Y
