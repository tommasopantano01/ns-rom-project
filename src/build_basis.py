import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from setup_fem import (
    A_operator, B_x_operator, B_y_operator,
    speed_n_dofs, pressure_n_dofs, tot_dofs
)
import other_utilities as other_ut


# ── Prodotto interno H1 sulla velocità ────────────────────────────────────────
X_1 = other_ut.make_np_sparse(
    A_operator.operator_dofs,
    [2 * speed_n_dofs, 2 * speed_n_dofs], [0, 0])
X_2 = other_ut.make_np_sparse(
    A_operator.operator_dofs,
    [2 * speed_n_dofs, 2 * speed_n_dofs], [speed_n_dofs, speed_n_dofs])
inner_product_u = X_1 + X_2

B_1 = other_ut.make_np_sparse(
    B_x_operator.operator_dofs,
    [pressure_n_dofs, 2 * speed_n_dofs], [0, 0])
B_2 = other_ut.make_np_sparse(
    B_y_operator.operator_dofs,
    [pressure_n_dofs, 2 * speed_n_dofs], [0, speed_n_dofs])
B_div = B_1 + B_2


# ── POD ───────────────────────────────────────────────────────────────────────
def pod(snapshots, X=None, tol=1.0 - 1.0e-6, N_max=100):
    """
    snapshots : (M, n_dofs)
    returns   : basis (n_dofs, N), eigenvalues lam, cumulative energy
    """
    C = snapshots @ (X @ snapshots.T) if X is not None else snapshots @ snapshots.T

    lam, alpha = np.linalg.eigh(C)
    idx   = np.argsort(lam)[::-1]
    lam   = np.clip(lam[idx], 0.0, None)
    alpha = alpha[:, idx]

    energy = np.cumsum(lam) / np.sum(lam)
    N = int(np.argmax(energy >= tol)) + 1 if np.any(energy >= tol) else N_max
    N = min(N, N_max)

    basis = np.zeros((snapshots.shape[1], N))
    for n in range(N):
        v    = snapshots.T @ alpha[:, n]
        norm = np.sqrt(v @ (X @ v)) if X is not None else np.linalg.norm(v)
        basis[:, n] = v / norm

    return basis, lam, energy


# ── Costruzione della base ridotta ────────────────────────────────────────────
def build_basis(W_train, pod_tol=1.0 - 1.0e-6, N_max=100, verbose=True):
    """
    W_train : (tot_dofs, N_train)
    returns : B (tot_dofs, N_tot), dizionario con basi e autovalori
    """
    snapshot_u = W_train[0:2 * speed_n_dofs, :].T
    snapshot_p = W_train[2 * speed_n_dofs:, :].T

    RHS_s      = B_div.T @ snapshot_p.T
    snapshot_s = scipy.sparse.linalg.spsolve(inner_product_u, RHS_s).T

    if verbose:
        print(f"snapshot_u: {snapshot_u.shape}")
        print(f"snapshot_s: {snapshot_s.shape}")
        print(f"snapshot_p: {snapshot_p.shape}")

    V_u, lam_u, energy_u = pod(snapshot_u, X=inner_product_u,
                                tol=pod_tol, N_max=N_max)
    V_s, lam_s, energy_s = pod(snapshot_s, X=inner_product_u,
                                tol=pod_tol, N_max=N_max)
    V_p, lam_p, energy_p = pod(snapshot_p,
                                tol=pod_tol, N_max=N_max)

    if verbose:
        print(f"N_u={V_u.shape[1]}, N_s={V_s.shape[1]}, N_p={V_p.shape[1]}")

    N_u   = V_u.shape[1]
    N_s   = V_s.shape[1]
    N_p   = V_p.shape[1]
    N_tot = N_u + N_s + N_p

    B = np.zeros((tot_dofs, N_tot))
    B[0:2 * speed_n_dofs, 0:N_u]           = V_u
    B[0:2 * speed_n_dofs, N_u:N_u + N_s]   = V_s
    B[2 * speed_n_dofs:,  N_u + N_s:]       = V_p

    if verbose:
        print(f"Global basis B: {B.shape}")
        X_block   = scipy.sparse.block_diag([inner_product_u,
                                             scipy.sparse.eye(pressure_n_dofs)])
        orth_err  = np.linalg.norm(B.T @ (X_block @ B) - np.eye(N_tot), "fro")
        print(f"Orthogonality error ||B.T X B - I||_F = {orth_err:.2e}")

    return B, {
        "V_u": V_u, "V_s": V_s, "V_p": V_p,
        "lam_u": lam_u, "lam_s": lam_s, "lam_p": lam_p,
        "energy_u": energy_u, "energy_s": energy_s, "energy_p": energy_p,
        "N_u": N_u, "N_s": N_s, "N_p": N_p, "N_tot": N_tot,
        "inner_product_u": inner_product_u
    }
