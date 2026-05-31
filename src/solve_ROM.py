import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import scipy.sparse.linalg
from pypolydim import polydim
from setup_fem import (
    geometry_utilities, mesh, mesh_geometric_data,
    speed_dofs_data, speed_reference_element_data,
    pressure_n_dofs, speed_n_dofs, tot_dofs,
    u_x_strong, u_y_strong,
    J_A, J_B,
    assemble_f
)
import other_utilities as other_ut


def solve_ROM(mu0, mu1, B, newton_tol=1.0e-6, max_iterations=20, verbose=True):
    # ── Proiezione operatori lineari ─────────────────────────────────────────
    A_r = B.T @ (J_A @ B)
    B_r = B.T @ (J_B @ B)

    f_full   = assemble_f(mu1)
    f_N      = B.T @ f_full
    J_lin_N  = mu0 * A_r - B_r

    U_N           = np.zeros(B.shape[1])
    residual_norm = 1.0
    solution_norm = 1.0
    num_iteration = 1
    history       = []

    while num_iteration < max_iterations and residual_norm > newton_tol * solution_norm:

        U_delta_k   = B @ U_N
        u_x_numeric = U_delta_k[0:speed_n_dofs]
        u_y_numeric = U_delta_k[speed_n_dofs:2*speed_n_dofs]

        c_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_ns_operators(
            geometry_utilities, mesh, mesh_geometric_data,
            speed_dofs_data, speed_reference_element_data,
            u_x_numeric, u_y_numeric, u_x_strong, u_y_strong
        )

        J_C = other_ut.make_np_sparse(
            c_operator.convective_operator.operator_dofs,
            [tot_dofs, tot_dofs], [0, 0]
        )

        f_C_delta = np.concatenate([c_operator.convective_rhs,
                                    np.zeros(pressure_n_dofs)])

        J_C_N = B.T @ (J_C @ B)
        f_C_N = B.T @ f_C_delta

        G_N       = f_N - f_C_N - J_lin_N @ U_N
        delta_U_N = np.linalg.solve(J_lin_N + J_C_N, G_N)
        U_N       = U_N + delta_U_N

        residual_norm = np.linalg.norm(delta_U_N)
        solution_norm = np.linalg.norm(U_N)
        if solution_norm == 0.0:
            solution_norm = 1.0

        history.append(residual_norm / solution_norm)

        if verbose:
            print("it {:d}/{:d}  rel. residual = {:.3e}".format(
                num_iteration, max_iterations, residual_norm / solution_norm))

        num_iteration += 1

    return B @ U_N, history
