import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import scipy.sparse.linalg
from pypolydim import polydim
from setup_fem import (
    geometry_utilities, mesh, mesh_geometric_data,
    speed_dofs_data, speed_reference_element_data,
    pressure_dofs_data, pressure_reference_element_data,
    speed_n_dofs, pressure_n_dofs,
    speed_n_strongs, pressure_n_strongs,
    tot_dofs, J_A, J_B,
    u_x_strong, u_y_strong, p_strong,
    assemble_f
)
import other_utilities as other_ut


def solve_FOM(mu0, mu1, newton_tol=1.0e-6, max_iterations=20, verbose=True):

    f_delta = assemble_f(mu1)
    J_lin   = mu0 * J_A - J_B

    u_x_numeric = np.zeros(speed_n_dofs)
    u_y_numeric = np.zeros(speed_n_dofs)
    p_numeric   = np.zeros(pressure_n_dofs)
    U_delta     = np.concatenate([u_x_numeric, u_y_numeric, p_numeric])

    du_x_strong = np.zeros(speed_n_strongs)
    du_y_strong = np.zeros(speed_n_strongs)
    dp_strong   = np.zeros(pressure_n_strongs)

    residual_norm = 1.0
    solution_norm = 1.0
    num_iteration = 1
    history       = []

    while num_iteration < max_iterations and residual_norm > newton_tol * solution_norm:

        c_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_ns_operators(
            geometry_utilities, mesh, mesh_geometric_data,
            speed_dofs_data, speed_reference_element_data,
            u_x_numeric, u_y_numeric, u_x_strong, u_y_strong
        )

        J_C = other_ut.make_np_sparse(
            c_operator.convective_operator.operator_dofs,
            [tot_dofs, tot_dofs], [0, 0]
        )

        f_C_delta = np.concatenate([c_operator.convective_rhs, np.zeros(pressure_n_dofs)])

        G_delta = f_delta - f_C_delta - J_lin @ U_delta
        delta_U = scipy.sparse.linalg.spsolve(J_lin + J_C, G_delta)
        U_delta = U_delta + delta_U

        delta_ux = delta_U[0:speed_n_dofs]
        delta_uy = delta_U[speed_n_dofs:2*speed_n_dofs]
        delta_p  = delta_U[2*speed_n_dofs:]

        u_x_numeric = U_delta[0:speed_n_dofs]
        u_y_numeric = U_delta[speed_n_dofs:2*speed_n_dofs]
        p_numeric   = U_delta[2*speed_n_dofs:]

        du_x_e = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
            geometry_utilities, mesh, mesh_geometric_data,
            speed_dofs_data, speed_reference_element_data,
            delta_ux, du_x_strong)
        du_y_e = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
            geometry_utilities, mesh, mesh_geometric_data,
            speed_dofs_data, speed_reference_element_data,
            delta_uy, du_y_strong)
        dp_e = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
            geometry_utilities, mesh, mesh_geometric_data,
            pressure_dofs_data, pressure_reference_element_data,
            delta_p, dp_strong)

        u_x_e = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
            geometry_utilities, mesh, mesh_geometric_data,
            speed_dofs_data, speed_reference_element_data,
            u_x_numeric, u_x_strong)
        u_y_e = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
            geometry_utilities, mesh, mesh_geometric_data,
            speed_dofs_data, speed_reference_element_data,
            u_y_numeric, u_y_strong)
        p_e = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
            geometry_utilities, mesh, mesh_geometric_data,
            pressure_dofs_data, pressure_reference_element_data,
            p_numeric, p_strong)

        solution_norm = np.sqrt(
            u_x_e.numeric_norm_l2**2 +
            u_y_e.numeric_norm_l2**2 +
            p_e.numeric_norm_l2**2)
        residual_norm = np.sqrt(
            du_x_e.numeric_norm_l2**2 +
            du_y_e.numeric_norm_l2**2 +
            dp_e.numeric_norm_l2**2)

        if solution_norm == 0.0:
            solution_norm = 1.0

        history.append(residual_norm / solution_norm)

        if verbose:
            print("it {:d}/{:d}  rel. residual = {:.3e}".format(
                num_iteration, max_iterations, residual_norm / solution_norm))

        num_iteration += 1

    return U_delta, history
