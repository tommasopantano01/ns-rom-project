import numpy as np
import os
import scipy.sparse
from pypolydim import polydim, gedim
from pypolydim.export_vtk_utilities import ExportVTKUtilities
import sys
sys.path.insert(1, '../')
import other_utilities as other_ut

# ── Geometry ──────────────────────────────────────────────────────────────────
geometry_utilities_config = gedim.GeometryUtilitiesConfig()
geometry_utilities_config.tolerance1_d = 1.0e-6
geometry_utilities_config.tolerance2_d = 1.0e-12
geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
mesh_utilities = gedim.MeshUtilities()
vtk_utilities  = ExportVTKUtilities()

# ── Export folders ─────────────────────────────────────────────────────────────
export_file_path     = "./Export/PJ"
export_mesh_path     = export_file_path + "/Mesh"
export_solution_path = export_file_path + "/Solution"
for p in [export_file_path, export_mesh_path, export_solution_path]:
    if not os.path.exists(p):
        os.makedirs(p)

# ── Mesh ───────────────────────────────────────────────────────────────────────
pde_domain = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D()
pde_domain.vertices = np.array([[0.0, 1.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0, 1.0],
                                 [0.0, 0.0, 0.0, 0.0]])
pde_domain.shape_type = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D.Domain_Shape_Types.parallelogram
pde_domain.area = 1.0

mesh_type   = polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.triangular
method_type = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.fem_pcc
mesh_size   = 0.001

mesh_data = gedim.MeshMatrices()
mesh      = gedim.MeshMatricesDAO(mesh_data)

polydim.pde_tools.mesh.pde_mesh_utilities.create_mesh_2_d(
    geometry_utilities, mesh_utilities, mesh_type, pde_domain, mesh_size, mesh
)
mesh_geometric_data = polydim.pde_tools.mesh.pde_mesh_utilities.compute_mesh_2_d_geometry_data(
    geometry_utilities, mesh_utilities, mesh
)
vtk_utilities.export_mesh(export_mesh_path, mesh)

# ── DOF setup ─────────────────────────────────────────────────────────────────
info_internal = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(
    polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.none)
info_internal.marker = 0

info_dirichlet = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(
    polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.strong)
info_dirichlet.marker = 1

info_neumann_none = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(
    polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.none)
info_neumann_none.marker = 0

pressure_boundary_info = {i: (info_dirichlet if i == 1 else
                              info_internal  if i == 0 else
                              info_neumann_none) for i in range(9)}

speed_boundary_info = {0: info_internal,
                       **{i: info_dirichlet for i in range(1, 9)}}

mesh_connectivity_data = polydim.pde_tools.mesh.MeshMatricesDAO_mesh_connectivity_data(mesh)

pressure_reference_element_data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(method_type, 1)
speed_reference_element_data    = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(method_type, 2)

dof_manager = polydim.pde_tools.do_fs.DOFsManager()

pressure_mesh_dofs_info = polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(
    pressure_reference_element_data, mesh, pressure_boundary_info)
pressure_dofs_data = dof_manager.create_do_fs_2_d(pressure_mesh_dofs_info, mesh_connectivity_data)

speed_mesh_dofs_info = polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(
    speed_reference_element_data, mesh, speed_boundary_info)
speed_dofs_data = dof_manager.create_do_fs_2_d(speed_mesh_dofs_info, mesh_connectivity_data)

pressure_n_dofs    = pressure_dofs_data.number_do_fs
pressure_n_strongs = pressure_dofs_data.number_strongs
speed_n_dofs       = speed_dofs_data.number_do_fs
speed_n_strongs    = speed_dofs_data.number_strongs
tot_dofs           = 2 * speed_n_dofs + pressure_n_dofs
tot_strongs        = 2 * speed_n_strongs + pressure_n_strongs

print("P dofs\t", "P stgs\t", "U dofs\t", "U stgs\t", "T dofs\t", "T stgs")
print(pressure_n_dofs, "\t", pressure_n_strongs, "\t",
      speed_n_dofs,    "\t", speed_n_strongs,    "\t",
      tot_dofs,        "\t", tot_strongs)

# ── Strong BCs (omogenee) ─────────────────────────────────────────────────────
def zero_strong(marker, x, y, z):
    return 0.0

u_x_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(
    geometry_utilities, mesh, mesh_geometric_data,
    speed_mesh_dofs_info, speed_dofs_data, speed_reference_element_data, zero_strong)
u_y_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(
    geometry_utilities, mesh, mesh_geometric_data,
    speed_mesh_dofs_info, speed_dofs_data, speed_reference_element_data, zero_strong)
p_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(
    geometry_utilities, mesh, mesh_geometric_data,
    pressure_mesh_dofs_info, pressure_dofs_data, pressure_reference_element_data, zero_strong)

# ── Linear operators (parameter-independent) ──────────────────────────────────
def unit_term(x, y, z): return 1.0
def b_x_term(x, y, z):  return np.array([1.0, 0.0, 0.0])
def b_y_term(x, y, z):  return np.array([0.0, 1.0, 0.0])

A_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_diffusion_operator(
    geometry_utilities, mesh, mesh_geometric_data,
    speed_dofs_data, speed_dofs_data,
    speed_reference_element_data, speed_reference_element_data, unit_term)

A_x = other_ut.make_np_sparse(A_operator.operator_dofs, [tot_dofs, tot_dofs], [0, 0])
A_y = other_ut.make_np_sparse(A_operator.operator_dofs, [tot_dofs, tot_dofs], [speed_n_dofs, speed_n_dofs])
J_A = A_x + A_y

B_x_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_advection_operator(
    geometry_utilities, mesh, mesh_geometric_data,
    speed_dofs_data, pressure_dofs_data,
    speed_reference_element_data, pressure_reference_element_data, b_x_term)
B_y_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_advection_operator(
    geometry_utilities, mesh, mesh_geometric_data,
    speed_dofs_data, pressure_dofs_data,
    speed_reference_element_data, pressure_reference_element_data, b_y_term)

J_B_x  = other_ut.make_np_sparse(B_x_operator.operator_dofs, [tot_dofs, tot_dofs], [2*speed_n_dofs, 0])
J_B_y  = other_ut.make_np_sparse(B_y_operator.operator_dofs, [tot_dofs, tot_dofs], [2*speed_n_dofs, speed_n_dofs])
J_BT_x = other_ut.make_np_sparse(B_x_operator.operator_dofs, [tot_dofs, tot_dofs], [2*speed_n_dofs, 0], True)
J_BT_y = other_ut.make_np_sparse(B_y_operator.operator_dofs, [tot_dofs, tot_dofs], [2*speed_n_dofs, speed_n_dofs], True)
J_B    = J_B_x + J_B_y + J_BT_x + J_BT_y
# ── Termine sorgente f(x; mu1) ────────────────────────────────────────────────
def make_f_functions(mu1):
    pi = np.pi

    def f_x_function(x, y, z):
        return -(mu1**3 * pi**2 * np.cos(mu1**2 * pi * x) - mu1**2 * pi**2) \
               * np.sin(mu1 * pi * y) * np.cos(mu1 * pi * y) \
               + mu1 * pi * np.cos(mu1 * pi * x) * np.cos(mu1 * pi * y)

    def f_y_function(x, y, z):
        return -(-mu1**3 * pi**2 * np.cos(mu1**2 * pi * y) + mu1**2 * pi**2) \
               * np.sin(mu1 * pi * x) * np.cos(mu1 * pi * x) \
               - mu1 * pi * np.sin(mu1 * pi * x) * np.sin(mu1 * pi * y)

    return f_x_function, f_y_function


def assemble_f(mu1):
    f_x_function, f_y_function = make_f_functions(mu1)

    f_x = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(
        geometry_utilities, mesh, mesh_geometric_data,
        speed_dofs_data, speed_reference_element_data, speed_reference_element_data,
        f_x_function)

    f_y = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(
        geometry_utilities, mesh, mesh_geometric_data,
        speed_dofs_data, speed_reference_element_data, speed_reference_element_data,
        f_y_function)

    return np.concatenate([f_x, f_y, np.zeros(pressure_n_dofs)])
print("Setup complete.")
