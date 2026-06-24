# src/build_pinn_data.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from setup_fem import mesh, pressure_dofs_data, speed_dofs_data, speed_n_dofs

def build_pinn_data(snapshots, params, data_dir):
    N_verts = mesh.cell0_d_total_number()

    free_verts, p_idx_list, ux_idx_list = [], [], []
    for i in range(N_verts):
        p_dofs = pressure_dofs_data.cells_do_fs[0][i]
        u_dofs = speed_dofs_data.cells_do_fs[0][i]
        if len(p_dofs) > 0 and int(p_dofs[0].type) == 2:
            free_verts.append(i)
            p_idx_list.append(p_dofs[0].global_index)
            ux_idx_list.append(u_dofs[0].global_index if len(u_dofs) > 0 else -1)

    free_verts = np.array(free_verts)
    p_idx      = np.array(p_idx_list)
    ux_idx     = np.array(ux_idx_list)

    coords = np.array([[mesh.cell0_d_coordinate_x(i),
                        mesh.cell0_d_coordinate_y(i)] for i in free_verts])

    ux = snapshots[ux_idx, :]                   # (829, N_snap)
    uy = snapshots[speed_n_dofs + ux_idx, :]    # (829, N_snap)
    p  = snapshots[2*speed_n_dofs + p_idx, :]   # (829, N_snap)

    os.makedirs(data_dir, exist_ok=True)
    np.save(f"{data_dir}/mesh_coords_p1.npy", coords)
    np.save(f"{data_dir}/params_pinn.npy",    params)
    np.save(f"{data_dir}/ux_on_nodes.npy",    ux)
    np.save(f"{data_dir}/uy_on_nodes.npy",    uy)
    np.save(f"{data_dir}/p_on_nodes.npy",     p)
    print(f"Saved: coords{coords.shape} ux{ux.shape} p{p.shape}")

if __name__ == "__main__":
    snapshots = np.load("./data/snapshots_train.npy")
    params    = np.load("./data/parameters_train.npy")
    build_pinn_data(snapshots, params, "./data")
