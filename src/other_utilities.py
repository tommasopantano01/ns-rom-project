import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.tri

def make_np_sparse(A_sparse_data, new_size = None, shifts = None, transpose = None):
    if new_size is None:
        new_size = [A_sparse_data.size[0], A_sparse_data.size[1]]
    if shifts is None:
        shifts = [0, 0]
    if transpose is None:
        transpose = False
    if transpose:
        return scipy.sparse.csc_array((A_sparse_data.values, 
                                       ([i + shifts[1] for i in A_sparse_data.cols],
                                        [i + shifts[0] for i in A_sparse_data.rows])), 
                                      shape=(new_size[0], new_size[1]))
    else:
        return scipy.sparse.csc_array((A_sparse_data.values, 
                                       ([i + shifts[0] for i in A_sparse_data.rows],
                                        [i + shifts[1] for i in A_sparse_data.cols])), 
                                      shape=(new_size[0], new_size[1]))
    
def plot_mesh(mesh, export_folder = ""):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_aspect('equal')
    
    coordinates = mesh.cell0_ds_coordinates()
    ax1.triplot(matplotlib.tri.Triangulation(coordinates[0, :], coordinates[1, :]), 'ko-', lw=1)
    ax1.grid(True)

    # Etichette marker sui lati di bordo (a metà del primo lato di ogni marker)
    markers_lati_visti = set()
    for i in range(mesh.cell1_d_total_number()):
        m = mesh.cell1_d_marker(i)
        if m == 0 or m in markers_lati_visti:
            continue
        markers_lati_visti.add(m)
        extremes = mesh.cell1_d_extremes(i)
        p0, p1 = extremes[0], extremes[1]
        x_mid = 0.5 * (coordinates[0, p0] + coordinates[0, p1])
        y_mid = 0.5 * (coordinates[1, p0] + coordinates[1, p1])
        ax1.text(x_mid, y_mid, f"{m}", color='blue', fontsize=12, fontweight='bold')

    # Etichette marker sui vertici angolari (marker non-zero non già visti sui lati)
    markers_vertici_visti = set()
    for i in range(mesh.cell0_d_total_number()):
        m = mesh.cell0_d_marker(i)
        if m == 0 or m in markers_vertici_visti or m in markers_lati_visti:
            continue
        markers_vertici_visti.add(m)
        ax1.text(coordinates[0, i], coordinates[1, i], f"{m}",
                 color='blue', fontsize=12, fontweight='bold')

    if export_folder != "":
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        file_name = 'Mesh.png'
        file_path = os.path.join(export_folder, file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close(fig)
    else:
        plt.pause(0.1)
        plt.close(fig)
        
def plot_solution(mesh, solution_cell0Ds, title = None, export_folder = None):
    if title is None:
        title = "Solution"
    if export_folder is None:
        export_folder = ""
    
    coordinates = mesh.cell0_ds_coordinates()
    x = coordinates[0,:]
    y = coordinates[1,:]
    z = solution_cell0Ds
    triang = matplotlib.tri.Triangulation(x, y)
    
    fig = plt.figure(figsize = plt.figaspect(0.5))
    fig.suptitle(title)
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(triang, z, shading='flat')
    ax1.triplot(matplotlib.tri.Triangulation(coordinates[0, :], coordinates[1, :]), 'k--', lw=1)
    fig.colorbar(tpc)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_trisurf(x, y, z, triangles=triang.triangles, cmap=plt.cm.Spectral)
    
    if export_folder != "": 
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        file_name = title + '.png'
        file_path = os.path.join(export_folder, file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close(fig)
    else:
        plt.pause(0.1)
        plt.close(fig)