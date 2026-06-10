import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from tqdm import tqdm
from setup_fem import speed_n_dofs, J_A
from solve_FOM import solve_FOM
from solve_PODNN import solve_PODNN, load_PODNN
from build_basis import build_basis


def validate_podnn(W_test, param_test, net, B, x_mean, x_std, y_scale):

    X_ux = J_A[:speed_n_dofs, :speed_n_dofs]
    X_uy = J_A[speed_n_dofs:2*speed_n_dofs, speed_n_dofs:2*speed_n_dofs]

    err_ux, err_uy, err_p = [], [], []
    t_fom, t_nn           = [], []

    for j, (m0, m1) in enumerate(tqdm(param_test, desc="Validating POD-NN")):
        U_f = W_test[:, j]

        t0 = time.time()
        U_n = solve_PODNN(m0, m1, net, B, x_mean, x_std, y_scale)
        t_nn.append(time.time() - t0)

        e    = U_f - U_n
        e_ux = e[:speed_n_dofs]
        e_uy = e[speed_n_dofs:2*speed_n_dofs]
        e_p  = e[2*speed_n_dofs:]
        n_ux = U_f[:speed_n_dofs]
        n_uy = U_f[speed_n_dofs:2*speed_n_dofs]
        n_p  = U_f[2*speed_n_dofs:]

        err_ux.append(np.sqrt(e_ux @ (X_ux @ e_ux)) /
                      np.sqrt(n_ux @ (X_ux @ n_ux)))
        err_uy.append(np.sqrt(e_uy @ (X_uy @ e_uy)) /
                      np.sqrt(n_uy @ (X_uy @ n_uy)))
        err_p.append(np.linalg.norm(e_p) / np.linalg.norm(n_p))

    err_ux = np.array(err_ux)
    err_uy = np.array(err_uy)
    err_p  = np.array(err_p)
    t_nn   = np.array(t_nn)

    print(f"\n=== POD-NN validation — {len(param_test)} test points ===")
    print(f"{'Component':<12} {'Mean':>10} {'Median':>10} {'95th':>10} {'Max':>10}")
    print("-" * 46)
    for label, errs in [("u_x (H1)", err_ux),
                        ("u_y (H1)", err_uy),
                        ("p  (L2)",  err_p)]:
        print(f"{label:<12} {np.mean(errs):>10.2e} {np.median(errs):>10.2e} "
              f"{np.percentile(errs, 95):>10.2e} {np.max(errs):>10.2e}")

    print(f"\nMean POD-NN time : {np.mean(t_nn)*1000:.3f} ms")

    return {
        "err_ux": err_ux, "err_uy": err_uy, "err_p": err_p,
        "t_nn": t_nn, "params": param_test,
    }


if __name__ == "__main__":
    W_test     = np.load("./data/snapshots_test.npy")
    param_test = np.load("./data/parameters_test.npy")

    net, x_mean, x_std, y_scale = load_PODNN("./models/podnn_weights.pt")

    # La base POD deve essere salvata insieme ai pesi
    pod_data = np.load("./models/pod_basis.npy", allow_pickle=True).item()
    B = pod_data["B"]

    validate_podnn(W_test, param_test, net, B, x_mean, x_std, y_scale)
