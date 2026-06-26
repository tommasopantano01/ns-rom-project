import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from tqdm import tqdm
from setup_fem import speed_n_dofs, J_A, J_B
from solve_ROM import solve_ROM


def validate_rom(W_test, param_test):

    pod_data = np.load("./data/pod_basis.npy", allow_pickle=True).item()
    B        = pod_data["B"]

    A_r = B.T @ (J_A @ B)
    B_r = B.T @ (J_B @ B)

    X_ux = J_A[:speed_n_dofs, :speed_n_dofs]
    X_uy = J_A[speed_n_dofs:2*speed_n_dofs, speed_n_dofs:2*speed_n_dofs]

    errors_ux, errors_uy, errors_p = [], [], []
    times_rom                      = []

    for j, (mu0, mu1) in enumerate(tqdm(param_test, desc="Validating ROM")):

        U_fom = W_test[:, j]

        t0 = time.time()
        U_rom, _ = solve_ROM(mu0, mu1, B, A_r, B_r, verbose=False)
        times_rom.append(time.time() - t0)

        err    = U_fom - U_rom
        err_ux = err[:speed_n_dofs]
        err_uy = err[speed_n_dofs:2*speed_n_dofs]
        err_p  = err[2*speed_n_dofs:]

        norm_ux = U_fom[:speed_n_dofs]
        norm_uy = U_fom[speed_n_dofs:2*speed_n_dofs]
        norm_p  = U_fom[2*speed_n_dofs:]

        errors_ux.append(np.sqrt(err_ux @ (X_ux @ err_ux)) /
                         np.sqrt(norm_ux @ (X_ux @ norm_ux)))
        errors_uy.append(np.sqrt(err_uy @ (X_uy @ err_uy)) /
                         np.sqrt(norm_uy @ (X_uy @ norm_uy)))
        errors_p.append(np.linalg.norm(err_p) / np.linalg.norm(norm_p))

    errors_ux = np.array(errors_ux)
    errors_uy = np.array(errors_uy)
    errors_p  = np.array(errors_p)
    times_rom = np.array(times_rom)

    print(f"\n=== ROM validation — {len(param_test)} test points ===")
    print(f"{'Component':<12} {'Mean':>10} {'Median':>10} {'95th':>10} {'Max':>10}")
    print("-" * 46)
    for label, errs in [("u_x (H1)", errors_ux),
                        ("u_y (H1)", errors_uy),
                        ("p  (L2)",  errors_p)]:
        print(f"{label:<12} {np.mean(errs):>10.2e} {np.median(errs):>10.2e} "
              f"{np.percentile(errs, 95):>10.2e} {np.max(errs):>10.2e}")

    print(f"\nMean ROM time : {np.mean(times_rom)*1000:.1f} ms")
    fom_path = os.path.join(".", "results", "fom_times.npy")
    
    if os.path.exists(fom_path):
        fom_times = np.load(fom_path)
        speedup = fom_times / times_rom
        print(f"Mean speedup vs FOM : {speedup.mean():.0f}x  |  "
              f"median: {np.median(speedup):.0f}x")
        
    return {
        "errors_ux": errors_ux, "errors_uy": errors_uy, "errors_p": errors_p,
        "times_rom": times_rom, "params": param_test,
        "fom_times": fom_times if os.path.exists(fom_path) else None,
    }

if __name__ == "__main__":
    W_test     = np.load("./data/snapshots_test.npy")
    param_test = np.load("./data/parameters_test.npy")

    validate_rom(W_test, param_test)
