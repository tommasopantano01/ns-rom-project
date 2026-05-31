import numpy as np
import time
from tqdm import tqdm
from setup_fem import speed_n_dofs, J_A
from solve_FOM import solve_FOM
from solve_ROM import solve_ROM
from build_basis import build_basis


def compare_FOM_ROM(W_train, param_test, n_test=None, pod_tol=1.0-1.0e-6, N_max=100):
    """
    Confronto FOM vs ROM su parametri di test.

    Parameters
    ----------
    W_train    : array (tot_dofs, N_train)
    param_test : array (N_test, 2) — colonne [mu0, mu1]
    n_test     : int, opzionale — se None usa tutti i param_test
    """

    # ── Costruzione base ─────────────────────────────────────────────────────
    B, _ = build_basis(W_train, pod_tol=pod_tol, N_max=N_max, verbose=True)

    # ── Parametri di test ────────────────────────────────────────────────────
    if n_test is not None:
        param_test = param_test[:n_test]

    mu0_vals = param_test[:, 0]
    mu1_vals = param_test[:, 1]

    X_ux = J_A[:speed_n_dofs, :speed_n_dofs]
    X_uy = J_A[speed_n_dofs:2*speed_n_dofs, speed_n_dofs:2*speed_n_dofs]

    errors_ux, errors_uy, errors_p = [], [], []
    times_fom, times_rom           = [], []

    for mu0, mu1 in tqdm(zip(mu0_vals, mu1_vals), total=len(mu0_vals),
                         desc="FOM vs ROM"):

        t0 = time.time()
        U_fom, _ = solve_FOM(mu0, mu1, verbose=False)
        times_fom.append(time.time() - t0)

        t0 = time.time()
        U_rom, _ = solve_ROM(mu0, mu1, B, verbose=False)
        times_rom.append(time.time() - t0)

        err    = U_fom - U_rom
        err_ux = err[:speed_n_dofs]
        err_uy = err[speed_n_dofs:2*speed_n_dofs]
        err_p  = err[2*speed_n_dofs:]

        norm_ux = U_fom[:speed_n_dofs]
        norm_uy = U_fom[speed_n_dofs:2*speed_n_dofs]
        norm_p  = U_fom[2*speed_n_dofs:]

        errors_ux.append(
            np.sqrt(err_ux @ (X_ux @ err_ux)) /
            np.sqrt(norm_ux @ (X_ux @ norm_ux)))
        errors_uy.append(
            np.sqrt(err_uy @ (X_uy @ err_uy)) /
            np.sqrt(norm_uy @ (X_uy @ norm_uy)))
        errors_p.append(
            np.linalg.norm(err_p) / np.linalg.norm(norm_p))

    errors_ux = np.array(errors_ux)
    errors_uy = np.array(errors_uy)
    errors_p  = np.array(errors_p)
    speedups  = np.array(times_fom) / np.array(times_rom)

    # ── Stampa risultati ─────────────────────────────────────────────────────
    print(f"\n=== FOM vs ROM — {len(mu0_vals)} test points ===")
    print(f"{'Componente':<12} {'Mean':>10} {'Median':>10} {'95th':>10} {'Max':>10}")
    print("-" * 46)
    for label, errs in [("u_x (H1)", errors_ux),
                         ("u_y (H1)", errors_uy),
                         ("p  (L2)",  errors_p)]:
        print(f"{label:<12} {np.mean(errs):>10.2e} {np.median(errs):>10.2e} "
              f"{np.percentile(errs, 95):>10.2e} {np.max(errs):>10.2e}")

    print(f"\nSpeedup  mean={np.mean(speedups):.1f}x  "
          f"median={np.median(speedups):.1f}x  "
          f"min={np.min(speedups):.1f}x")

    percentiles = [25, 50, 75, 90, 95, 99]
    print("\nPercentili errore u_x (H1):")
    for p in percentiles:
        print(f"  {p:3d}th : {np.percentile(errors_ux, p):.2e}")
    print("\nPercentili errore u_y (H1):")
    for p in percentiles:
        print(f"  {p:3d}th : {np.percentile(errors_uy, p):.2e}")
    print("\nPercentili errore p (L2):")
    for p in percentiles:
        print(f"  {p:3d}th : {np.percentile(errors_p, p):.2e}")
    print("\nPercentili speedup:")
    for p in percentiles:
        print(f"  {p:3d}th : {np.percentile(speedups, p):.1f}x")

    return {
        "errors_ux": errors_ux,
        "errors_uy": errors_uy,
        "errors_p":  errors_p,
        "times_fom": np.array(times_fom),
        "times_rom": np.array(times_rom),
        "speedups":  speedups,
        "mu0":       mu0_vals,
        "mu1":       mu1_vals,
    }


if __name__ == "__main__":
    W_train    = np.load("./data/snapshots_train.npy")
    param_test = np.load("./data/parameters_test.npy")

    results = compare_FOM_ROM(W_train, param_test)
