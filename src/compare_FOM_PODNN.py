import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from setup_fem import speed_n_dofs, J_A
from solve_FOM import solve_FOM
from solve_PODNN import solve_PODNN
from build_basis import build_basis
from train_PODNN import load_PODNN


def compare_FOM_PODNN(param_test, net, B,
                      n_compare=None, seed=42):
    """
    Confronto FOM vs POD-NN su parametri di test.

    Parameters
    ----------
    param_test : array (N, 2)
    net        : rete caricata con load_PODNN()
    B          : base ridotta (tot_dofs, N_tot)
    n_compare  : int, opzionale — se None usa tutti i param_test
    seed       : int
    """
    np.random.seed(seed)

    if n_compare is not None:
        idx = np.random.choice(len(param_test), n_compare, replace=False)
        params = param_test[idx]
    else:
        params = param_test

    X_ux = J_A[:speed_n_dofs, :speed_n_dofs]
    X_uy = J_A[speed_n_dofs:2*speed_n_dofs, speed_n_dofs:2*speed_n_dofs]

    err_ux, err_uy, err_p = [], [], []
    t_fom, t_nn           = [], []

    for m0, m1 in tqdm(params, desc="FOM vs POD-NN"):

        t0 = time.time()
        U_f, _ = solve_FOM(m0, m1, verbose=False)
        t_fom.append(time.time() - t0)

        t0 = time.time()
        U_n = solve_PODNN(m0, m1, net, B)
        t_nn.append(time.time() - t0)

        e      = U_f - U_n
        e_ux   = e[:speed_n_dofs]
        e_uy   = e[speed_n_dofs:2*speed_n_dofs]
        e_p    = e[2*speed_n_dofs:]

        n_ux   = U_f[:speed_n_dofs]
        n_uy   = U_f[speed_n_dofs:2*speed_n_dofs]
        n_p    = U_f[2*speed_n_dofs:]

        err_ux.append(np.sqrt(e_ux @ (X_ux @ e_ux)) /
                      np.sqrt(n_ux @ (X_ux @ n_ux)))
        err_uy.append(np.sqrt(e_uy @ (X_uy @ e_uy)) /
                      np.sqrt(n_uy @ (X_uy @ n_uy)))
        err_p.append(np.linalg.norm(e_p) / np.linalg.norm(n_p))

    err_ux = np.array(err_ux)
    err_uy = np.array(err_uy)
    err_p  = np.array(err_p)
    t_fom  = np.array(t_fom)
    t_nn   = np.array(t_nn)

    # ── Stampa ───────────────────────────────────────────────────────────────
    print(f"\n=== FOM vs POD-NN — {len(params)} test points ===")
    print(f"{'Component':<12} {'Mean':>10} {'Median':>10} {'95th':>10} {'Max':>10}")
    print("-" * 46)
    for label, errs in [("u_x (H1)", err_ux),
                        ("u_y (H1)", err_uy),
                        ("p  (L2)",  err_p)]:
        print(f"{label:<12} {np.mean(errs):>10.2e} {np.median(errs):>10.2e} "
              f"{np.percentile(errs, 95):>10.2e} {np.max(errs):>10.2e}")

    print(f"\nMean FOM time    : {np.mean(t_fom):.2f} s")
    print(f"Mean POD-NN time : {np.mean(t_nn)*1000:.3f} ms")
    print(f"Mean speedup     : {np.mean(t_fom)/np.mean(t_nn):.0f}x")

    # ── Plot: istogrammi errori ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, errs, title in zip(
        axes,
        [err_ux, err_uy, err_p],
        [r"$\|u_x - u_{x,NN}\|_{H^1}/\|u_x\|_{H^1}$",
         r"$\|u_y - u_{y,NN}\|_{H^1}/\|u_y\|_{H^1}$",
         r"$\|p - p_{NN}\|_{L^2}/\|p\|_{L^2}$"]
    ):
        ax.hist(errs, bins=30, edgecolor="k", color="steelblue", alpha=0.8)
        ax.axvline(np.mean(errs),   color="red",    linestyle="--",
                   label=f"mean = {np.mean(errs):.2e}")
        ax.axvline(np.median(errs), color="orange", linestyle=":",
                   label=f"median = {np.median(errs):.2e}")
        ax.set_xlabel("Relative error")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(r"FOM vs POD-NN over $\mathcal{P}$", fontsize=13)
    plt.show()

    # ── Plot: scatter errore vs mu0, mu1 ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, pidx, plabel in zip(axes, [0, 1],
                                 [r"$\mu_0$ (viscosity)",
                                  r"$\mu_1$ (forcing)"]):
        for errs, marker, label in [(err_ux, "o", r"$u_x$"),
                                    (err_uy, "s", r"$u_y$"),
                                    (err_p,  "^", r"$p$")]:
            ax.semilogy(params[:, pidx], errs,
                        marker, ms=4, alpha=0.5, label=label)
        ax.set_xlabel(plabel)
        ax.set_ylabel("Relative error")
        ax.set_title(f"Error vs {plabel}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    plt.suptitle(r"POD-NN error distribution over $\mathcal{P}$", fontsize=13)
    plt.show()

    return {
        "err_ux": err_ux, "err_uy": err_uy, "err_p": err_p,
        "t_fom": t_fom, "t_nn": t_nn,
        "params": params,
    }


if __name__ == "__main__":
    W_train     = np.load("./data/snapshots_train.npy")
    param_test  = np.load("./data/parameters_test.npy")

    B, _  = build_basis(W_train, verbose=False)
    net   = load_PODNN("./models/podnn_weights.pt")

    results = compare_FOM_PODNN(param_test, net, B)
