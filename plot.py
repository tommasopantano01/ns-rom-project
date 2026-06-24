import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata


def _save_or_show(fig, results_dir, fname):
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_eigenvalues(pod_data, results_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, lam, label, N_pod in [
        (axes[0], pod_data["lam_u"], r"velocity basis $V_u$",   pod_data["N_u"]),
        (axes[1], pod_data["lam_s"], r"supremizer basis $V_s$", pod_data["N_s"]),
        (axes[2], pod_data["lam_p"], r"pressure basis $V_p$",   pod_data["N_p"]),
    ]:
        lam_plot = lam[:400]
        ax.semilogy(range(1, len(lam_plot) + 1), lam_plot, lw=1.2, color="steelblue")
        ax.axvline(N_pod, linestyle="--", color="darkorange", lw=1.2, label=f"$N = {N_pod}$")
        ax.set_xlim(1, 400)
        ax.set_ylim(1e-14, lam[0] * 2)
        ax.set_xlabel("Mode index", fontsize=11)
        ax.set_ylabel(r"Eigenvalue $\lambda_n$", fontsize=11)
        ax.set_title(f"Eigenvalue decay — {label}", fontsize=12)
        ax.grid(True, which="both", alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10)
    plt.suptitle("POD eigenvalue spectra", fontsize=13)
    _save_or_show(fig, results_dir, "eigenvalues.png")


def plot_parameter_space(param_train, param_test, results_dir=None):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.scatter(param_train[:, 0], param_train[:, 1], s=8, alpha=0.5, label=f"train ({len(param_train)})")
    ax.scatter(param_test[:, 0],  param_test[:, 1],  s=8, alpha=0.5, label=f"test ({len(param_test)})")
    ax.set_xlabel(r"$\mu_0$", fontsize=11)
    ax.set_ylabel(r"$\mu_1$", fontsize=11)
    ax.set_title("Train/test parameter distribution", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle="--")
    _save_or_show(fig, results_dir, "parameter_space.png")


def plot_training_curve(train_losses_vel, test_losses_vel,
                        train_losses_p, test_losses_p,
                        N_EPOCHS, LR, LR_2, EPOCH_LR, results_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for ax, train_l, test_l, title in [
        (axes[0], train_losses_vel, test_losses_vel, "velocity (+ supremizer)"),
        (axes[1], train_losses_p,   test_losses_p,   "pressure"),
    ]:
        n_ep = len(train_l)
        ax.semilogy(range(1, n_ep + 1), train_l, lw=1.2, color="steelblue",  label="train")
        ax.semilogy(range(1, n_ep + 1), test_l,  lw=1.2, color="darkorange", label="test")
        if EPOCH_LR <= n_ep:
            ax.axvline(EPOCH_LR, color="gray", linestyle="--", lw=0.9,
                       label=f"lr: {LR:.0e} → {LR_2:.0e}")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(1, n_ep)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="both", alpha=0.15, linestyle="--")
        ax.legend(fontsize=10)
    plt.suptitle(f"POD-NN training — {N_EPOCHS} epochs  |  lr$_0$={LR}", fontsize=13)
    _save_or_show(fig, results_dir, "training_curve.png")


def plot_errors(results, method_name, results_dir=None):
    key    = "errors" if "errors_ux" in results else "err"
    err_ux = results[f"{key}_ux"]
    err_uy = results[f"{key}_uy"]
    err_p  = results[f"{key}_p"]
    params = results["params"]
    slug   = method_name.lower().replace("-", "").replace(" ", "_")

    # heatmap
    mu0_grid = np.linspace(0.1, 10.0, 50)
    mu1_grid = np.linspace(1.0,  3.0, 50)
    MU0, MU1 = np.meshgrid(mu0_grid, mu1_grid)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, vals, title in zip(axes,
        [err_ux, err_uy, err_p],
        [r"$\|u_x\|$ rel.", r"$\|u_y\|$ rel.", r"$\|p\|$ rel."]):
        Z  = griddata((params[:, 0], params[:, 1]), vals, (MU0, MU1), method="nearest")
        im = ax.pcolormesh(MU0, MU1, Z, cmap="RdYlGn_r", shading="auto")
        plt.colorbar(im, ax=ax, label="relative error")
        ax.set_xlabel(r"$\mu_0$", fontsize=11)
        ax.set_ylabel(r"$\mu_1$", fontsize=11)
        ax.set_title(title, fontsize=11)
    plt.suptitle(rf"{method_name} error over $\mathcal{{P}}$", fontsize=13)
    _save_or_show(fig, results_dir, f"errors_{slug}_heatmap.png")

    # percentili
    percentiles = [25, 50, 75, 90, 95, 99]
    labels      = [f"{p}th" for p in percentiles]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, errs, title in zip(axes,
        [err_ux, err_uy, err_p],
        [r"$u_x$", r"$u_y$", r"$p$"]):
        ax.bar(labels, [np.percentile(errs, p) for p in percentiles],
               color="steelblue", alpha=0.8, edgecolor="k")
        ax.set_yscale("log")
        ax.set_ylabel("Relative error", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(rf"{method_name} error percentiles", fontsize=13)
    _save_or_show(fig, results_dir, f"errors_{slug}_percentiles.png")


# ── Alias per main.py ─────────────────────────────────────────────────────────
def plot_errors_rom(results, results_dir=None):
    plot_errors(results, "POD-Galerkin", results_dir)

def plot_errors_podnn(results, results_dir=None):
    plot_errors(results, "POD-NN", results_dir)

def plot_errors_pinn(results, results_dir=None):
    plot_errors(results, "PINN", results_dir)
