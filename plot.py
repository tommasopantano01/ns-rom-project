import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata


# ── Utils ─────────────────────────────────────────────────────────────────────
def _save_or_show(fig, results_dir, fname):
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)
    else:
        plt.show()


# ── Eigenvalues ───────────────────────────────────────────────────────────────
def plot_eigenvalues(pod_data, results_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for ax, lam, label, N_pod in [
        (axes[0], pod_data["lam_u"], r"velocity basis $V_u$",   pod_data["N_u"]),
        (axes[1], pod_data["lam_s"], r"supremizer basis $V_s$", pod_data["N_s"]),
        (axes[2], pod_data["lam_p"], r"pressure basis $V_p$",   pod_data["N_p"]),
    ]:
        lam_plot = lam[:400]
        ax.semilogy(range(1, len(lam_plot) + 1), lam_plot, lw=1.2, color="steelblue")
        ax.axvline(N_pod, linestyle="--", color="darkorange", lw=1.2,
                   label=f"$N = {N_pod}$")
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


# ── Parameter space ───────────────────────────────────────────────────────────
def plot_parameter_space(param_train, param_test, results_dir=None):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.scatter(param_train[:, 0], param_train[:, 1], s=8,  alpha=0.5,
               label=f"train ({len(param_train)})")
    ax.scatter(param_test[:, 0],  param_test[:, 1],  s=8,  alpha=0.5,
               label=f"test ({len(param_test)})")
    ax.set_xlabel(r"$\mu_0$", fontsize=11)
    ax.set_ylabel(r"$\mu_1$", fontsize=11)
    ax.set_title("Train/test parameter distribution", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle="--")
    _save_or_show(fig, results_dir, "parameter_space.png")


# ── Training curve ────────────────────────────────────────────────────────────
def plot_training_curve(train_losses, test_losses, N_EPOCHS, LR, LR_2,
                        EPOCH_LR, results_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    epochs_ax = range(1, N_EPOCHS + 1)
    ax.semilogy(epochs_ax, train_losses, lw=1.2, color="steelblue",  label="train")
    ax.semilogy(epochs_ax, test_losses,  lw=1.2, color="darkorange", label="test")
    ax.axvline(EPOCH_LR, color="gray", linestyle="--", lw=0.9,
               label=f"lr: {LR:.0e} → {LR_2:.0e}")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title(f"POD-NN training  —  {N_EPOCHS} epochs  |  lr$_0$={LR}", fontsize=12)
    ax.set_xlim(1, N_EPOCHS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="both", alpha=0.15, linestyle="--")
    ax.legend(fontsize=10)
    _save_or_show(fig, results_dir, "training_curve.png")


# ── Heatmap errori ────────────────────────────────────────────────────────────
def plot_error_heatmap(err_ux, err_uy, err_p, params, suptitle, results_dir=None,
                       fname="error_heatmap.png"):
    mu0_grid = np.linspace(0.1, 10.0, 50)
    mu1_grid = np.linspace(1.0,  3.0, 50)
    MU0, MU1 = np.meshgrid(mu0_grid, mu1_grid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    def _heatmap(ax, vals, title):
        Z  = griddata((params[:, 0], params[:, 1]), vals,
                      (MU0, MU1), method="nearest")
        im = ax.pcolormesh(MU0, MU1, Z, cmap="RdYlGn_r", shading="auto")
        plt.colorbar(im, ax=ax, label="relative error")
        ax.set_xlabel(r"$\mu_0$ viscosity", fontsize=11)
        ax.set_ylabel(r"$\mu_1$ forcing",   fontsize=11)
        ax.set_title(title, fontsize=11)

    _heatmap(axes[0], err_ux, r"$\|u_x - u_{x,N}\|_{H^1}/\|u_x\|_{H^1}$")
    _heatmap(axes[1], err_uy, r"$\|u_y - u_{y,N}\|_{H^1}/\|u_y\|_{H^1}$")
    _heatmap(axes[2], err_p,  r"$\|p - p_N\|_{L^2}/\|p\|_{L^2}$")

    plt.suptitle(suptitle, fontsize=13)
    _save_or_show(fig, results_dir, fname)


# ── Percentili errori ─────────────────────────────────────────────────────────
def plot_error_percentiles(err_ux, err_uy, err_p, suptitle, results_dir=None,
                           fname="error_percentiles.png"):
    percentiles = [25, 50, 75, 90, 95, 99]
    labels      = [f"{p}th" for p in percentiles]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    for ax, errs, title in zip(
        axes,
        [err_ux, err_uy, err_p],
        [r"$u_x$ ($H^1$)", r"$u_y$ ($H^1$)", r"$p$ ($L^2$)"]
    ):
        vals = [np.percentile(errs, p) for p in percentiles]
        ax.bar(labels, vals, color="steelblue", alpha=0.8, edgecolor="k")
        ax.set_yscale("log")
        ax.set_ylabel("Relative error", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(suptitle, fontsize=13)
    _save_or_show(fig, results_dir, fname)


# ── Entry points per il main ──────────────────────────────────────────────────
def plot_errors_rom(results, results_dir=None):
    params = np.column_stack([results["mu0"], results["mu1"]])
    plot_error_heatmap(
        results["errors_ux"], results["errors_uy"], results["errors_p"],
        params,
        suptitle    = r"POD-Galerkin error over $\mathcal{P}$",
        results_dir = results_dir,
        fname       = "errors_rom_heatmap.png"
    )
    plot_error_percentiles(
        results["errors_ux"], results["errors_uy"], results["errors_p"],
        suptitle    = r"POD-Galerkin error percentiles",
        results_dir = results_dir,
        fname       = "errors_rom_percentiles.png"
    )


def plot_errors_podnn(results, results_dir=None):
    plot_error_heatmap(
        results["err_ux"], results["err_uy"], results["err_p"],
        results["params"],
        suptitle    = r"POD-NN error over $\mathcal{P}$",
        results_dir = results_dir,
        fname       = "errors_podnn_heatmap.png"
    )
    plot_error_percentiles(
        results["err_ux"], results["err_uy"], results["err_p"],
        suptitle    = r"POD-NN error percentiles",
        results_dir = results_dir,
        fname       = "errors_podnn_percentiles.png"
    )
