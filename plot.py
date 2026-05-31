import numpy as np
import matplotlib.pyplot as plt
import os


# ── Eigenvalues ───────────────────────────────────────────────────────────────
def plot_eigenvalues(pod_data, results_dir=None):
    """
    pod_data : dict con chiavi lam_u, lam_s, lam_p, N_u, N_s, N_p
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for ax, lam, label, N_pod in [
        (axes[0], pod_data["lam_u"], r"velocity basis $V_u$",   pod_data["N_u"]),
        (axes[1], pod_data["lam_s"], r"supremizer basis $V_s$", pod_data["N_s"]),
        (axes[2], pod_data["lam_p"], r"pressure basis $V_p$",   pod_data["N_p"]),
    ]:
        ax.semilogy(range(1, len(lam) + 1), lam, "o-", ms=4)
        ax.axvline(N_pod, linestyle="--", label=f"Selected N = {N_pod}")
        ax.set_xlabel("Mode index")
        ax.set_ylabel(r"Eigenvalue $\lambda_n$")
        ax.set_title(f"Eigenvalue decay\n{label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

    plt.suptitle("POD eigenvalue decay", fontsize=13)
    _save_or_show(fig, results_dir, "eigenvalues.png")


# ── Parameter space ───────────────────────────────────────────────────────────
def plot_parameter_space(param_train, param_test, results_dir=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(param_train[:, 0], param_train[:, 1], s=15, label="train")
    ax.scatter(param_test[:, 0],  param_test[:, 1],  s=25, label="test")
    ax.set_xlabel(r"$\mu_0$ (viscosity)")
    ax.set_ylabel(r"$\mu_1$ (forcing)")
    ax.set_title("Train/test parameter distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, results_dir, "parameter_space.png")


# ── Training curve ────────────────────────────────────────────────────────────
def plot_training_curve(train_losses, test_losses, results_dir=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(500, (len(train_losses)) * 500 + 1, 500)
    ax.semilogy(epochs, train_losses, label="train")
    ax.semilogy(epochs, test_losses,  label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("POD-NN training curve")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, results_dir, "training_curve.png")


# ── Errors ROM ────────────────────────────────────────────────────────────────
def plot_errors_rom(results, results_dir=None):
    """
    results : dict con err_ux, err_uy, err_p, mu0, mu1
    """
    _plot_error_histograms(
        errors    = [results["errors_ux"], results["errors_uy"], results["errors_p"]],
        titles    = [r"$\|u_x - u_{x,ROM}\|_{H^1}/\|u_x\|_{H^1}$",
                     r"$\|u_y - u_{y,ROM}\|_{H^1}/\|u_y\|_{H^1}$",
                     r"$\|p - p_{ROM}\|_{L^2}/\|p\|_{L^2}$"],
        suptitle  = r"FOM vs ROM over $\mathcal{P}$",
        results_dir = results_dir,
        fname     = "errors_rom_hist.png"
    )
    _plot_error_scatter(
        errors    = [results["errors_ux"], results["errors_uy"], results["errors_p"]],
        params    = np.column_stack([results["mu0"], results["mu1"]]),
        suptitle  = r"ROM error distribution over $\mathcal{P}$",
        results_dir = results_dir,
        fname     = "errors_rom_scatter.png"
    )
    _plot_speedup(
        times_fom = results["times_fom"],
        times_rom = results["times_rom"],
        label_rom = "ROM",
        results_dir = results_dir,
        fname     = "speedup_rom.png"
    )


# ── Errors POD-NN ─────────────────────────────────────────────────────────────
def plot_errors_podnn(results, results_dir=None):
    """
    results : dict con err_ux, err_uy, err_p, params, t_fom, t_nn
    """
    _plot_error_histograms(
        errors    = [results["err_ux"], results["err_uy"], results["err_p"]],
        titles    = [r"$\|u_x - u_{x,NN}\|_{H^1}/\|u_x\|_{H^1}$",
                     r"$\|u_y - u_{y,NN}\|_{H^1}/\|u_y\|_{H^1}$",
                     r"$\|p - p_{NN}\|_{L^2}/\|p\|_{L^2}$"],
        suptitle  = r"FOM vs POD-NN over $\mathcal{P}$",
        results_dir = results_dir,
        fname     = "errors_podnn_hist.png"
    )
    _plot_error_scatter(
        errors    = [results["err_ux"], results["err_uy"], results["err_p"]],
        params    = results["params"],
        suptitle  = r"POD-NN error distribution over $\mathcal{P}$",
        results_dir = results_dir,
        fname     = "errors_podnn_scatter.png"
    )
    _plot_speedup(
        times_fom = results["t_fom"],
        times_rom = results["t_nn"],
        label_rom = "POD-NN",
        results_dir = results_dir,
        fname     = "speedup_podnn.png"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _plot_error_histograms(errors, titles, suptitle, results_dir, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, errs, title in zip(axes, errors, titles):
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
    plt.suptitle(suptitle, fontsize=13)
    _save_or_show(fig, results_dir, fname)


def _plot_error_scatter(errors, params, suptitle, results_dir, fname):
    err_ux, err_uy, err_p = errors
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
    plt.suptitle(suptitle, fontsize=13)
    _save_or_show(fig, results_dir, fname)


def _plot_speedup(times_fom, times_rom, label_rom, results_dir, fname):
    speedups = times_fom / times_rom
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.hist(speedups, bins=30, edgecolor="k", color="seagreen", alpha=0.8)
    ax.axvline(np.mean(speedups),   color="red",    linestyle="--",
               label=f"mean = {np.mean(speedups):.1f}x")
    ax.axvline(np.median(speedups), color="orange", linestyle=":",
               label=f"median = {np.median(speedups):.1f}x")
    ax.set_xlabel("Speedup")
    ax.set_ylabel("Count")
    ax.set_title(f"FOM / {label_rom} speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, results_dir, fname)


def _save_or_show(fig, results_dir, fname):
    """
    Se results_dir è specificato salva il plot, altrimenti lo mostra.
    """
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)
    else:
        plt.show()
