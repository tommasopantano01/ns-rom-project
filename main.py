import argparse
import yaml
import numpy as np
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "./src")
sys.path.insert(0, "./validation")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ns-rom — Navier-Stokes ROM methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["build_basis", "train_podnn",
                                 "validate_rom", "validate_podnn",
                                 "train_pinn", "validate_pinn", "plot"])

    # ── Newton ────────────────────────────────────────────────────────────────
    parser.add_argument("--newton_tol", type=float)
    parser.add_argument("--max_iter",   type=int)

    # ── POD ───────────────────────────────────────────────────────────────────
    parser.add_argument("--no_enriched", action="store_true",
                        help="Use base train snapshots instead of enriched")
    parser.add_argument("--pod_tol", type=float)
    parser.add_argument("--n_max",   type=int)

    # ── POD-NN architettura ───────────────────────────────────────────────────
    parser.add_argument("--hidden_layers", type=int)
    parser.add_argument("--nodes",         type=int)
    parser.add_argument("--seed_nn",       type=int)

    # ── POD-NN training ───────────────────────────────────────────────────────
    parser.add_argument("--epoch_max",      type=int)
    parser.add_argument("--lr",             type=float)
    parser.add_argument("--lr_decay",       type=float)
    parser.add_argument("--lr_decay_epoch", type=int)
    parser.add_argument("--train_tol",      type=float)

    # ── Plot ──────────────────────────────────────────────────────────────────
    parser.add_argument("--what", type=str,
                        choices=["eigenvalues", "errors_rom", "errors_podnn",
                                 "training_curve", "parameter_space",
                                 "errors_pinn", "all"])

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir",    type=str)
    parser.add_argument("--models_dir",  type=str)
    parser.add_argument("--results_dir", type=str)

    return parser.parse_args()


def merge(config, args):
    def ov(cfg_val, arg_val):
        return arg_val if arg_val is not None else cfg_val

    c = config
    c["newton"]["tol"]      = ov(c["newton"]["tol"],      args.newton_tol)
    c["newton"]["max_iter"] = ov(c["newton"]["max_iter"], args.max_iter)

    c["pod"]["tol"]   = ov(c["pod"]["tol"],   args.pod_tol)
    c["pod"]["n_max"] = ov(c["pod"]["n_max"], args.n_max)

    c["podnn"]["hidden_layers"] = ov(c["podnn"]["hidden_layers"], args.hidden_layers)
    c["podnn"]["nodes"]         = ov(c["podnn"]["nodes"],         args.nodes)
    c["podnn"]["seed"]          = ov(c["podnn"]["seed"],          args.seed_nn)

    c["training"]["epoch_max"]      = ov(c["training"]["epoch_max"],      args.epoch_max)
    c["training"]["lr"]             = ov(c["training"]["lr"],             args.lr)
    c["training"]["lr_decay"]       = ov(c["training"]["lr_decay"],       args.lr_decay)
    c["training"]["lr_decay_epoch"] = ov(c["training"]["lr_decay_epoch"], args.lr_decay_epoch)
    c["training"]["tol"]            = ov(c["training"]["tol"],            args.train_tol)

    if args.data_dir:    c["paths"]["data"]    = args.data_dir
    if args.models_dir:  c["paths"]["models"]  = args.models_dir
    if args.results_dir: c["paths"]["results"] = args.results_dir

    return c


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ask_enriched(data_dir, no_enriched=False):
    if not no_enriched:
        snap  = os.path.join(data_dir, "snapshots_train_enriched.npy")
        param = os.path.join(data_dir, "parameters_train_enriched.npy")
        if os.path.exists(snap) and os.path.exists(param):
            print("Using enriched snapshots.")
            return snap, param
    print("Using base train snapshots.")
    return (os.path.join(data_dir, "snapshots_train.npy"),
            os.path.join(data_dir, "parameters_train.npy"))


def _ensure_pinn_data(c):
    p     = c["pinn"]
    files = [p["coords"], p["params"], p["ux_nodes"], p["uy_nodes"], p["p_nodes"]]
    if all(os.path.exists(f) for f in files):
        return

    print("Extracting PINN data from existing snapshots...")
    from setup_fem import mesh, speed_n_dofs, pressure_n_dofs
    from scipy.interpolate import griddata

    # ── coordinate nodi P1 (= vertici mesh = Cell0Ds) ────────────────────────
    # Cell0DsCoordinates ha shape (3, N_vertices) anche in 2D
    coords_p1 = mesh.Cell0DsCoordinates[:2, :].T   # (N_nodes, 2)

    # ── carica snapshot esistenti ─────────────────────────────────────────────
    data_dir = c["paths"]["data"]
    W_train  = np.load(os.path.join(data_dir, "snapshots_train.npy"))   # (tot_dofs, N_train)
    W_test   = np.load(os.path.join(data_dir, "snapshots_test.npy"))    # (tot_dofs, N_test)
    p_train  = np.load(os.path.join(data_dir, "parameters_train.npy"))  # (N_train, 2)
    p_test   = np.load(os.path.join(data_dir, "parameters_test.npy"))   # (N_test, 2)

    W      = np.hstack([W_train, W_test])          # (tot_dofs, N_snap)
    params = np.vstack([p_train, p_test])           # (N_snap, 2)

    # ── estrai campi ──────────────────────────────────────────────────────────
    ux_p2 = W[:speed_n_dofs, :]                    # (N_p2, N_snap)
    uy_p2 = W[speed_n_dofs:2*speed_n_dofs, :]
    p_p1  = W[2*speed_n_dofs:, :]                  # (pressure_n_dofs, N_snap) — già P1 interni

    # ── coordinate DOF P2 (per interpolazione) ────────────────────────────────
    # I DOF P2 includono vertici + midpoint lati: usiamo le coordinate dalla mesh
    # tramite il p_strong che mappa i DOF forti → coordinate
    # Alternativa robusta: griddata dai DOF interni con coordinate noti
    # Per ora usiamo Cell0DsCoordinates per P1 e un'approssimazione per P2
    # TODO: sostituire con speed_dofs_data coordinate quando disponibili
    from setup_fem import speed_dofs_data
    # Prova a ottenere le coordinate P2 — se fallisce usa solo P1 subset
    try:
        coords_p2 = np.array([[dof.x, dof.y] for row in speed_dofs_data.cells_do_fs
                               for dof in row])   # potrebbe non funzionare
    except Exception:
        coords_p2 = None

    N_snap = W.shape[1]
    N_p1   = coords_p1.shape[0]

    if coords_p2 is not None:
        ux_nodes = np.zeros((N_p1, N_snap))
        uy_nodes = np.zeros((N_p1, N_snap))
        for j in range(N_snap):
            ux_nodes[:, j] = griddata(coords_p2, ux_p2[:, j], coords_p1, method="linear")
            uy_nodes[:, j] = griddata(coords_p2, uy_p2[:, j], coords_p1, method="linear")
    else:
        # fallback: prendi solo le prime N_p1 righe (approssimazione)
        print("WARNING: P2 coords not available, using truncation fallback")
        ux_nodes = ux_p2[:N_p1, :]
        uy_nodes = uy_p2[:N_p1, :]

    # pressione: aggiunge DOF forte (=0, gauge) per arrivare a N_p1 nodi
    p_nodes = np.vstack([p_p1, np.zeros((N_p1 - pressure_n_dofs, N_snap))])

    for path in files:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    np.save(p["coords"],   coords_p1)
    np.save(p["params"],   params)
    np.save(p["ux_nodes"], ux_nodes)
    np.save(p["uy_nodes"], uy_nodes)
    np.save(p["p_nodes"],  p_nodes)
    print(f"Done — {N_snap} snapshots, {N_p1} P1 nodes.")
    
# ── Runners ───────────────────────────────────────────────────────────────────

def run_build_basis(c, args):
    from build_basis import build_basis
    snap_path, _ = _ask_enriched(c["paths"]["data"], args.no_enriched)
    W_train      = np.load(snap_path)
    B, pod_data  = build_basis(W_train, c["pod"]["tol"], c["pod"]["n_max"])
    np.save(c["paths"]["pod_basis"], {"B": B, **pod_data}, allow_pickle=True)
    print(f"Basis saved → {c['paths']['pod_basis']}")


def run_train_podnn(c, args):
    from train_PODNN import train_PODNN
    snap_path, param_path = _ask_enriched(c["paths"]["data"], args.no_enriched)
    W_train     = np.load(snap_path)
    param_train = np.load(param_path)
    W_test      = np.load(os.path.join(c["paths"]["data"], "snapshots_test.npy"))
    param_test  = np.load(os.path.join(c["paths"]["data"], "parameters_test.npy"))
    train_PODNN(
        W_train, W_test, param_train, param_test,
        pod_tol       = c["pod"]["tol"],
        N_max         = c["pod"]["n_max"],
        hidden_layers = c["podnn"]["hidden_layers"],
        nodes         = c["podnn"]["nodes"],
        seed          = c["podnn"]["seed"],
        N_EPOCHS      = c["training"]["epoch_max"],
        LR            = c["training"]["lr"],
        LR_2          = c["training"]["lr_decay"],
        EPOCH_LR      = c["training"]["lr_decay_epoch"],
        weights_path  = os.path.join(c["paths"]["models"], "podnn_weights.pt"),
        results_dir   = c["paths"]["results"],
    )


def run_validate_rom(c):
    from validate_rom import validate_rom
    W_test     = np.load(os.path.join(c["paths"]["data"], "snapshots_test.npy"))
    param_test = np.load(os.path.join(c["paths"]["data"], "parameters_test.npy"))
    results    = validate_rom(W_test, param_test)
    np.save(os.path.join(c["paths"]["results"], "results_rom.npy"),
            results, allow_pickle=True)
    print(f"Results saved → {c['paths']['results']}/results_rom.npy")


def run_validate_podnn(c):
    from validate_podnn import validate_podnn
    from solve_PODNN import load_PODNN
    from setup_fem import speed_n_dofs, tot_dofs

    W_test     = np.load(os.path.join(c["paths"]["data"], "snapshots_test.npy"))
    param_test = np.load(os.path.join(c["paths"]["data"], "parameters_test.npy"))
    pod_data   = np.load(os.path.join(c["paths"]["results"], "pod_data.npy"),
                         allow_pickle=True).item()
    N_u = pod_data["N_u"]
    N_p = pod_data["N_p"]
    B   = np.zeros((tot_dofs, N_u + N_p))
    B[:2*speed_n_dofs, :N_u] = pod_data["V_u"]
    B[2*speed_n_dofs:, N_u:] = pod_data["V_p"]
    net_vel, net_p, x_mean, x_std, y_scale_vel, y_scale_p = load_PODNN(
        os.path.join(c["paths"]["models"], "podnn_weights.pt"))
    results = validate_podnn(W_test, param_test, net_vel, net_p, B,
                             x_mean, x_std, y_scale_vel, y_scale_p)
    np.save(os.path.join(c["paths"]["results"], "results_podnn.npy"),
            results, allow_pickle=True)
    print(f"Results saved → {c['paths']['results']}/results_podnn.npy")


def run_train_pinn(c):
    from train_PINN import train_PINN
    _ensure_pinn_data(c)
    p        = c["pinn"]
    coords   = np.load(p["coords"])
    params   = np.load(p["params"])
    ux_nodes = np.load(p["ux_nodes"])
    uy_nodes = np.load(p["uy_nodes"])
    p_nodes  = np.load(p["p_nodes"])
    train_PINN(
        coords, params, ux_nodes, uy_nodes, p_nodes,
        layers_vel    = p["layers_vel"],
        layers_p      = p["layers_p"],
        seed          = p["seed"],
        mu0_min       = c["domain"]["mu0_min"],
        mu0_max       = c["domain"]["mu0_max"],
        mu1_min       = c["domain"]["mu1_min"],
        mu1_max       = c["domain"]["mu1_max"],
        n_pde         = p["n_pde"],
        n_bc          = p["n_bc"],
        n_gauge       = p["n_gauge"],
        k_data        = p["k_data"],
        w_bc          = p["w_bc"],
        w_div         = p["w_div"],
        w_data        = p["w_data"],
        n_pretrain    = p["n_pretrain"],
        n_epochs_adam = p["n_epochs_adam"],
        lr_adam       = p["lr_adam"],
        n_steps_lbfgs = p["n_steps_lbfgs"],
        train_split   = p["train_split"],
        split_seed    = p["split_seed"],
        weights_path  = os.path.join(c["paths"]["models"], "pinn_weights.pt"),
        results_dir   = c["paths"]["results"],
    )


def run_validate_pinn(c):
    from validate_pinn import validate_pinn
    from solve_PINN import load_PINN
    p        = c["pinn"]
    coords   = np.load(p["coords"])
    params   = np.load(p["params"])
    ux_nodes = np.load(p["ux_nodes"])
    uy_nodes = np.load(p["uy_nodes"])
    p_nodes  = np.load(p["p_nodes"])
    net_vel, net_p, _, test_idx = load_PINN(
        os.path.join(c["paths"]["models"], "pinn_weights.pt"))
    results = validate_pinn(coords, params, ux_nodes, uy_nodes, p_nodes,
                            net_vel, net_p, test_idx)
    np.save(os.path.join(c["paths"]["results"], "results_pinn.npy"),
            results, allow_pickle=True)
    print(f"Results saved → {c['paths']['results']}/results_pinn.npy")


def run_plot(c, what):
    import matplotlib
    matplotlib.use("Agg")
    from plot import (plot_errors_rom, plot_errors_podnn, plot_errors_pinn,
                      plot_training_curve, plot_parameter_space,
                      plot_eigenvalues)
    results_dir = c["paths"]["results"]
    data_dir    = c["paths"]["data"]

    if what in ("errors_rom", "all"):
        results = np.load(os.path.join(results_dir, "results_rom.npy"),
                          allow_pickle=True).item()
        plot_errors_rom(results, results_dir)

    if what in ("errors_podnn", "all"):
        results = np.load(os.path.join(results_dir, "results_podnn.npy"),
                          allow_pickle=True).item()
        plot_errors_podnn(results, results_dir)

    if what in ("errors_pinn", "all"):
        results = np.load(os.path.join(results_dir, "results_pinn.npy"),
                          allow_pickle=True).item()
        plot_errors_pinn(results, results_dir)

    if what in ("training_curve", "all"):
        results = np.load(os.path.join(results_dir, "training_curve.npy"),
                          allow_pickle=True).item()
        plot_training_curve(
            results["train_losses_vel"], results["test_losses_vel"],
            results["train_losses_p"],   results["test_losses_p"],
            results["N_EPOCHS"], results["LR"], results["LR_2"], results["EPOCH_LR"],
            results_dir
        )

    if what in ("parameter_space", "all"):
        plot_parameter_space(
            np.load(os.path.join(data_dir, "parameters_train.npy")),
            np.load(os.path.join(data_dir, "parameters_test.npy")),
            results_dir
        )

    if what in ("eigenvalues", "all"):
        pod_data = np.load(os.path.join(results_dir, "pod_data.npy"),
                           allow_pickle=True).item()
        plot_eigenvalues(pod_data, results_dir)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)
    config = merge(config, args)

    os.makedirs(config["paths"]["data"],    exist_ok=True)
    os.makedirs(config["paths"]["models"],  exist_ok=True)
    os.makedirs(config["paths"]["results"], exist_ok=True)

    if args.mode == "build_basis":
        run_build_basis(config, args)
    elif args.mode == "train_podnn":
        run_train_podnn(config, args)
    elif args.mode == "validate_rom":
        run_validate_rom(config)
    elif args.mode == "validate_podnn":
        run_validate_podnn(config)
    elif args.mode == "train_pinn":
        run_train_pinn(config)
    elif args.mode == "validate_pinn":
        run_validate_pinn(config)
    elif args.mode == "plot":
        if args.what is None:
            print("Please, specify --what: eigenvalues | errors_rom | errors_podnn | "
                  "errors_pinn | training_curve | parameter_space | all")
            sys.exit(1)
        run_plot(config, args.what)
