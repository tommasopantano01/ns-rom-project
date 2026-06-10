import argparse
import yaml
import numpy as np
import sys
import os

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
                                 "validate_rom", "validate_podnn", "plot"])

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
                                 "training_curve", "parameter_space", "all"])

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


# ── Helper ────────────────────────────────────────────────────────────────────

def _ask_enriched(data_dir, no_enriched=False):
    if not no_enriched:
        snap  = os.path.join(data_dir, "snapshots_train_enriched.npy")
        param = os.path.join(data_dir, "parameters_train_enriched.npy")
        if os.path.exists(snap) and os.path.exists(param):
            print("Usando snapshots enriched.")
            return snap, param
    print("Uso snapshots train normali.")
    return (os.path.join(data_dir, "snapshots_train.npy"),
            os.path.join(data_dir, "parameters_train.npy"))


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
        pod_tol      = c["pod"]["tol"],
        N_max        = c["pod"]["n_max"],
        hidden_layers= c["podnn"]["hidden_layers"],
        nodes        = c["podnn"]["nodes"],
        seed         = c["podnn"]["seed"],
        N_EPOCHS     = c["training"]["epoch_max"],
        LR           = c["training"]["lr"],
        LR_2         = c["training"]["lr_decay"],
        EPOCH_LR     = c["training"]["lr_decay_epoch"],
        weights_path = os.path.join(c["paths"]["models"], "podnn_weights.pt"),
        results_dir  = c["paths"]["results"],
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
    W_test     = np.load(os.path.join(c["paths"]["data"], "snapshots_test.npy"))
    param_test = np.load(os.path.join(c["paths"]["data"], "parameters_test.npy"))
    pod_data   = np.load(c["paths"]["pod_basis"], allow_pickle=True).item()
    B          = pod_data["B"]
    net, x_mean, x_std, y_scale = load_PODNN(
        os.path.join(c["paths"]["models"], "podnn_weights.pt"))
    results = validate_podnn(W_test, param_test, net, B, x_mean, x_std, y_scale)
    np.save(os.path.join(c["paths"]["results"], "results_podnn.npy"),
            results, allow_pickle=True)
    print(f"Results saved → {c['paths']['results']}/results_podnn.npy")


def run_plot(c, what):
    import matplotlib
    matplotlib.use("Agg")
    from plot import (plot_errors_rom, plot_errors_podnn,
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

    if what in ("training_curve", "all"):
        results = np.load(os.path.join(results_dir, "training_curve.npy"),
                          allow_pickle=True).item()
        plot_training_curve(results["train_losses"], results["test_losses"],
                            results_dir)

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
    elif args.mode == "plot":
        if args.what is None:
            print("Specifica --what: eigenvalues | errors_rom | "
                  "errors_podnn | training_curve | parameter_space | all")
            sys.exit(1)
        run_plot(config, args.what)
