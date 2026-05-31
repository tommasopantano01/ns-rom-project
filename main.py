import argparse
import yaml
import numpy as np
import sys
import os

sys.path.insert(0, "./src")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ns-rom-comparison — Navier-Stokes ROM methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate", "train_podnn",
                                 "compare_rom", "compare_podnn", "plot"],
                        help="What to run")

    # ── Snapshots ─────────────────────────────────────────────────────────────
    parser.add_argument("--n_base",    type=int,   help="Total uniform snapshots")
    parser.add_argument("--n_train",   type=int,   help="Training snapshots (uniform)")
    parser.add_argument("--n_enrich",  type=int,   help="Enrichment snapshots")
    parser.add_argument("--seed_base", type=int,   help="Seed for uniform sampling")
    parser.add_argument("--seed_enrich", type=int, help="Seed for enrichment sampling")

    # ── Newton ────────────────────────────────────────────────────────────────
    parser.add_argument("--newton_tol",  type=float, help="Newton tolerance")
    parser.add_argument("--max_iter",    type=int,   help="Newton max iterations")

    # ── POD ───────────────────────────────────────────────────────────────────
    parser.add_argument("--pod_tol", type=float, help="POD energy tolerance")
    parser.add_argument("--n_max",   type=int,   help="POD max modes")

    # ── POD-NN architettura ───────────────────────────────────────────────────
    parser.add_argument("--hidden_layers", type=int, help="POD-NN hidden layers")
    parser.add_argument("--nodes",         type=int, help="POD-NN nodes per layer")
    parser.add_argument("--seed_nn",       type=int, help="POD-NN random seed")

    # ── POD-NN training ───────────────────────────────────────────────────────
    parser.add_argument("--epoch_max",      type=int,   help="Max training epochs")
    parser.add_argument("--lr",             type=float, help="Initial learning rate")
    parser.add_argument("--lr_decay",       type=float, help="Learning rate after decay")
    parser.add_argument("--lr_decay_epoch", type=int,   help="Epoch at which lr decays")
    parser.add_argument("--train_tol",      type=float, help="Training loss tolerance")

    # ── Confronto ─────────────────────────────────────────────────────────────
    parser.add_argument("--n_compare",  type=int, help="Test points for comparison")
    parser.add_argument("--seed_compare", type=int, help="Seed for comparison sampling")

    # ── Plot ──────────────────────────────────────────────────────────────────
    parser.add_argument("--what", type=str,
                        choices=["eigenvalues", "errors_rom",
                                 "errors_podnn", "training_curve",
                                 "parameter_space", "all"],
                        help="What to plot (only with --mode plot)")

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir",    type=str, help="Data directory")
    parser.add_argument("--models_dir",  type=str, help="Models directory")
    parser.add_argument("--results_dir", type=str, help="Results directory")

    return parser.parse_args()


def merge(config, args):
    """
    Sovrascrive i valori del config con quelli passati da CLI
    solo se esplicitamente forniti (non None).
    """
    def override(cfg_val, arg_val):
        return arg_val if arg_val is not None else cfg_val

    c = config
    c["snapshots"]["n_base"]       = override(c["snapshots"]["n_base"],      args.n_base)
    c["snapshots"]["n_train"]      = override(c["snapshots"]["n_train"],     args.n_train)
    c["snapshots"]["n_enrich"]     = override(c["snapshots"]["n_enrich"],    args.n_enrich)
    c["snapshots"]["seed_base"]    = override(c["snapshots"]["seed_base"],   args.seed_base)
    c["snapshots"]["seed_enrich"]  = override(c["snapshots"]["seed_enrich"], args.seed_enrich)

    c["newton"]["tol"]      = override(c["newton"]["tol"],      args.newton_tol)
    c["newton"]["max_iter"] = override(c["newton"]["max_iter"], args.max_iter)

    c["pod"]["tol"]   = override(c["pod"]["tol"],   args.pod_tol)
    c["pod"]["n_max"] = override(c["pod"]["n_max"], args.n_max)

    c["podnn"]["hidden_layers"] = override(c["podnn"]["hidden_layers"], args.hidden_layers)
    c["podnn"]["nodes"]         = override(c["podnn"]["nodes"],         args.nodes)
    c["podnn"]["seed"]          = override(c["podnn"]["seed"],          args.seed_nn)

    c["training"]["epoch_max"]      = override(c["training"]["epoch_max"],      args.epoch_max)
    c["training"]["lr"]             = override(c["training"]["lr"],             args.lr)
    c["training"]["lr_decay"]       = override(c["training"]["lr_decay"],       args.lr_decay)
    c["training"]["lr_decay_epoch"] = override(c["training"]["lr_decay_epoch"], args.lr_decay_epoch)
    c["training"]["tol"]            = override(c["training"]["tol"],            args.train_tol)

    c["compare"]["n_compare"] = override(c["compare"]["n_compare"], args.n_compare)
    c["compare"]["seed"]      = override(c["compare"]["seed"],      args.seed_compare)

    if args.data_dir:
        c["paths"]["data"]    = args.data_dir
    if args.models_dir:
        c["paths"]["models"]  = args.models_dir
    if args.results_dir:
        c["paths"]["results"] = args.results_dir

    return c


# ══════════════════════════════════════════════════════════════════════════════

def run_generate(c):
    from generate_snapshots import generate_snapshots
    generate_snapshots(
        n_base        = c["snapshots"]["n_base"],
        n_train       = c["snapshots"]["n_train"],
        n_enrich      = c["snapshots"]["n_enrich"],
        seed_base     = c["snapshots"]["seed_base"],
        seed_enrich   = c["snapshots"]["seed_enrich"],
        newton_tol    = c["newton"]["tol"],
        max_iter      = c["newton"]["max_iter"],
        data_dir      = c["paths"]["data"],
    )


def run_train_podnn(c):
    from train_PODNN import train_PODNN
    W_train     = np.load(os.path.join(c["paths"]["data"], "snapshots_train.npy"))
    W_test      = np.load(os.path.join(c["paths"]["data"], "snapshots_test.npy"))
    param_train = np.load(os.path.join(c["paths"]["data"], "parameters_train.npy"))
    param_test  = np.load(os.path.join(c["paths"]["data"], "parameters_test.npy"))

    train_PODNN(
        W_train, W_test, param_train, param_test,
        pod_tol        = c["pod"]["tol"],
        N_max          = c["pod"]["n_max"],
        hidden_layers  = c["podnn"]["hidden_layers"],
        nodes          = c["podnn"]["nodes"],
        seed           = c["podnn"]["seed"],
        epoch_max      = c["training"]["epoch_max"],
        lr             = c["training"]["lr"],
        lr_decay       = c["training"]["lr_decay"],
        lr_decay_epoch = c["training"]["lr_decay_epoch"],
        tol            = c["training"]["tol"],
        weights_path   = os.path.join(c["paths"]["models"], "podnn_weights.pt"),
    )


def run_compare_rom(c):
    from compare_FOM_ROM import compare_FOM_ROM
    from build_basis import build_basis
    W_train     = np.load(os.path.join(c["paths"]["data"], "snapshots_train.npy"))
    param_test  = np.load(os.path.join(c["paths"]["data"], "parameters_test.npy"))

    results = compare_FOM_ROM(
        W_train, param_test,
        pod_tol    = c["pod"]["tol"],
        N_max      = c["pod"]["n_max"],
        newton_tol = c["newton"]["tol"],
        max_iter   = c["newton"]["max_iter"],
    )
    np.save(os.path.join(c["paths"]["results"], "results_rom.npy"), results)


def run_compare_podnn(c):
    from compare_FOM_PODNN import compare_FOM_PODNN
    from build_basis import build_basis
    from solve_PODNN import load_PODNN
    W_train    = np.load(os.path.join(c["paths"]["data"], "snapshots_train.npy"))
    param_test = np.load(os.path.join(c["paths"]["data"], "parameters_test.npy"))

    B, _  = build_basis(W_train, pod_tol=c["pod"]["tol"],
                        N_max=c["pod"]["n_max"], verbose=False)
    net   = load_PODNN(os.path.join(c["paths"]["models"], "podnn_weights.pt"))

    results = compare_FOM_PODNN(
        param_test, net, B,
        n_compare = c["compare"]["n_compare"],
        seed      = c["compare"]["seed"],
    )
    np.save(os.path.join(c["paths"]["results"], "results_podnn.npy"), results)


def run_plot(c, what):
    from plot import (plot_eigenvalues, plot_errors_rom,
                      plot_errors_podnn, plot_training_curve,
                      plot_parameter_space)
    results_dir = c["paths"]["results"]

    if what in ("eigenvalues", "all"):
        plot_eigenvalues(results_dir)
    if what in ("errors_rom", "all"):
        plot_errors_rom(results_dir)
    if what in ("errors_podnn", "all"):
        plot_errors_podnn(results_dir)
    if what in ("training_curve", "all"):
        plot_training_curve(results_dir)
    if what in ("parameter_space", "all"):
        plot_parameter_space(
            np.load(os.path.join(c["paths"]["data"], "parameters_train.npy")),
            np.load(os.path.join(c["paths"]["data"], "parameters_test.npy")),
        )


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)
    config = merge(config, args)

    os.makedirs(config["paths"]["data"],    exist_ok=True)
    os.makedirs(config["paths"]["models"],  exist_ok=True)
    os.makedirs(config["paths"]["results"], exist_ok=True)

    if args.mode == "generate":
        run_generate(config)

    elif args.mode == "train_podnn":
        run_train_podnn(config)

    elif args.mode == "compare_rom":
        run_compare_rom(config)

    elif args.mode == "compare_podnn":
        run_compare_podnn(config)

    elif args.mode == "plot":
        if args.what is None:
            print("Specifica --what: eigenvalues | errors_rom | "
                  "errors_podnn | training_curve | parameter_space | all")
            sys.exit(1)
        run_plot(config, args.what)
