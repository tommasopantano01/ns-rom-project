import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from setup_fem import speed_n_dofs
from build_basis import build_basis


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, nodes):
        super().__init__()
        layers = [nn.Linear(input_dim, nodes), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(nodes, nodes), nn.Tanh()]
        layers += [nn.Linear(nodes, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def print_mlp(name, input_dim, hidden_layers, nodes, output_dim):
    n_params = sum((s_in + 1) * s_out for s_in, s_out in zip(
        [input_dim] + [nodes] * hidden_layers,
        [nodes] * hidden_layers + [output_dim]))
    print(f"POD-NN [{name:>9s}]:  input {input_dim} | {hidden_layers} hidden x {nodes} | "
          f"output {output_dim} | Tanh | {n_params:,} params")


# ── Target separati: velocità(+supremizer) e pressione ───────────────────────
def compute_targets_vel(W_snap, B_us, inner_product_u):
    targets = []
    for j in range(W_snap.shape[1]):
        u_snap  = W_snap[:2 * speed_n_dofs, j]
        coeff_u = B_us.T @ (inner_product_u @ u_snap)
        targets.append(coeff_u)
    return np.array(targets)


def compute_targets_p(W_snap, B_p):
    targets = []
    for j in range(W_snap.shape[1]):
        p_snap  = W_snap[2 * speed_n_dofs:, j]
        coeff_p = B_p.T @ p_snap
        targets.append(coeff_p)
    return np.array(targets)


# ── Loop di training, condiviso dalle due reti ───────────────────────────────
def _train_one(net, x_train_t, y_train_t, x_test_t, y_test_t,
               N_EPOCHS, LR, LR_2, EPOCH_LR, loss_fn, desc):
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)

    train_losses   = []
    test_losses    = []
    best_test      = float('inf')
    best_net_state = None

    pbar = tqdm(range(1, N_EPOCHS + 1), desc=f"Training POD-NN [{desc}]")
    for epoch in pbar:
        if epoch == EPOCH_LR:
            optimizer.param_groups[0]['lr'] = LR_2

        net.train()
        optimizer.zero_grad()
        loss = loss_fn(net(x_train_t), y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        optimizer.step()

        net.eval()
        with torch.no_grad():
            loss_test = loss_fn(net(x_test_t), y_test_t).item()

        train_losses.append(loss.item())
        test_losses.append(loss_test)

        if loss_test < best_test:
            best_test      = loss_test
            best_net_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % 200 == 0:
            pbar.set_postfix({"tr": f"{loss.item():.2e}",
                              "te": f"{loss_test:.2e}",
                              "best": f"{best_test:.2e}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.0e}"})

    net.load_state_dict(best_net_state)
    print(f"[{desc}] Best test loss: {best_test:.2e}  |  Final train loss: {train_losses[-1]:.2e}")
    return net, best_test, train_losses, test_losses


# ── Entry point principale ────────────────────────────────────────────────────
def train_PODNN(W_train, W_test, param_train, param_test,
                pod_tol, N_max,
                hidden_layers, nodes,
                N_EPOCHS, LR, LR_2, EPOCH_LR,
                weights_path="./models/podnn_weights.pt",
                results_dir="./results",
                seed=31):

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    loss_fn = nn.MSELoss()

    # ── Base POD (calcolata una sola volta, condivisa dalle due reti) ─────────
    B, pod_data     = build_basis(W_train, pod_tol, N_max, verbose=True)
    inner_product_u = pod_data["inner_product_u"]
    B_us            = np.concatenate([pod_data["V_u"], pod_data["V_s"]], axis=1)
    B_p             = pod_data["V_p"]
    np.save(os.path.join(results_dir, "pod_data.npy"), pod_data, allow_pickle=True)

    N_us = B_us.shape[1]
    N_p_ = B_p.shape[1]

    # ── Target ──────────────────────────────────────────────────────────────────
    y_train_vel = compute_targets_vel(W_train, B_us, inner_product_u)
    y_test_vel  = compute_targets_vel(W_test,  B_us, inner_product_u)
    y_train_p   = compute_targets_p(W_train, B_p)
    y_test_p    = compute_targets_p(W_test,  B_p)

    # ── Normalizzazione input (condivisa) ──────────────────────────────────────
    x_mean = param_train.mean(axis=0)
    x_std  = param_train.std(axis=0) + 1e-8

    x_train_t = torch.tensor(np.float32((param_train - x_mean) / x_std))
    x_test_t  = torch.tensor(np.float32((param_test  - x_mean) / x_std))

    # ── Normalizzazione target (separata per blocco) ───────────────────────────
    y_scale_vel = float(np.sqrt((y_train_vel**2).mean()))
    y_scale_p   = float(np.sqrt((y_train_p**2).mean()))

    y_train_vel_t = torch.tensor(np.float32(y_train_vel / y_scale_vel))
    y_test_vel_t  = torch.tensor(np.float32(y_test_vel  / y_scale_vel))
    y_train_p_t   = torch.tensor(np.float32(y_train_p / y_scale_p))
    y_test_p_t    = torch.tensor(np.float32(y_test_p  / y_scale_p))

    # ── Reti: stessa architettura, output diversi ──────────────────────────────
    torch.manual_seed(seed)
    net_vel = Net(input_dim=2, output_dim=N_us, hidden_layers=hidden_layers, nodes=nodes)
    print_mlp("velocità", 2, hidden_layers, nodes, N_us)

    net_p = Net(input_dim=2, output_dim=N_p_, hidden_layers=hidden_layers, nodes=nodes)
    print_mlp("pressione", 2, hidden_layers, nodes, N_p_)

    # ── Training: una rete dopo l'altra, stesso schema di iperparametri ───────
    net_vel, best_vel, train_losses_vel, test_losses_vel = _train_one(
        net_vel, x_train_t, y_train_vel_t, x_test_t, y_test_vel_t,
        N_EPOCHS, LR, LR_2, EPOCH_LR, loss_fn, desc="velocità")

    net_p, best_p, train_losses_p, test_losses_p = _train_one(
        net_p, x_train_t, y_train_p_t, x_test_t, y_test_p_t,
        N_EPOCHS, LR, LR_2, EPOCH_LR, loss_fn, desc="pressione")

    # ── Salva tutto in un unico checkpoint ──────────────────────────────────────
    torch.save({
        "model_state_vel": net_vel.state_dict(),
        "model_state_p":   net_p.state_dict(),
        "output_dim_vel":  N_us,
        "output_dim_p":    N_p_,
        "hidden_layers":   hidden_layers,
        "nodes":           nodes,
        "x_mean":          x_mean,
        "x_std":           x_std,
        "y_scale_vel":     y_scale_vel,
        "y_scale_p":       y_scale_p,
    }, weights_path)
    print(f"Weights saved → {weights_path}")

    np.save(os.path.join(results_dir, "training_curve.npy"),
        {
            "train_losses_vel": train_losses_vel,
            "test_losses_vel":  test_losses_vel,
            "train_losses_p":   train_losses_p,
            "test_losses_p":    test_losses_p,
            "N_EPOCHS":         N_EPOCHS,
            "LR":               LR,
            "LR_2":             LR_2,
            "EPOCH_LR":         EPOCH_LR,
        }, allow_pickle=True)

    return net_vel, net_p, B, x_mean, x_std, y_scale_vel, y_scale_p


if __name__ == "__main__":
    W_train     = np.load("./data/snapshots_train.npy")
    W_test      = np.load("./data/snapshots_test.npy")
    param_train = np.load("./data/parameters_train.npy")
    param_test  = np.load("./data/parameters_test.npy")
    train_PODNN(W_train, W_test, param_train, param_test)
