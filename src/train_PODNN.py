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


def print_mlp(input_dim, hidden_layers, nodes, output_dim):
    n_params = sum((s_in + 1) * s_out for s_in, s_out in zip(
        [input_dim] + [nodes] * hidden_layers,
        [nodes] * hidden_layers + [output_dim]))
    print(f"POD-NN:  input {input_dim} | {hidden_layers} hidden x {nodes} | "
          f"output {output_dim} | Tanh | {n_params:,} params")


def compute_targets(W_snap, B_us, B_p, inner_product_u):
    targets = []
    for j in range(W_snap.shape[1]):
        u_snap  = W_snap[:2 * speed_n_dofs, j]
        p_snap  = W_snap[2 * speed_n_dofs:, j]
        coeff_u = B_us.T @ (inner_product_u @ u_snap)
        coeff_p = B_p.T @ p_snap
        targets.append(np.concatenate([coeff_u, coeff_p]))
    return np.array(targets)


def train_PODNN(W_train, W_test, param_train, param_test,
                pod_tol, N_max,
                hidden_layers, nodes,
                N_EPOCHS, LR, LR_2, EPOCH_LR,
                weights_path="./models/podnn_weights.pt",
                results_dir="./results",
                seed=31):

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    torch.manual_seed(seed)

    # ── Base POD ──────────────────────────────────────────────────────────────
    B, pod_data     = build_basis(W_train, pod_tol, N_max, verbose=True)
    inner_product_u = pod_data["inner_product_u"]
    B_us            = np.concatenate([pod_data["V_u"], pod_data["V_s"]], axis=1)
    B_p             = pod_data["V_p"]
    np.save(os.path.join(results_dir, "pod_data.npy"), pod_data, allow_pickle=True)

    # ── Target ────────────────────────────────────────────────────────────────
    y_train    = compute_targets(W_train, B_us, B_p, inner_product_u)
    y_test     = compute_targets(W_test,  B_us, B_p, inner_product_u)
    output_dim = y_train.shape[1]

    # ── Normalizzazione ───────────────────────────────────────────────────────
    x_mean  = param_train.mean(axis=0)
    x_std   = param_train.std(axis=0) + 1e-8
    y_scale = float(np.sqrt((y_train**2).mean()))

    x_train_t = torch.tensor(np.float32((param_train - x_mean) / x_std))
    x_test_t  = torch.tensor(np.float32((param_test  - x_mean) / x_std))
    y_train_t = torch.tensor(np.float32(y_train / y_scale))
    y_test_t  = torch.tensor(np.float32(y_test  / y_scale))

    # ── Rete ──────────────────────────────────────────────────────────────────
    net = Net(input_dim=2, output_dim=output_dim,
              hidden_layers=hidden_layers, nodes=nodes)
    print_mlp(2, hidden_layers, nodes, output_dim)

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer      = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn        = nn.MSELoss()
    train_losses   = []
    test_losses    = []
    best_test      = float('inf')
    best_net_state = None

    pbar = tqdm(range(1, N_EPOCHS + 1), desc="Training POD-NN")
    for epoch in pbar:
        if epoch == EPOCH_LR:
            optimizer.param_groups[0]['lr'] = LR_2

        net.train()
        optimizer.zero_grad()
        loss = loss_fn(net(x_train_t), y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
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
            pbar.write(f"epoch {epoch}/{N_EPOCHS} | train={loss.item():.2e} | "
                       f"test={loss_test:.2e} | lr={optimizer.param_groups[0]['lr']:.0e}")

    net.load_state_dict(best_net_state)
    print(f"\nBest test loss: {best_test:.2e}  |  Final train loss: {train_losses[-1]:.2e}")

    # ── Salva ─────────────────────────────────────────────────────────────────
    torch.save({
        "model_state":   net.state_dict(),
        "output_dim":    output_dim,
        "hidden_layers": hidden_layers,
        "nodes":         nodes,
        "x_mean":        x_mean,
        "x_std":         x_std,
        "y_scale":       y_scale,
    }, weights_path)
    print(f"Weights saved → {weights_path}")

    np.save(os.path.join(results_dir, "training_curve.npy"),
        {
            "train_losses": train_losses,
            "test_losses":  test_losses,
            "N_EPOCHS":     N_EPOCHS,
            "LR":           LR,
            "LR_2":         LR_2,
            "EPOCH_LR":     EPOCH_LR,
        }, allow_pickle=True)

    return net, B, train_losses, test_losses, x_mean, x_std, y_scale


if __name__ == "__main__":
    W_train     = np.load("./data/snapshots_train.npy")
    W_test      = np.load("./data/snapshots_test.npy")
    param_train = np.load("./data/parameters_train.npy")
    param_test  = np.load("./data/parameters_test.npy")
    train_PODNN(W_train, W_test, param_train, param_test)
