import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from setup_fem import speed_n_dofs
from build_basis import build_basis

mu_min = torch.tensor([0.1, 1.0], dtype=torch.float32)
mu_max = torch.tensor([10.0, 3.0], dtype=torch.float32)

def normalize(x):
    return 2.0 * (x - mu_min) / (mu_max - mu_min) - 1.0


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=4, nodes=128):
        super().__init__()
        layers = [nn.Linear(input_dim, nodes), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(nodes, nodes), nn.Tanh()]
        layers += [nn.Linear(nodes, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(normalize(x))


def print_mlp(input_dim, hidden_layers, nodes, output_dim):
    """Stampa l'architettura MLP in ASCII, senza figure."""
    sizes  = [input_dim] + [nodes] * hidden_layers + [output_dim]
    labels = (["input  (mu0, mu1)"] +
              [f"hidden {i+1}  [Tanh]" for i in range(hidden_layers)] +
              ["output (u_N)"])

    box_w = 26
    line  = "+" + "-" * (box_w - 2) + "+"

    print("\n" + "=" * box_w)
    print("POD-NN ARCHITECTURE".center(box_w))
    print("=" * box_w)

    for i, (s, lab) in enumerate(zip(sizes, labels)):
        print(line)
        print(f"| {lab:<{box_w - 4}} |")
        print(f"| {('dim = ' + str(s)):<{box_w - 4}} |")
        print(line)
        if i < len(sizes) - 1:
            print(f"{'|':>{box_w // 2}}")
            print(f"{('W' + str(i+1) + ' : ' + str(s) + ' -> ' + str(sizes[i+1])):^{box_w}}")
            print(f"{'v':>{box_w // 2}}")

    n_params = sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1))
    print("=" * box_w)
    print(f" total trainable params : {n_params:,}")
    print("=" * box_w + "\n")


def compute_targets(W_snap, B_us, B_p, inner_product_u):
    X_us     = B_us.T @ (inner_product_u @ B_us)
    X_pp     = B_p.T @ B_p
    targets  = []
    for j in range(W_snap.shape[1]):
        u_snap  = W_snap[:2 * speed_n_dofs, j]
        p_snap  = W_snap[2 * speed_n_dofs:, j]
        coeff_u = np.linalg.solve(X_us, B_us.T @ (inner_product_u @ u_snap))
        coeff_p = np.linalg.solve(X_pp, B_p.T @ p_snap)
        targets.append(np.concatenate([coeff_u, coeff_p]))
    return np.array(targets)


def train_PODNN(W_train, W_test, param_train, param_test,
                pod_tol=1.0 - 1.0e-6, N_max=100,
                hidden_layers=4, nodes=128,
                epoch_max=150000, lr=1e-3, lr_decay_epoch=20000,
                lr_decay=1e-4, tol=1e-5,
                weights_path="./models/podnn_weights.pt",
                results_dir="./results",
                seed=31):

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    torch.manual_seed(seed)

    # ── Base POD ──────────────────────────────────────────────────────────────
    B, pod_data     = build_basis(W_train, pod_tol=pod_tol, N_max=N_max, verbose=True)
    inner_product_u = pod_data["inner_product_u"]
    B_us            = np.concatenate([pod_data["V_u"], pod_data["V_s"]], axis=1)
    B_p             = pod_data["V_p"]
    np.save(os.path.join(results_dir, "pod_data.npy"), pod_data, allow_pickle=True)

    # ── Target ────────────────────────────────────────────────────────────────
    y_train    = compute_targets(W_train, B_us, B_p, inner_product_u)
    y_test     = compute_targets(W_test,  B_us, B_p, inner_product_u)
    output_dim = y_train.shape[1]

    # ── Rete ──────────────────────────────────────────────────────────────────
    net = Net(input_dim=2, output_dim=output_dim,
              hidden_layers=hidden_layers, nodes=nodes)
    print_mlp(2, hidden_layers, nodes, output_dim)

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    x_train_t = torch.tensor(np.float32(param_train))
    y_train_t = torch.tensor(np.float32(y_train))
    x_test_t  = torch.tensor(np.float32(param_test))
    y_test_t  = torch.tensor(np.float32(y_test))

    train_losses, test_losses = [], []

    pbar = tqdm(range(1, epoch_max + 1), desc="Training POD-NN")
    for epoch in pbar:
        net.train()
        optimizer.zero_grad()
        loss_val = loss_fn(net(x_train_t), y_train_t)
        loss_val.backward()
        optimizer.step()

        if epoch >= lr_decay_epoch:
            optimizer.param_groups[0]['lr'] = lr_decay

        if epoch % 500 == 0:
            net.eval()
            with torch.no_grad():
                loss_test = loss_fn(net(x_test_t), y_test_t).item()
            train_losses.append(loss_val.item())
            test_losses.append(loss_test)
            pbar.set_postfix(train=f"{loss_val.item():.2e}",
                             test=f"{loss_test:.2e}")

        if loss_val.item() < tol:
            print(f"Converged at epoch {epoch}, loss = {loss_val.item():.2e}")
            break

    # ── Salva ─────────────────────────────────────────────────────────────────
    torch.save({
        "model_state":   net.state_dict(),
        "output_dim":    output_dim,
        "hidden_layers": hidden_layers,
        "nodes":         nodes,
    }, weights_path)
    print(f"Weights saved → {weights_path}")

    np.save(os.path.join(results_dir, "training_curve.npy"),
            {"train_losses": train_losses, "test_losses": test_losses},
            allow_pickle=True)
    print(f"Training curve saved → {results_dir}/training_curve.npy")

    return net, B, train_losses, test_losses


if __name__ == "__main__":
    W_train     = np.load("./data/snapshots_train.npy")
    W_test      = np.load("./data/snapshots_test.npy")
    param_train = np.load("./data/parameters_train.npy")
    param_test  = np.load("./data/parameters_test.npy")
    train_PODNN(W_train, W_test, param_train, param_test)
