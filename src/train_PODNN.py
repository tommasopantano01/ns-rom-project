import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
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


def plot_mlp(input_dim, hidden_layers, nodes, output_dim):
    layer_sizes = [input_dim] + [nodes] * hidden_layers + [output_dim]
    layer_names = (["Input\n$\\mu=(\\mu_0,\\mu_1)$"] +
                   [f"Hidden {i+1}\nTanh" for i in range(hidden_layers)] +
                   ["Output\n$\\mathbf{u}_N$"])

    n_layers          = len(layer_sizes)
    fig_w             = max(14, n_layers * 2.2)
    fig, ax           = plt.subplots(figsize=(fig_w, 5))
    ax.axis("off")
    max_nodes_display = 6
    node_r            = 0.18
    x_coords          = np.linspace(0.5, fig_w - 0.5, n_layers)
    colors            = ["#4393c3"] + ["#f4a261"] * hidden_layers + ["#2a9d8f"]

    for li, (x, size, name, col) in enumerate(
            zip(x_coords, layer_sizes, layer_names, colors)):

        display_n = min(size, max_nodes_display)
        y_coords  = np.linspace(-(display_n - 1) / 2,
                                 (display_n - 1) / 2, display_n)

        if li > 0:
            prev_x    = x_coords[li - 1]
            prev_size = min(layer_sizes[li - 1], max_nodes_display)
            prev_y    = np.linspace(-(prev_size - 1) / 2,
                                     (prev_size - 1) / 2, prev_size)
            for yy in prev_y:
                for yy2 in y_coords:
                    ax.annotate("", xy=(x - node_r, yy2),
                                xytext=(prev_x + node_r, yy),
                                arrowprops=dict(arrowstyle="-",
                                                color="#bbbbbb", lw=0.5))

        for y in y_coords:
            ax.add_patch(plt.Circle((x, y), node_r, color=col,
                                    ec="white", lw=1.5, zorder=3))

        if size > max_nodes_display:
            ax.text(x, 0, "⋮", ha="center", va="center",
                    fontsize=14, color="gray", zorder=4)

        ax.text(x, -(display_n - 1) / 2 - 0.55, f"{size}",
                ha="center", va="top", fontsize=9,
                color="#333333", fontweight="bold")
        ax.text(x, (display_n - 1) / 2 + 0.55, name,
                ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xlim(0, fig_w)
    ax.set_ylim(-max_nodes_display / 2 - 1.5, max_nodes_display / 2 + 1.5)
    ax.set_title("POD-NN architecture", fontsize=12, pad=4)
    plt.tight_layout()
    plt.show()


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
    net      = Net(input_dim=2, output_dim=output_dim,
                   hidden_layers=hidden_layers, nodes=nodes)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Network: 2 → {nodes}×{hidden_layers} → {output_dim} "
          f"| Tanh | {n_params:,} parameters")
    plot_mlp(2, hidden_layers, nodes, output_dim)

    # ── Training setup ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    x_train_t = torch.tensor(np.float32(param_train))
    y_train_t = torch.tensor(np.float32(y_train))
    x_test_t  = torch.tensor(np.float32(param_test))
    y_test_t  = torch.tensor(np.float32(y_test))

    train_losses, test_losses, epochs_log = [], [], []

    # live plot
    plt.ion()
    fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.set_title("Training curve")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, which="both", alpha=0.3)
    line_tr, = ax_loss.plot([], [], label="train", color="#4393c3")
    line_te, = ax_loss.plot([], [], label="test",  color="#d6604d")
    ax_loss.legend()
    plt.tight_layout()
    plt.show()

    # ── Loop ──────────────────────────────────────────────────────────────────
    pbar = tqdm(range(1, epoch_max + 1), desc="Training POD-NN")
    for epoch in pbar:
        net.train()
        optimizer.zero_grad()
        loss_val = loss_fn(net(x_train_t), y_train_t)
        loss_val.backward()
        optimizer.step()

        if epoch >= lr_decay_epoch:
            optimizer.param_groups[0]['lr'] = lr_decay

        if epoch % 2000 == 0:
            net.eval()
            with torch.no_grad():
                loss_test = loss_fn(net(x_test_t), y_test_t).item()
            train_losses.append(loss_val.item())
            test_losses.append(loss_test)
            epochs_log.append(epoch)
            pbar.set_postfix(train=f"{loss_val.item():.2e}",
                             test=f"{loss_test:.2e}")

            line_tr.set_data(epochs_log, train_losses)
            line_te.set_data(epochs_log, test_losses)
            ax_loss.relim()
            ax_loss.autoscale_view()
            fig_loss.canvas.draw()
            fig_loss.canvas.flush_events()

        if loss_val.item() < tol:
            print(f"Converged at epoch {epoch}, loss = {loss_val.item():.2e}")
            break

    plt.ioff()

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
