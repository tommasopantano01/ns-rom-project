import torch
import torch.nn as nn
import numpy as np

MU0_MIN, MU0_MAX = 0.1, 10.0
MU1_MIN, MU1_MAX = 1.0,  3.0
PI = torch.tensor(np.pi, dtype=torch.float32)


# ── Normalizzazione input ─────────────────────────────────────────────────────

def normalize_inputs(x, y, mu0, mu1):
    x_n   = 2.0 * x - 1.0
    y_n   = 2.0 * y - 1.0
    mu0_n = 2.0 * (mu0 - MU0_MIN) / (MU0_MAX - MU0_MIN) - 1.0
    mu1_n = mu1 - 2.0
    return x_n, y_n, mu0_n, mu1_n


# ── Architettura ──────────────────────────────────────────────────────────────

class PINN(nn.Module):
    """MLP con attivazione Tanh.
    Input:  (x, y, mu0, mu1)
    Output: (u_x, u_y, p)
    """
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            net += [nn.Linear(layers[i], layers[i + 1]), nn.Tanh()]
        net += [nn.Linear(layers[-2], layers[-1])]
        self.net = nn.Sequential(*net)

    def forward(self, x, y, mu0, mu1):
        x_n, y_n, mu0_n, mu1_n = normalize_inputs(x, y, mu0, mu1)
        return self.net(torch.cat([x_n, y_n, mu0_n, mu1_n], dim=1))


# ── Termine sorgente ──────────────────────────────────────────────────────────

def source_f(x, y, mu1):
    f1 = (-(mu1**3 * PI**2 * torch.cos(mu1**2 * PI * x) - mu1**2 * PI**2)
           * torch.sin(mu1 * PI * y) * torch.cos(mu1 * PI * y)
           + mu1 * PI * torch.cos(mu1 * PI * x) * torch.cos(mu1 * PI * y))
    f2 = (-(-mu1**3 * PI**2 * torch.cos(mu1**2 * PI * y) + mu1**2 * PI**2)
           * torch.sin(mu1 * PI * x) * torch.cos(mu1 * PI * x)
           - mu1 * PI * torch.sin(mu1 * PI * x) * torch.sin(mu1 * PI * y))
    return f1, f2


# ── Residui PDE ───────────────────────────────────────────────────────────────

def _grad(u, v):
    return torch.autograd.grad(
        u, v, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=True)[0]


def pde_residuals(model, x, y, mu0, mu1):
    out = model(x, y, mu0, mu1)
    ux, uy, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    ux_x = _grad(ux, x);  ux_y = _grad(ux, y)
    uy_x = _grad(uy, x);  uy_y = _grad(uy, y)
    p_x  = _grad(p,  x);  p_y  = _grad(p,  y)

    ux_xx = _grad(ux_x, x);  ux_yy = _grad(ux_y, y)
    uy_xx = _grad(uy_x, x);  uy_yy = _grad(uy_y, y)

    f1, f2 = source_f(x, y, mu1)

    R1 = -mu0 * (ux_xx + ux_yy) + ux * ux_x + uy * ux_y + p_x - f1
    R2 = -mu0 * (uy_xx + uy_yy) + ux * uy_x + uy * uy_y + p_y - f2
    R3 = ux_x + uy_y
    return R1, R2, R3


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_PINN(model, layers, train_idx, test_idx, path):
    torch.save({
        "layers":    layers,
        "state":     model.state_dict(),
        "train_idx": train_idx,
        "test_idx":  test_idx,
    }, path)
    print(f"PINN saved → {path}")


def load_PINN(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device)
    model = PINN(ck["layers"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["layers"], ck["train_idx"], ck["test_idx"]
