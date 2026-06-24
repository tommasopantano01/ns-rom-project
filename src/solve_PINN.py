import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from pinn_model import PINN


def load_PINN(weights_path):
    ckpt    = torch.load(weights_path, map_location="cpu", weights_only=False)
    kw      = {k: ckpt[k] for k in ("mu0_min","mu0_max","mu1_min","mu1_max")}
    net_vel = PINN(ckpt["layers_vel"], **kw)
    net_p   = PINN(ckpt["layers_p"],   **kw)
    net_vel.load_state_dict(ckpt["state_vel"]); net_vel.eval()
    net_p.load_state_dict(ckpt["state_p"]);     net_p.eval()
    return net_vel, net_p, ckpt["train_idx"], ckpt["test_idx"]

def solve_PINN(mu0, mu1, net_vel, net_p, coords, device="cpu"):
    x_t   = torch.tensor(coords[:,0:1], dtype=torch.float32, device=device)
    y_t   = torch.tensor(coords[:,1:2], dtype=torch.float32, device=device)
    mu0_t = torch.full_like(x_t, mu0)
    mu1_t = torch.full_like(x_t, mu1)
    with torch.no_grad():
        vel = net_vel(x_t, y_t, mu0_t, mu1_t).cpu().numpy()  # (N,2)
        p   = net_p(x_t,   y_t, mu0_t, mu1_t).cpu().numpy()  # (N,1)
    return np.concatenate([vel, p], axis=1)  # (N,3)
