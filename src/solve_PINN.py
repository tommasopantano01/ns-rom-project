import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from pinn_model import PINN


def load_PINN(weights_path="./models/pinn_weights.pt"):
    ckpt  = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = PINN(ckpt["layers"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["train_idx"], ckpt["test_idx"]


def solve_PINN(mu0, mu1, model, coords, device="cpu"):
    x_t   = torch.tensor(coords[:,0:1], dtype=torch.float32, device=device)
    y_t   = torch.tensor(coords[:,1:2], dtype=torch.float32, device=device)
    mu0_t = torch.full_like(x_t, mu0)
    mu1_t = torch.full_like(x_t, mu1)
    with torch.no_grad():
        out = model(x_t, y_t, mu0_t, mu1_t).cpu().numpy()
    return out  # (N_nodes, 3): [ux, uy, p]
