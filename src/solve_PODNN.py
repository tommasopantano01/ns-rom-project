import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from train_PODNN import Net


def load_PODNN(weights_path="./models/podnn_weights.pt"):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    net_vel = Net(
        input_dim=2,
        output_dim=checkpoint["output_dim_vel"],
        hidden_layers=checkpoint["hidden_layers"],
        nodes=checkpoint["nodes"],
    )
    net_vel.load_state_dict(checkpoint["model_state_vel"])
    net_vel.eval()

    net_p = Net(
        input_dim=2,
        output_dim=checkpoint["output_dim_p"],
        hidden_layers=checkpoint["hidden_layers"],
        nodes=checkpoint["nodes"],
    )
    net_p.load_state_dict(checkpoint["model_state_p"])
    net_p.eval()

    x_mean      = checkpoint["x_mean"]
    x_std       = checkpoint["x_std"]
    y_scale_vel = checkpoint["y_scale_vel"]
    y_scale_p   = checkpoint["y_scale_p"]

    return net_vel, net_p, x_mean, x_std, y_scale_vel, y_scale_p


def solve_PODNN(mu0, mu1, net_vel, net_p, B, x_mean, x_std, y_scale_vel, y_scale_p):
    with torch.no_grad():
        x_raw = np.float32([[mu0, mu1]])
        x     = torch.tensor(((x_raw - x_mean) / x_std).astype(np.float32))

        coeff_vel = net_vel(x).numpy()[0] * y_scale_vel
        coeff_p   = net_p(x).numpy()[0]   * y_scale_p
        coeff = np.concatenate([coeff_vel, coeff_p])

    return B @ coeff
