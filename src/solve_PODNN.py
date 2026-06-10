import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from train_PODNN import Net
from build_basis import build_basis

def load_PODNN(weights_path="./models/podnn_weights.pt"):
    checkpoint = torch.load(weights_path, map_location="cpu")
    net = Net(
        input_dim=2,
        output_dim=checkpoint["output_dim"],
        hidden_layers=checkpoint["hidden_layers"],
        nodes=checkpoint["nodes"],
    )
    net.load_state_dict(checkpoint["model_state"])
    net.eval()
    x_mean  = checkpoint["x_mean"]
    x_std   = checkpoint["x_std"]
    y_scale = checkpoint["y_scale"]
    return net, x_mean, x_std, y_scale


def solve_PODNN(mu0, mu1, net, B, x_mean, x_std, y_scale):
    with torch.no_grad():
        x_raw = np.float32([[mu0, mu1]])
        x     = torch.tensor(((x_raw - x_mean) / x_std).astype(np.float32))
        coeff = net(x).numpy()[0] * y_scale
    return B @ coeff
