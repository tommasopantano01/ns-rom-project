import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from train_PODNN import Net
from build_basis import build_basis


def load_PODNN(weights_path="./models/podnn_weights.pt"):
    """
    Carica la rete e restituisce la net pronta per l'inference.
    """
    checkpoint = torch.load(weights_path, map_location="cpu")

    net = Net(
        input_dim=2,
        output_dim=checkpoint["output_dim"],
        hidden_layers=checkpoint["hidden_layers"],
        nodes=checkpoint["nodes"],
    )
    net.load_state_dict(checkpoint["model_state"])
    net.eval()
    return net


def solve_PODNN(mu0, mu1, net, B):
    """
    Forward pass online: nessun assemblaggio, nessun sistema lineare.

    Parameters
    ----------
    mu0, mu1 : float
    net      : rete caricata con load_PODNN()
    B        : base ridotta (tot_dofs, N_tot)

    Returns
    -------
    U : array (tot_dofs,)
    """
    with torch.no_grad():
        x     = torch.tensor(np.float32([[mu0, mu1]]))
        coeff = net(x).numpy()[0]

    return B @ coeff
