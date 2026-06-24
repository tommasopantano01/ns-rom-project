import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pinn_model import PINN, pde_residuals, MU0_MIN, MU0_MAX, MU1_MIN, MU1_MAX


def _to(arr, device, requires_grad=False):
    return torch.tensor(arr, dtype=torch.float32,
                        requires_grad=requires_grad).to(device)


def _sample_interior(N, device):
    x   = np.random.uniform(0., 1., (N, 1))
    y   = np.random.uniform(0., 1., (N, 1))
    mu0 = np.random.uniform(MU0_MIN, MU0_MAX, (N, 1))
    mu1 = np.random.uniform(MU1_MIN, MU1_MAX, (N, 1))
    return (_to(x, device, True), _to(y, device, True),
            _to(mu0, device), _to(mu1, device))


def _sample_boundary(N, device):
    n = N // 4
    t = np.random.uniform(0., 1., (n, 1))
    sides = [
        np.hstack([t,                np.zeros((n, 1))]),
        np.hstack([t,                np.ones((n, 1)) ]),
        np.hstack([np.zeros((n, 1)), t               ]),
        np.hstack([np.ones((n, 1)),  t               ]),
    ]
    xy  = np.vstack(sides)
    mu0 = np.random.uniform(MU0_MIN, MU0_MAX, (xy.shape[0], 1))
    mu1 = np.random.uniform(MU1_MIN, MU1_MAX, (xy.shape[0], 1))
    return (_to(xy[:,0:1], device), _to(xy[:,1:2], device),
            _to(mu0, device), _to(mu1, device))


def train_PINN(coords, params, ux_nodes, uy_nodes, p_nodes,
               layers, seed,
               n_pde, n_bc, n_gauge, k_data,
               w_bc, w_div, w_data,
               n_pretrain, n_epochs_adam, lr_adam,
               n_steps_lbfgs,
               train_split, split_seed,
               weights_path="./models/pinn_weights.pt",
               results_dir="./results"):

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Train/test split ──────────────────────────────────────────────────────
    rng       = np.random.default_rng(split_seed)
    N_snap    = params.shape[0]
    N_train   = int(train_split * N_snap)
    all_idx   = rng.permutation(N_snap)
    train_idx = all_idx[:N_train]
    test_idx  = all_idx[N_train:]
    print(f"Train: {N_train} snap  |  Test: {N_snap - N_train} snap")

    # ── Tensori su device ─────────────────────────────────────────────────────
    N_nodes    = coords.shape[0]
    x_mesh     = torch.tensor(coords[:,0:1], dtype=torch.float32, device=device)
    y_mesh     = torch.tensor(coords[:,1:2], dtype=torch.float32, device=device)
    params_d   = torch.tensor(params,        dtype=torch.float32, device=device)
    ux_nodes_d = torch.tensor(ux_nodes.T,    dtype=torch.float32, device=device)
    uy_nodes_d = torch.tensor(uy_nodes.T,    dtype=torch.float32, device=device)
    p_nodes_d  = torch.tensor(p_nodes.T,     dtype=torch.float32, device=device)

    mse = nn.MSELoss()
    PI  = torch.tensor(np.pi, dtype=torch.float32, device=device)

    # ── Modello ───────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    model    = PINN(layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PINN {layers}  |  {n_params:,} params")

    # ── Pre-training BC ───────────────────────────────────────────────────────
    opt_pre = torch.optim.Adam(model.parameters(), lr=lr_adam)
    print(f"Pre-training BC ({n_pretrain} epochs)...")
    for _ in range(n_pretrain):
        opt_pre.zero_grad()
        x_b, y_b, mu0_b, mu1_b = _sample_boundary(n_bc, device)
        out_b = model(x_b, y_b, mu0_b, mu1_b)
        z_bc  = torch.zeros(out_b.shape[0], 1, device=device)
        mu0_g = _to(np.random.uniform(MU0_MIN, MU0_MAX, (n_gauge,1)), device)
        mu1_g = _to(np.random.uniform(MU1_MIN, MU1_MAX, (n_gauge,1)), device)
        out_g = model(torch.zeros(n_gauge,1,device=device),
                      torch.zeros(n_gauge,1,device=device), mu0_g, mu1_g)
        loss  = (mse(out_b[:,0:1], z_bc) + mse(out_b[:,1:2], z_bc)
               + mse(out_g[:,2:3], torch.zeros(n_gauge,1,device=device)))
        loss.backward(); opt_pre.step()
    print(f"  pre-training done (loss={loss.item():.2e})")

    # ── Adam ──────────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs_adam)
    adam_losses = []

    pbar = tqdm(range(1, n_epochs_adam + 1), desc="PINN Adam")
    for epoch in pbar:
        optimizer.zero_grad()

        x_c, y_c, mu0_c, mu1_c = _sample_interior(n_pde, device)
        R1, R2, R3 = pde_residuals(model, x_c, y_c, mu0_c, mu1_c)
        z_pde    = torch.zeros(n_pde, 1, device=device)
        f_scale  = (mu1_c**2 * PI**2).detach().mean()
        loss_mom = mse(R1/f_scale, z_pde) + mse(R2/f_scale, z_pde)
        loss_div = mse(R3, z_pde)

        x_b, y_b, mu0_b, mu1_b = _sample_boundary(n_bc, device)
        out_b   = model(x_b, y_b, mu0_b, mu1_b)
        z_bc    = torch.zeros(out_b.shape[0], 1, device=device)
        loss_bc = mse(out_b[:,0:1], z_bc) + mse(out_b[:,1:2], z_bc)

        mu0_g  = _to(np.random.uniform(MU0_MIN, MU0_MAX, (n_gauge,1)), device)
        mu1_g  = _to(np.random.uniform(MU1_MIN, MU1_MAX, (n_gauge,1)), device)
        out_g  = model(torch.zeros(n_gauge,1,device=device),
                       torch.zeros(n_gauge,1,device=device), mu0_g, mu1_g)
        loss_p = mse(out_g[:,2:3], torch.zeros(n_gauge,1,device=device))

        local_idx = np.random.choice(N_train, k_data, replace=False)
        idx       = train_idx[local_idx]
        x_d   = x_mesh.repeat(k_data, 1)
        y_d   = y_mesh.repeat(k_data, 1)
        mu0_d = params_d[idx, 0:1].repeat_interleave(N_nodes, dim=0)
        mu1_d = params_d[idx, 1:2].repeat_interleave(N_nodes, dim=0)
        out_d = model(x_d, y_d, mu0_d, mu1_d)
        loss_data = (mse(out_d[:,0:1], ux_nodes_d[idx].reshape(-1,1)) +
                     mse(out_d[:,1:2], uy_nodes_d[idx].reshape(-1,1)) +
                     mse(out_d[:,2:3],  p_nodes_d[idx].reshape(-1,1)))

        loss = loss_mom + w_div*loss_div + w_bc*loss_bc + w_data*loss_data + loss_p
        loss.backward(); optimizer.step(); scheduler.step()
        adam_losses.append(loss.item())

        if epoch % 500 == 0:
            pbar.set_postfix(mom=f"{loss_mom.item():.2e}",
                             data=f"{loss_data.item():.2e}",
                             bc=f"{loss_bc.item():.2e}")

    print(f"Adam done — final loss: {adam_losses[-1]:.2e}")

    # ── L-BFGS ───────────────────────────────────────────────────────────────
    BATCH_SNAP = 50
    n_batches  = N_train // BATCH_SNAP
    x_lc, y_lc, mu0_lc, mu1_lc = _sample_interior(n_pde, device)
    x_lb, y_lb, mu0_lb, mu1_lb = _sample_boundary(n_bc, device)
    z_pde_l = torch.zeros(n_pde, 1, device=device)
    z_bc_l  = torch.zeros(x_lb.shape[0], 1, device=device)
    x_all   = x_mesh.repeat(BATCH_SNAP, 1)
    y_all   = y_mesh.repeat(BATCH_SNAP, 1)

    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=50, tolerance_grad=1e-7,
        tolerance_change=1e-9, line_search_fn="strong_wolfe")

    lbfgs_losses = []
    step = [0]

    def closure():
        opt_lbfgs.zero_grad()
        R1, R2, R3 = pde_residuals(model, x_lc, y_lc, mu0_lc, mu1_lc)
        f_sc  = (mu1_lc**2 * PI**2).detach().mean()
        lm    = mse(R1/f_sc, z_pde_l) + mse(R2/f_sc, z_pde_l)
        ld    = mse(R3, z_pde_l)
        out_b = model(x_lb, y_lb, mu0_lb, mu1_lb)
        lb    = mse(out_b[:,0:1], z_bc_l) + mse(out_b[:,1:2], z_bc_l)
        ldata = torch.tensor(0., device=device)
        for b in range(n_batches):
            gi    = train_idx[b*BATCH_SNAP:(b+1)*BATCH_SNAP]
            mu0_d = params_d[gi, 0:1].repeat_interleave(N_nodes, dim=0)
            mu1_d = params_d[gi, 1:2].repeat_interleave(N_nodes, dim=0)
            out_d = model(x_all, y_all, mu0_d, mu1_d)
            ldata = ldata + (
                mse(out_d[:,0:1], ux_nodes_d[gi].reshape(-1,1)) +
                mse(out_d[:,1:2], uy_nodes_d[gi].reshape(-1,1)) +
                mse(out_d[:,2:3],  p_nodes_d[gi].reshape(-1,1)))
        ldata = ldata / n_batches
        loss_l = lm + w_div*ld + w_bc*lb + w_data*ldata
        loss_l.backward()
        step[0] += 1
        lbfgs_losses.append(loss_l.item())
        if step[0] % 100 == 0:
            print(f"  L-BFGS {step[0]:4d}/{n_steps_lbfgs}  loss={loss_l.item():.2e}")
        return loss_l

    model.train()
    print(f"L-BFGS ({n_steps_lbfgs} steps)...")
    for _ in range(n_steps_lbfgs):
        opt_lbfgs.step(closure)
        if step[0] >= n_steps_lbfgs:
            break
    print(f"L-BFGS done — final loss: {lbfgs_losses[-1]:.2e}")

    # ── Salva ─────────────────────────────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "layers":      layers,
        "train_idx":   train_idx,
        "test_idx":    test_idx,
    }, weights_path)
    print(f"Weights saved → {weights_path}")

    np.save(os.path.join(results_dir, "pinn_training_curve.npy"),
            {"adam": adam_losses, "lbfgs": lbfgs_losses},
            allow_pickle=True)

    return model, train_idx, test_idx
