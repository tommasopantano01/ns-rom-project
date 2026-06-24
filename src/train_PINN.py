# src/train_PINN.py
import torch
import torch.nn as nn
import numpy as np
import time
import os
from tqdm import tqdm

from solve_PINN import PINN, pde_residuals, save_PINN, PI
from setup_fem import mesh, pressure_dofs_data, speed_dofs_data, speed_n_dofs


# ── Estrazione dati nodali dagli snapshot ─────────────────────────────────────

def _extract_nodal_data(W):
    """Da snapshot matrix (tot_dofs, N_snap) → coords, ux, uy, p su nodi P1."""
    N_verts = mesh.cell0_d_total_number()
    free_verts, p_idx_list, ux_idx_list = [], [], []
    for i in range(N_verts):
        p_dofs = pressure_dofs_data.cells_do_fs[0][i]
        u_dofs = speed_dofs_data.cells_do_fs[0][i]
        if len(p_dofs) > 0 and int(p_dofs[0].type) == 2:
            free_verts.append(i)
            p_idx_list.append(p_dofs[0].global_index)
            ux_idx_list.append(u_dofs[0].global_index if len(u_dofs) > 0 else -1)

    free_verts = np.array(free_verts)
    p_idx      = np.array(p_idx_list)
    ux_idx     = np.array(ux_idx_list)

    coords = np.array([[mesh.cell0_d_coordinate_x(i),
                        mesh.cell0_d_coordinate_y(i)] for i in free_verts],
                      dtype=np.float32)
    ux = W[ux_idx, :].astype(np.float32)
    uy = W[speed_n_dofs + ux_idx, :].astype(np.float32)
    p  = W[2*speed_n_dofs + p_idx, :].astype(np.float32)
    return coords, ux, uy, p


# ── Campionamento ─────────────────────────────────────────────────────────────

def _to_tensor(arr, device, requires_grad=False):
    return torch.tensor(arr, dtype=torch.float32,
                        requires_grad=requires_grad).to(device)

def _sample_interior(N, mu0_min, mu0_max, mu1_min, mu1_max, device):
    x   = np.random.uniform(0., 1., (N, 1))
    y   = np.random.uniform(0., 1., (N, 1))
    mu0 = np.random.uniform(mu0_min, mu0_max, (N, 1))
    mu1 = np.random.uniform(mu1_min, mu1_max, (N, 1))
    return (_to_tensor(x,   device, requires_grad=True),
            _to_tensor(y,   device, requires_grad=True),
            _to_tensor(mu0, device),
            _to_tensor(mu1, device))

def _sample_boundary(N, mu0_min, mu0_max, mu1_min, mu1_max, device):
    n = N // 4
    t = np.random.uniform(0., 1., (n, 1))
    sides = [
        np.hstack([t,                np.zeros((n, 1))]),
        np.hstack([t,                np.ones((n, 1)) ]),
        np.hstack([np.zeros((n, 1)), t               ]),
        np.hstack([np.ones((n, 1)),  t               ]),
    ]
    xy  = np.vstack(sides)
    mu0 = np.random.uniform(mu0_min, mu0_max, (xy.shape[0], 1))
    mu1 = np.random.uniform(mu1_min, mu1_max, (xy.shape[0], 1))
    return (_to_tensor(xy[:, 0:1], device),
            _to_tensor(xy[:, 1:2], device),
            _to_tensor(mu0,        device),
            _to_tensor(mu1,        device))


# ── Entry point ───────────────────────────────────────────────────────────────

def train_PINN(
    W_train, W_test, param_train, param_test,
    layers        = [4, 128, 128, 128, 128, 3],
    seed          = 42,
    mu0_min       = 0.1,  mu0_max = 10.0,
    mu1_min       = 1.0,  mu1_max =  3.0,
    n_pde         = 2000,
    n_bc          = 500,
    n_gauge       = 20,
    k_data        = 20,
    w_bc          = 10.0,
    w_div         = 3.0,
    w_data        = 2.0,
    n_pretrain    = 500,
    n_epochs_adam = 10000,
    lr_adam       = 1e-3,
    n_steps_lbfgs = 10000,
    train_split   = 0.8,
    split_seed    = 0,
    weights_path  = "./models/pinn_weights.pt",
    results_dir   = "./results",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Estrai dati nodali ────────────────────────────────────────────────────
    coords, ux_train, uy_train, p_train = _extract_nodal_data(W_train)
    _,      ux_test,  uy_test,  p_test  = _extract_nodal_data(W_test)
    N_nodes   = coords.shape[0]
    N_train   = param_train.shape[0]
    N_test    = param_test.shape[0]
    train_idx = np.arange(N_train)
    test_idx  = np.arange(N_test)
    print(f"Nodi P1: {N_nodes} | Train snap: {N_train} | Test snap: {N_test}")

    # ── Tensori su device ─────────────────────────────────────────────────────
    coords_t   = torch.tensor(coords, dtype=torch.float32)
    params_d   = torch.tensor(param_train, dtype=torch.float32).to(device)
    ux_nodes_d = torch.tensor(ux_train.T, dtype=torch.float32).to(device)  # (N_train, N_nodes)
    uy_nodes_d = torch.tensor(uy_train.T, dtype=torch.float32).to(device)
    p_nodes_d  = torch.tensor(p_train.T,  dtype=torch.float32).to(device)
    x_mesh_d   = coords_t[:, 0:1].to(device)
    y_mesh_d   = coords_t[:, 1:2].to(device)

    model = PINN(layers).to(device)
    mse   = nn.MSELoss()
    kw    = dict(mu0_min=mu0_min, mu0_max=mu0_max,
                 mu1_min=mu1_min, mu1_max=mu1_max, device=device)

    # ── Pre-training BC ───────────────────────────────────────────────────────
    opt_pre = torch.optim.Adam(model.parameters(), lr=lr_adam)
    print(f"Pre-training BC ({n_pretrain} epoche)...")
    for epoch in range(1, n_pretrain + 1):
        opt_pre.zero_grad()
        x_b, y_b, mu0_b, mu1_b = _sample_boundary(n_bc, **kw)
        out_b    = model(x_b, y_b, mu0_b, mu1_b)
        zeros_bc = torch.zeros(out_b.shape[0], 1, device=device)
        loss_bc  = mse(out_b[:, 0:1], zeros_bc) + mse(out_b[:, 1:2], zeros_bc)

        mu0_g = _to_tensor(np.random.uniform(mu0_min, mu0_max, (n_gauge, 1)), device)
        mu1_g = _to_tensor(np.random.uniform(mu1_min, mu1_max, (n_gauge, 1)), device)
        z_g   = torch.zeros(n_gauge, 1, device=device)
        loss_p = mse(model(z_g, z_g, mu0_g, mu1_g)[:, 2:3], z_g)

        (loss_bc + loss_p).backward()
        opt_pre.step()
    print(f"  bc={loss_bc.item():.3e}  p={loss_p.item():.3e}")

    # ── Training Adam ─────────────────────────────────────────────────────────────
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs_adam)
    t0 = time.time()
    pbar = tqdm(range(1, n_epochs_adam + 1), desc="Training PINN [Adam]")
    for epoch in pbar:
        optimizer.zero_grad()
    
        x_c, y_c, mu0_c, mu1_c = _sample_interior(n_pde, **kw)
        R1, R2, R3 = pde_residuals(model, x_c, y_c, mu0_c, mu1_c)
        z_pde    = torch.zeros(n_pde, 1, device=device)
        f_scale  = (mu1_c**2 * PI**2).detach().mean()
        loss_mom = mse(R1 / f_scale, z_pde) + mse(R2 / f_scale, z_pde)
        loss_div = mse(R3, z_pde)
    
        x_b, y_b, mu0_b, mu1_b = _sample_boundary(n_bc, **kw)
        out_b    = model(x_b, y_b, mu0_b, mu1_b)
        zeros_bc = torch.zeros(out_b.shape[0], 1, device=device)
        loss_bc  = mse(out_b[:, 0:1], zeros_bc) + mse(out_b[:, 1:2], zeros_bc)
    
        mu0_g = _to_tensor(np.random.uniform(mu0_min, mu0_max, (n_gauge, 1)), device)
        mu1_g = _to_tensor(np.random.uniform(mu1_min, mu1_max, (n_gauge, 1)), device)
        z_g   = torch.zeros(n_gauge, 1, device=device)
        loss_p = mse(model(z_g, z_g, mu0_g, mu1_g)[:, 2:3], z_g)
    
        local_idx = np.random.choice(N_train, k_data, replace=False)
        x_d   = x_mesh_d.repeat(k_data, 1)
        y_d   = y_mesh_d.repeat(k_data, 1)
        mu0_d = params_d[local_idx, 0:1].repeat_interleave(N_nodes, dim=0)
        mu1_d = params_d[local_idx, 1:2].repeat_interleave(N_nodes, dim=0)
        out_d = model(x_d, y_d, mu0_d, mu1_d)
        loss_data = (mse(out_d[:, 0:1], ux_nodes_d[local_idx].reshape(-1, 1)) +
                     mse(out_d[:, 1:2], uy_nodes_d[local_idx].reshape(-1, 1)) +
                     mse(out_d[:, 2:3],  p_nodes_d[local_idx].reshape(-1, 1)))
    
        loss = loss_mom + w_div * loss_div + w_bc * loss_bc + w_data * loss_data + loss_p
        loss.backward()
        optimizer.step()
        scheduler.step()
    
        if epoch % 200 == 0:
            pbar.set_postfix({
                "mom":  f"{loss_mom.item():.2e}",
                "div":  f"{loss_div.item():.2e}",
                "bc":   f"{loss_bc.item():.2e}",
                "data": f"{loss_data.item():.2e}",
            })
    
    print(f"Adam completato in {time.time()-t0:.1f} s")

    # ── Fine-tuning L-BFGS ───────────────────────────────────────────────────
    if n_steps_lbfgs > 0:
        BATCH_SNAP = 50
        n_batches  = N_train // BATCH_SNAP

        x_lc, y_lc, mu0_lc, mu1_lc = _sample_interior(n_pde, **kw)
        x_lb, y_lb, mu0_lb, mu1_lb = _sample_boundary(n_bc, **kw)
        z_pde_l = torch.zeros(n_pde,         1, device=device)
        z_bc_l  = torch.zeros(x_lb.shape[0], 1, device=device)
        x_all   = x_mesh_d.repeat(BATCH_SNAP, 1)
        y_all   = y_mesh_d.repeat(BATCH_SNAP, 1)

        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=20,
            history_size=50, tolerance_grad=1e-7,
            tolerance_change=1e-9, line_search_fn="strong_wolfe")

        step  = [0]
        t_lb  = time.time()
        print_every_lb = max(n_steps_lbfgs // 20, 1)

        def closure():
            opt_lbfgs.zero_grad()
            R1, R2, R3 = pde_residuals(model, x_lc, y_lc, mu0_lc, mu1_lc)
            f_sc = (mu1_lc**2 * PI**2).detach().mean()
            lm   = mse(R1/f_sc, z_pde_l) + mse(R2/f_sc, z_pde_l)
            ld   = mse(R3, z_pde_l)
            out_b = model(x_lb, y_lb, mu0_lb, mu1_lb)
            lb    = mse(out_b[:, 0:1], z_bc_l) + mse(out_b[:, 1:2], z_bc_l)

            ldata = torch.tensor(0.0, device=device)
            for b in range(n_batches):
                gi    = np.arange(b*BATCH_SNAP, (b+1)*BATCH_SNAP)
                mu0_d = params_d[gi, 0:1].repeat_interleave(N_nodes, dim=0)
                mu1_d = params_d[gi, 1:2].repeat_interleave(N_nodes, dim=0)
                out_d = model(x_all, y_all, mu0_d, mu1_d)
                ldata = ldata + (
                    mse(out_d[:, 0:1], ux_nodes_d[gi].reshape(-1, 1)) +
                    mse(out_d[:, 1:2], uy_nodes_d[gi].reshape(-1, 1)) +
                    mse(out_d[:, 2:3],  p_nodes_d[gi].reshape(-1, 1)))
            ldata = ldata / n_batches

            total = lm + w_div * ld + w_bc * lb + w_data * ldata
            total.backward()
            step[0] += 1
            if step[0] % print_every_lb == 0:
                print(f"L-BFGS {step[0]:5d}/{n_steps_lbfgs}  "
                      f"mom={lm.item():.3e}  data={ldata.item():.3e}  "
                      f"t={time.time()-t_lb:.0f}s")
            return total

        model.train()
        print(f"Fine-tuning L-BFGS ({n_steps_lbfgs} passi)...")
        for _ in range(n_steps_lbfgs):
            opt_lbfgs.step(closure)
            if step[0] >= n_steps_lbfgs:
                break
        print(f"L-BFGS completato in {time.time()-t_lb:.1f} s")

    # ── Salvataggio ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
    save_PINN(model, layers, train_idx, test_idx, weights_path)
    print(f"Salvato → {weights_path}")
