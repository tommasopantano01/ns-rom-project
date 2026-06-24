import torch
import numpy as np
import time

def validate_pinn(W_test, param_test, model, test_idx, fom_times=None):
    from train_PINN import _extract_nodal_data

    coords, ux_nodes, uy_nodes, p_nodes = _extract_nodal_data(W_test)

    device = next(model.parameters()).device
    model.eval()
    x_eval = torch.tensor(coords[:, 0:1], dtype=torch.float32).to(device)
    y_eval = torch.tensor(coords[:, 1:2], dtype=torch.float32).to(device)

    def rel_err(pred, ref):
        denom = np.linalg.norm(ref)
        return np.linalg.norm(pred - ref) / denom if denom > 1e-14 else float("nan")

    err_ux, err_uy, err_p, pinn_t = [], [], [], []
    with torch.no_grad():
        for j in test_idx:
            m0, m1 = param_test[j]
            mu0_ev = torch.full_like(x_eval, m0)
            mu1_ev = torch.full_like(x_eval, m1)
            t0  = time.time()
            out = model(x_eval, y_eval, mu0_ev, mu1_ev).cpu().numpy()
            pinn_t.append(time.time() - t0)
            err_ux.append(rel_err(out[:, 0], ux_nodes[:, j]))
            err_uy.append(rel_err(out[:, 1], uy_nodes[:, j]))
            err_p.append( rel_err(out[:, 2],  p_nodes[:, j]))

    err_ux = np.array(err_ux)
    err_uy = np.array(err_uy)
    err_p  = np.array(err_p)
    pinn_t = np.array(pinn_t)

    print(f"\n=== PINN validation ({len(test_idx)} snapshot test) ===")
    print(f"  e_ux medio : {err_ux.mean():.3e}")
    print(f"  e_uy medio : {err_uy.mean():.3e}")
    print(f"  e_p  medio : {err_p.mean():.3e}")
    print(f"  t_PINN medio: {pinn_t.mean()*1e3:.2f} ms")
    if fom_times is not None:
        sp = fom_times[list(test_idx)] / pinn_t
        print(f"  Speedup medio: {sp.mean():.0f}x  |  mediano: {np.median(sp):.0f}x")

    return {
        "err_ux":     err_ux,
        "err_uy":     err_uy,
        "err_p":      err_p,
        "params":     param_test[test_idx],
        "pinn_times": pinn_t,
    }
