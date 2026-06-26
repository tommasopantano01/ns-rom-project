import torch
import numpy as np
import time
import os


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

    print(f"\n=== PINN validation — {len(test_idx)} test points ===")
    print(f"{'Component':<12} {'Mean':>10} {'Median':>10} {'95th':>10} {'Max':>10}")
    print("-" * 46)
    for label, errs in [("u_x (L2)", err_ux),
                        ("u_y (L2)", err_uy),
                        ("p  (L2)",  err_p)]:
        print(f"{label:<12} {np.mean(errs):>10.2e} {np.median(errs):>10.2e} "
              f"{np.percentile(errs, 95):>10.2e} {np.max(errs):>10.2e}")

    print(f"\nMean PINN time : {pinn_t.mean()*1e3:.2f} ms")

    if fom_times is None:
        fom_path = os.path.join(".", "results", "fom_times.npy")
        if os.path.exists(fom_path):
            fom_times = np.load(fom_path)

    if fom_times is not None:
        sp = fom_times[list(test_idx)] / pinn_t
        print(f"Mean speedup vs FOM : {sp.mean():.0f}x  |  "
              f"median: {np.median(sp):.0f}x")

    return {
        "err_ux":     err_ux,
        "err_uy":     err_uy,
        "err_p":      err_p,
        "params":     param_test[test_idx],
        "pinn_times": pinn_t,
        "fom_times":  fom_times,
    }
