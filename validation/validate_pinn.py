import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from tqdm import tqdm
from solve_PINN import solve_PINN, load_PINN


def validate_pinn(coords, params, ux_nodes, uy_nodes, p_nodes,
                  net_vel, net_p, test_idx, device="cpu"):
    # solve_PINN(m0, m1, net_vel, net_p, coords, device)

    err_ux, err_uy, err_p, t_pinn = [], [], [], []

    def rel_l2(pred, ref):
        d = np.linalg.norm(ref)
        return np.linalg.norm(pred - ref) / d if d > 1e-14 else float("nan")

    for j in tqdm(test_idx, desc="Validating PINN"):
        m0, m1 = params[j]
        t0  = time.time()
        out = solve_PINN(m0, m1, net_vel, net_p, coords, device)
        t_pinn.append(time.time() - t0)
        err_ux.append(rel_l2(out[:,0], ux_nodes[:,j]))
        err_uy.append(rel_l2(out[:,1], uy_nodes[:,j]))
        err_p.append( rel_l2(out[:,2],  p_nodes[:,j]))

    err_ux = np.array(err_ux)
    err_uy = np.array(err_uy)
    err_p  = np.array(err_p)
    t_pinn = np.array(t_pinn)

    print(f"\n=== PINN validation — {len(test_idx)} test points ===")
    print(f"{'Component':<12} {'Mean':>10} {'Median':>10} {'95th':>10} {'Max':>10}")
    print("-" * 46)
    for label, errs in [("u_x (L2)", err_ux),
                        ("u_y (L2)", err_uy),
                        ("p  (L2)",  err_p)]:
        print(f"{label:<12} {np.mean(errs):>10.2e} {np.median(errs):>10.2e} "
              f"{np.percentile(errs,95):>10.2e} {np.max(errs):>10.2e}")
    print(f"\nMean PINN time: {np.mean(t_pinn)*1000:.3f} ms")

    return {
        "err_ux": err_ux, "err_uy": err_uy, "err_p": err_p,
        "t_pinn": t_pinn, "params": params[test_idx],
    }
