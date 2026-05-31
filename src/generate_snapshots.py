import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from tqdm import tqdm
from solve_FOM import solve_FOM
from setup_fem import tot_dofs

mu0_range = [0.1, 10.0]
mu1_range = [1.0, 3.0]


def in_enrich_zone(params):
    """True per ogni parametro che cade in almeno una zona di enrichment."""
    return (
        (params[:, 0] <= 2.0) |
        (params[:, 0] >= 8.0) |
        (params[:, 1] >= 2.0)
    )


def generate_snapshots(n_base=1000, n_train=800, n_enrich=100,
                       seed_base=42, seed_enrich=123,
                       newton_tol=1e-6, max_iter=20,
                       data_dir="./data"):

    os.makedirs(data_dir, exist_ok=True)

    train_snap_path   = os.path.join(data_dir, "snapshots_train.npy")
    train_params_path = os.path.join(data_dir, "parameters_train.npy")
    test_snap_path    = os.path.join(data_dir, "snapshots_test.npy")
    test_params_path  = os.path.join(data_dir, "parameters_test.npy")

    # ── 1. carica pool completo o genera da zero ───────────────────────────────
    if os.path.exists(train_snap_path) and os.path.exists(test_snap_path):
        W_train_ex      = np.load(train_snap_path)
        params_train_ex = np.load(train_params_path)
        W_test_ex       = np.load(test_snap_path)
        params_test_ex  = np.load(test_params_path)

        W_pool      = np.concatenate([W_train_ex, W_test_ex], axis=1)
        params_pool = np.concatenate([params_train_ex, params_test_ex], axis=0)

        existing_num = W_pool.shape[1]
        print(f"Found {existing_num} existing snapshots.")

        if n_base <= existing_num:
            print(f"Requested {n_base} <= {existing_num}: no FOM needed.")
            W_base      = W_pool[:, :n_base]
            params_base = params_pool[:n_base]
        else:
            missing = n_base - existing_num
            print(f"Generating {missing} additional snapshots...")
            np.random.seed(seed_base)
            new_params = np.random.uniform(
                low=[mu0_range[0], mu1_range[0]],
                high=[mu0_range[1], mu1_range[1]],
                size=(missing, 2)
            )
            W_new = np.zeros((tot_dofs, missing))
            for j, (m0, m1) in tqdm(enumerate(new_params), total=missing,
                                     desc="New snapshots"):
                W_new[:, j], _ = solve_FOM(m0, m1,
                                           newton_tol=newton_tol,
                                           max_iterations=max_iter,
                                           verbose=False)
            W_base      = np.concatenate([W_pool, W_new], axis=1)
            params_base = np.concatenate([params_pool, new_params], axis=0)

    else:
        print(f"No existing snapshots. Generating {n_base} from scratch...")
        np.random.seed(seed_base)
        params_base = np.random.uniform(
            low=[mu0_range[0], mu1_range[0]],
            high=[mu0_range[1], mu1_range[1]],
            size=(n_base, 2)
        )
        W_base = np.zeros((tot_dofs, n_base))
        for j, (m0, m1) in tqdm(enumerate(params_base), total=n_base,
                                 desc="Base snapshots"):
            W_base[:, j], _ = solve_FOM(m0, m1,
                                        newton_tol=newton_tol,
                                        max_iterations=max_iter,
                                        verbose=False)
        W_pool      = W_base
        params_pool = params_base

    # ── 2. enrichment ─────────────────────────────────────────────────────────
    mask_enrich        = in_enrich_zone(params_pool)
    W_enrich_pool      = W_pool[:, mask_enrich]
    params_enrich_pool = params_pool[mask_enrich]

    print(f"\nEnrichment zones: {len(params_enrich_pool)} existing snapshots available.")

    if n_enrich <= len(params_enrich_pool):
        print(f"Recycling {n_enrich} existing enrichment snapshots: no FOM needed.")
        W_enrich      = W_enrich_pool[:, :n_enrich]
        params_enrich = params_enrich_pool[:n_enrich]
    else:
        missing_enrich = n_enrich - len(params_enrich_pool)
        print(f"Recycling {len(params_enrich_pool)} + "
              f"generating {missing_enrich} new enrichment snapshots...")
        np.random.seed(seed_enrich)
        n_each = missing_enrich // 3
        n_last = missing_enrich - 2 * n_each

        new_enrich_params = np.vstack([
            np.column_stack([np.random.uniform(0.1, 2.0,  n_each),
                             np.random.uniform(1.0, 3.0,  n_each)]),
            np.column_stack([np.random.uniform(8.0, 10.0, n_each),
                             np.random.uniform(1.0, 3.0,  n_each)]),
            np.column_stack([np.random.uniform(0.1, 10.0, n_last),
                             np.random.uniform(2.0, 3.0,  n_last)]),
        ])
        W_new_enrich = np.zeros((tot_dofs, missing_enrich))
        for j, (m0, m1) in tqdm(enumerate(new_enrich_params), total=missing_enrich,
                                 desc="New enrichment snapshots"):
            W_new_enrich[:, j], _ = solve_FOM(m0, m1,
                                              newton_tol=newton_tol,
                                              max_iterations=max_iter,
                                              verbose=False)
        W_enrich      = np.concatenate([W_enrich_pool, W_new_enrich], axis=1)
        params_enrich = np.concatenate([params_enrich_pool, new_enrich_params], axis=0)

    # ── 3. split e salva ──────────────────────────────────────────────────────
    n_train_actual = min(n_train, W_base.shape[1])
    W_train     = np.concatenate([W_base[:, :n_train_actual], W_enrich], axis=1)
    param_train = np.concatenate([params_base[:n_train_actual], params_enrich], axis=0)
    W_test      = W_base[:, n_train_actual:]
    param_test  = params_base[n_train_actual:]

    np.save(train_snap_path,   W_train)
    np.save(train_params_path, param_train)
    np.save(test_snap_path,    W_test)
    np.save(test_params_path,  param_test)

    print(f"\nTrain : {W_train.shape[1]} ({n_train_actual} base + {n_enrich} enriched)")
    print(f"Test  : {W_test.shape[1]}  (base only)")
    print(f"Saved → {data_dir}")


if __name__ == "__main__":
    generate_snapshots()
