import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from tqdm import tqdm
from solve_FOM import solve_FOM
from setup_fem import tot_dofs

mu0_range = [0.1, 10.0]
mu1_range = [1.0, 3.0]


def generate_snapshots(n_base=650, n_enrich=0,
                       test_ratio=0.2,
                       seed_base=42, seed_enrich=123,
                       newton_tol=1e-6, max_iter=20,
                       data_dir="./data"):

    os.makedirs(data_dir, exist_ok=True)

    train_snap_path   = os.path.join(data_dir, "snapshots_train.npy")
    train_params_path = os.path.join(data_dir, "parameters_train.npy")
    test_snap_path    = os.path.join(data_dir, "snapshots_test.npy")
    test_params_path  = os.path.join(data_dir, "parameters_test.npy")

    # ── 1. carica o genera snapshot uniformi ──────────────────────────────────
    if os.path.exists(train_snap_path) and os.path.exists(test_snap_path):
        W_train_ex      = np.load(train_snap_path)
        params_train_ex = np.load(train_params_path)
        W_test          = np.load(test_snap_path)
        params_test     = np.load(test_params_path)

        W_uniform      = np.concatenate([W_train_ex, W_test], axis=1)
        params_uniform = np.concatenate([params_train_ex, params_test], axis=0)
        n_base         = W_uniform.shape[1]
        print(f"Found {n_base} existing snapshots "
              f"({W_train_ex.shape[1]} train, {W_test.shape[1]} test).")

    else:
        print(f"No existing data. Generating {n_base} snapshots from scratch...")
        np.random.seed(seed_base)
        params_uniform = np.random.uniform(
            low =[mu0_range[0], mu1_range[0]],
            high=[mu0_range[1], mu1_range[1]],
            size=(n_base, 2)
        )
        W_uniform = np.zeros((tot_dofs, n_base))
        for j, (m0, m1) in tqdm(enumerate(params_uniform), total=n_base,
                                 desc="Base snapshots"):
            W_uniform[:, j], _ = solve_FOM(m0, m1,
                                           newton_tol=newton_tol,
                                           max_iterations=max_iter,
                                           verbose=False)

    # ── 2. split train / test ─────────────────────────────────────────────────
    n_test  = max(1, int(n_base * test_ratio))
    n_train = n_base - n_test

    W_test_out     = W_uniform[:, n_train:]
    params_test_out = params_uniform[n_train:]

    # ── 3. enrichment (opzionale) ─────────────────────────────────────────────
    if n_enrich > 0:
        print(f"\nGenerating {n_enrich} enrichment snapshots...")
        np.random.seed(seed_enrich)
        n_each = n_enrich // 3
        n_last = n_enrich - 2 * n_each

        enrich_params = np.vstack([
            np.column_stack([np.random.uniform(0.1,  2.0, n_each),
                             np.random.uniform(1.0,  3.0, n_each)]),
            np.column_stack([np.random.uniform(8.0, 10.0, n_each),
                             np.random.uniform(1.0,  3.0, n_each)]),
            np.column_stack([np.random.uniform(0.1, 10.0, n_last),
                             np.random.uniform(2.0,  3.0, n_last)]),
        ])
        W_enrich = np.zeros((tot_dofs, n_enrich))
        for j, (m0, m1) in tqdm(enumerate(enrich_params), total=n_enrich,
                                 desc="Enrichment snapshots"):
            W_enrich[:, j], _ = solve_FOM(m0, m1,
                                          newton_tol=newton_tol,
                                          max_iterations=max_iter,
                                          verbose=False)
        W_train_out     = np.concatenate([W_uniform[:, :n_train], W_enrich], axis=1)
        params_train_out = np.concatenate([params_uniform[:n_train], enrich_params], axis=0)
    else:
        print("\nNo enrichment.")
        W_train_out      = W_uniform[:, :n_train]
        params_train_out = params_uniform[:n_train]

    # ── 4. salva ──────────────────────────────────────────────────────────────
    np.save(train_snap_path,   W_train_out)
    np.save(train_params_path, params_train_out)
    np.save(test_snap_path,    W_test_out)
    np.save(test_params_path,  params_test_out)

    print(f"\nTrain : {W_train_out.shape[1]} "
          f"({n_train} uniform" +
          (f" + {n_enrich} enriched)" if n_enrich > 0 else ")"))
    print(f"Test  : {W_test_out.shape[1]}  (uniform only)")
    print(f"Saved → {data_dir}")


if __name__ == "__main__":
    generate_snapshots()
