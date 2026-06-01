import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from tqdm import tqdm
from solve_FOM import solve_FOM
from setup_fem import tot_dofs
from download import download_data

mu0_range = [0.1, 10.0]
mu1_range = [1.0, 3.0]


def _add_uniform(W, params, n, newton_tol, max_iter, seed):
    np.random.seed(seed)
    new_params = np.random.uniform(
        low =[mu0_range[0], mu1_range[0]],
        high=[mu0_range[1], mu1_range[1]],
        size=(n, 2)
    )
    W_new = np.zeros((tot_dofs, n))
    for j, (m0, m1) in tqdm(enumerate(new_params), total=n, desc="Adding uniform snapshots"):
        W_new[:, j], _ = solve_FOM(m0, m1, newton_tol=newton_tol,
                                   max_iterations=max_iter, verbose=False)
    return np.concatenate([W, W_new], axis=1), np.concatenate([params, new_params], axis=0)


def _remove_random(W, params, n, seed):
    np.random.seed(seed)
    n_cur = W.shape[1]
    assert n < n_cur, f"Cannot remove {n} from {n_cur} snapshots."
    keep = np.sort(np.random.choice(n_cur, n_cur - n, replace=False))
    return W[:, keep], params[keep]


def generate_snapshots(n_train=None, n_test=None, n_enrich=0,
                       seed_base=42, seed_enrich=123,
                       newton_tol=1e-6, max_iter=20,
                       data_dir="./data"):

    os.makedirs(data_dir, exist_ok=True)

    train_snap_path   = os.path.join(data_dir, "snapshots_train.npy")
    train_params_path = os.path.join(data_dir, "parameters_train.npy")
    test_snap_path    = os.path.join(data_dir, "snapshots_test.npy")
    test_params_path  = os.path.join(data_dir, "parameters_test.npy")

    # ── 1. scarica da Drive (sempre, se non esistono) ─────────────────────────
    download_data(data_dir)

    W_train      = np.load(train_snap_path)
    params_train = np.load(train_params_path)
    W_test       = np.load(test_snap_path)
    params_test  = np.load(test_params_path)
    print(f"Loaded: {W_train.shape[1]} train, {W_test.shape[1]} test.")

    # ── 2. modifica train se richiesto ────────────────────────────────────────
    if n_train is not None and n_train != 0:
        if n_train > 0:
            print(f"Adding {n_train} uniform snapshots to train...")
            W_train, params_train = _add_uniform(
                W_train, params_train, n_train, newton_tol, max_iter, seed_base)
        else:
            print(f"Removing {-n_train} random snapshots from train...")
            W_train, params_train = _remove_random(
                W_train, params_train, -n_train, seed_base)

    # ── 3. modifica test se richiesto ─────────────────────────────────────────
    if n_test is not None and n_test != 0:
        if n_test > 0:
            print(f"Adding {n_test} uniform snapshots to test...")
            W_test, params_test = _add_uniform(
                W_test, params_test, n_test, newton_tol, max_iter, seed_base + 1)
        else:
            print(f"Removing {-n_test} random snapshots from test...")
            W_test, params_test = _remove_random(
                W_test, params_test, -n_test, seed_base + 1)

    # ── 4. enrichment (opzionale) ─────────────────────────────────────────────
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
        W_train      = np.concatenate([W_train, W_enrich], axis=1)
        params_train = np.concatenate([params_train, enrich_params], axis=0)

    # ── 5. salva ──────────────────────────────────────────────────────────────
    np.save(train_snap_path,   W_train)
    np.save(train_params_path, params_train)
    np.save(test_snap_path,    W_test)
    np.save(test_params_path,  params_test)

    print(f"\nTrain : {W_train.shape[1]}")
    print(f"Test  : {W_test.shape[1]}")
    print(f"Saved → {data_dir}")


if __name__ == "__main__":
    generate_snapshots()
