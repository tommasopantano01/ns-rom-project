import numpy as np
import os
from tqdm import tqdm
from solve_FOM import solve_FOM
from setup_fem import tot_dofs

mu0_range = [0.1, 10.0]
mu1_range = [1.0, 3.0]


def find_existing(params_candidate, params_pool, tol=1e-3):
    idx_found = np.full(len(params_candidate), -1, dtype=int)
    for i, p in enumerate(params_candidate):
        dists = np.linalg.norm(params_pool - p, axis=1)
        j = np.argmin(dists)
        if dists[j] < tol:
            idx_found[i] = j
    return idx_found


def generate_snapshots(n_base=1000, n_train=800, n_enrich=100,
                       seed_base=42, seed_enrich=123,
                       newton_tol=1e-6, max_iter=20,
                       data_dir="./data"):

    os.makedirs(data_dir, exist_ok=True)

    train_snap_path   = os.path.join(data_dir, "snapshots_train.npy")
    train_params_path = os.path.join(data_dir, "parameters_train.npy")
    test_snap_path    = os.path.join(data_dir, "snapshots_test.npy")
    test_params_path  = os.path.join(data_dir, "parameters_test.npy")

    # ── 1. carica esistenti o genera da zero ──────────────────────────────────
    if os.path.exists(train_snap_path) and os.path.exists(test_snap_path):
        W_train_ex      = np.load(train_snap_path)
        params_train_ex = np.load(train_params_path)
        W_test_ex       = np.load(test_snap_path)
        params_test_ex  = np.load(test_params_path)

        # ricostruisci il pool base (train senza enriched + test)
        # i primi n_train_ex - n_enrich sono uniform, gli ultimi n_enrich sono enriched
        n_uniform_ex = W_train_ex.shape[1] - n_enrich
        W_base      = np.concatenate([W_train_ex[:, :n_uniform_ex], W_test_ex], axis=1)
        params_base = np.concatenate([params_train_ex[:n_uniform_ex], params_test_ex], axis=0)

        existing_num = W_base.shape[1]
        print(f"Found {existing_num} existing uniform snapshots.")

        if n_base <= existing_num:
            # prendi solo quelli che servono, niente FOM
            print(f"Requested {n_base} <= {existing_num} existing: no FOM needed.")
            W_base      = W_base[:, :n_base]
            params_base = params_base[:n_base]
        else:
            # genera solo i mancanti
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
            W_base      = np.concatenate([W_base, W_new], axis=1)
            params_base = np.concatenate([params_base, new_params], axis=0)

    else:
        # niente esistenti, genera tutto da zero
        print(f"No existing snapshots found. Generating {n_base} from scratch...")
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

    # ── 2. enrichment ─────────────────────────────────────────────────────────
    np.random.seed(seed_enrich)
    n_each = n_enrich // 3
    n_last = n_enrich - 2 * n_each

    params_enrich_candidates = np.vstack([
        np.column_stack([np.random.uniform(0.1, 2.0,  n_each),
                         np.random.uniform(1.0, 3.0,  n_each)]),
        np.column_stack([np.random.uniform(8.0, 10.0, n_each),
                         np.random.uniform(1.0, 3.0,  n_each)]),
        np.column_stack([np.random.uniform(0.1, 10.0, n_last),
                         np.random.uniform(2.0, 3.0,  n_last)]),
    ])

    idx_found    = find_existing(params_enrich_candidates, params_base)
    mask_recycle = idx_found >= 0
    mask_new     = ~mask_recycle

    print(f"\nEnrichment: {mask_recycle.sum()} recycled, "
          f"{mask_new.sum()} need FOM solve.")

    W_new_only = np.zeros((tot_dofs, mask_new.sum()))
    for j, (m0, m1) in tqdm(enumerate(params_enrich_candidates[mask_new]),
                             total=mask_new.sum(), desc="Enrichment"):
        W_new_only[:, j], _ = solve_FOM(m0, m1,
                                        newton_tol=newton_tol,
                                        max_iterations=max_iter,
                                        verbose=False)

    W_enrich      = np.zeros((tot_dofs, n_enrich))
    params_enrich = np.zeros((n_enrich, 2))
    new_counter   = 0
    for i in range(n_enrich):
        if mask_recycle[i]:
            W_enrich[:, i]   = W_base[:, idx_found[i]]
            params_enrich[i] = params_base[idx_found[i]]
        else:
            W_enrich[:, i]   = W_new_only[:, new_counter]
            params_enrich[i] = params_enrich_candidates[mask_new][new_counter]
            new_counter += 1

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

    print(f"\nTrain : {W_train.shape[1]} ({n_train_actual} uniform + {n_enrich} enriched)")
    print(f"Test  : {W_test.shape[1]}  (uniform only)")
    print(f"Saved → {data_dir}")


if __name__ == "__main__":
    generate_snapshots()
