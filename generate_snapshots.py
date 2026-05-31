import numpy as np
import os
from tqdm import tqdm
from solve_FOM import solve_FOM
from setup_fem import tot_dofs

# ── Parametri ─────────────────────────────────────────────────────────────────
snapshot_num = 1000
n_train      = 800
n_enrich     = 100

mu0_range = [0.1, 10.0]
mu1_range = [1.0, 3.0]
P = np.array([mu0_range, mu1_range])

base_snapshot_path  = "./data/snapshots_base.npy"
base_params_path    = "./data/parameters_base.npy"
train_snapshot_path = "./data/snapshots_train.npy"
train_params_path   = "./data/parameters_train.npy"
test_snapshot_path  = "./data/snapshots_test.npy"
test_params_path    = "./data/parameters_test.npy"

os.makedirs("./data", exist_ok=True)

# ── 1. snapshot base uniformi ─────────────────────────────────────────────────
if os.path.exists(base_snapshot_path) and os.path.exists(base_params_path):
    print("Loading existing base snapshots...")
    W_base      = np.load(base_snapshot_path)
    params_base = np.load(base_params_path)
    print(f"  Found {W_base.shape[1]} base snapshots.")

    if W_base.shape[1] < snapshot_num:
        missing_num = snapshot_num - W_base.shape[1]
        print(f"  Generating {missing_num} missing base snapshots...")
        new_params = np.random.uniform(low=P[:, 0], high=P[:, 1],
                                       size=(missing_num, 2))
        W_new = np.zeros((tot_dofs, missing_num))
        for j, (m0, m1) in tqdm(enumerate(new_params), total=missing_num,
                                 desc="Base snapshots (missing)"):
            W_new[:, j], _ = solve_FOM(m0, m1, verbose=False)
        W_base      = np.concatenate([W_base, W_new], axis=1)
        params_base = np.concatenate([params_base, new_params], axis=0)
        np.save(base_snapshot_path, W_base)
        np.save(base_params_path,   params_base)
else:
    print("Generating base snapshots from scratch...")
    np.random.seed(42)
    params_base = np.random.uniform(low=P[:, 0], high=P[:, 1],
                                    size=(snapshot_num, 2))
    W_base = np.zeros((tot_dofs, snapshot_num))
    for j, (m0, m1) in tqdm(enumerate(params_base), total=snapshot_num,
                             desc="Base snapshots"):
        W_base[:, j], _ = solve_FOM(m0, m1, verbose=False)
    np.save(base_snapshot_path, W_base)
    np.save(base_params_path,   params_base)

# ── 2. enrichment nelle zone difficili ────────────────────────────────────────
np.random.seed(123)
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

def find_existing(params_candidate, params_pool, tol=1e-3):
    idx_found = np.full(len(params_candidate), -1, dtype=int)
    for i, p in enumerate(params_candidate):
        dists = np.linalg.norm(params_pool - p, axis=1)
        j = np.argmin(dists)
        if dists[j] < tol:
            idx_found[i] = j
    return idx_found

idx_found    = find_existing(params_enrich_candidates, params_base)
mask_recycle = idx_found >= 0
mask_new     = ~mask_recycle

print(f"\nEnrichment: {mask_recycle.sum()} recycled from base, "
      f"{mask_new.sum()} need FOM solve.")

params_new_only = params_enrich_candidates[mask_new]
W_new_only      = np.zeros((tot_dofs, mask_new.sum()))
for j, (m0, m1) in tqdm(enumerate(params_new_only), total=mask_new.sum(),
                         desc="Enrichment snapshots"):
    W_new_only[:, j], _ = solve_FOM(m0, m1, verbose=False)

W_enrich      = np.zeros((tot_dofs, n_enrich))
params_enrich = np.zeros((n_enrich, 2))
new_counter   = 0
for i in range(n_enrich):
    if mask_recycle[i]:
        W_enrich[:, i]   = W_base[:, idx_found[i]]
        params_enrich[i] = params_base[idx_found[i]]
    else:
        W_enrich[:, i]   = W_new_only[:, new_counter]
        params_enrich[i] = params_new_only[new_counter]
        new_counter += 1

# ── 3. salva train e test ─────────────────────────────────────────────────────
W_train     = np.concatenate([W_base[:, :n_train], W_enrich], axis=1)
param_train = np.concatenate([params_base[:n_train], params_enrich], axis=0)
W_test      = W_base[:, n_train:]
param_test  = params_base[n_train:]

np.save(train_snapshot_path, W_train)
np.save(train_params_path,   param_train)
np.save(test_snapshot_path,  W_test)
np.save(test_params_path,    param_test)

print(f"\nTrain : {W_train.shape[1]} ({n_train} uniform + {n_enrich} enriched)")
print(f"Test  : {W_test.shape[1]}  (uniform only, never contaminated)")
print(f"\nSaved → {train_snapshot_path}")
print(f"Saved → {train_params_path}")
print(f"Saved → {test_snapshot_path}")
print(f"Saved → {test_params_path}")
