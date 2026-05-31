# ns-rom-comparison

Comparison of Reduced Order Methods for the parametric Navier-Stokes equations
on the unit square domain Ω = (0,1)², with parameters μ = (μ₀, μ₁) ∈ [0.1,10] × [1.0,3.0].

## Problem

Given μ = (μ₀, μ₁), find u(μ) such that:

    -μ₀∇·(∇u) + (∇u)u + ∇p = f(x; μ₁)    in Ω
    ∇·u = 0                                  in Ω
    u = 0                                    on ∂Ω

where μ₀ is the kinematic viscosity and f(x; μ₁) is an explicit parametric forcing term.

## Methods

| Method | Description |
|---|---|
| FOM | Full Order Model — Taylor-Hood finite elements, Newton solver |
| POD-Galerkin | Proper Orthogonal Decomposition + reduced Newton |
| POD-NN | POD coefficients learned by a feedforward neural network |
| PINN | Physics-Informed Neural Network |

## Repository Structure
ns-rom-comparison/
├── notebooks/
│   ├── 00_mesh_and_setup.ipynb
│   ├── 01_FOM.ipynb
│   ├── 02_POD_Galerkin.ipynb
│   ├── 03_POD_NN.ipynb
│   ├── 04_PINN.ipynb
│   └── 05_comparison.ipynb
├── src/
│   ├── generate_snapshots.py
│   └── other_utilities.py
├── data/
│   └── README.md
├── requirements.txt
└── README.md

## Data

Snapshots are not tracked by git. You have two options:

**Option 1 — Download** (recommended):

> Google Drive: - Snapshots Train: https://drive.google.com/file/d/16iag1bDzGUrzfbygCK1jyG6-61OmrSfN/view?usp=sharing
>               - Snapshots Test: https://drive.google.com/file/d/1TucyKJJYN8Thq7HHiGQmr8i_juzho43A/view?usp=sharing
>               - Parameters Train:https://drive.google.com/file/d/1DmJAcsbwwDd0SJBxMECJFCA5t8IsRjX0/view?usp=sharing
>               - Parameters Test: https://drive.google.com/file/d/1NjYTmP23npkJmTsAi5JmdDZvHcX9z9lz/view?usp=sharing

Place the downloaded files in `data/`:
- `snapshots_train.npy` — 900 FOM snapshots (800 uniform + 100 enriched)
- `parameters_train.npy`
- `snapshots_test.npy` — 200 uniform FOM snapshots
- `parameters_test.npy`

**Option 2 — Regenerate from scratch:**

```bash
python src/generate_snapshots.py
```

> ⚠️ Requires a working `pypolydim` installation. Snapshot generation takes time.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in order (00 → 05). Each notebook is self-contained
and loads data from `data/`. The comparison notebook (05) collects
results from all methods.

## Requirements

- Python 3.10+
- pypolydim
- numpy, scipy, matplotlib
- torch
- tqdm
