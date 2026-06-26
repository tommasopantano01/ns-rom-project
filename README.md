# ns-rom-project
Comparison of reduced order methods for the parametric Navier-Stokes equations on the unit square domain.

## Problem setting
The spatial domain is
$$\Omega = (0,1)^2.$$
The parameter vector is
$$\boldsymbol{\mu} = (\mu_0,\mu_1) \in [0.1,10] \times [1.0,3.0].$$
For each parameter value $\boldsymbol{\mu}$, the goal is to compute the velocity field $u(\boldsymbol{\mu})$ and the pressure field $p(\boldsymbol{\mu})$ satisfying the parametrized Navier-Stokes problem
$$-\mu_0 \nabla \cdot (\nabla u) + (\nabla u)u + \nabla p = f(x;\mu_1) \qquad \text{in } \Omega,$$
$$\nabla \cdot u = 0 \qquad \text{in } \Omega,$$
$$u = 0 \qquad \text{on } \partial \Omega.$$
The pressure is fixed by imposing $p = 0$ at the vertex $(0,0)$. The parameter $\mu_0$ controls the viscosity, while $\mu_1$ enters the explicit parametric forcing term $f(x;\mu_1)$.

## Methods
| Method | Description |
|---|---|
| FOM | Full Order Model based on Taylor-Hood finite elements and Newton iterations |
| POD-Galerkin | Projection-based reduced model using Proper Orthogonal Decomposition and a reduced Newton solver |
| POD-NN | Non-intrusive reduced model where POD coefficients are learned by a feedforward neural network |
| PINN | Hybrid Physics-Informed Neural Network trained on PDE residuals, boundary conditions, and FOM snapshots (80/20 split); fine-tuned with L-BFGS |

## Repository structure
```text
ns-rom-project/
│
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── download.py
├── plot.py
│
├── src/
│   ├── setup_fem.py
│   ├── other_utilities.py
│   ├── build_basis.py
│   ├── solve_FOM.py
│   ├── solve_ROM.py
│   ├── solve_PODNN.py
│   ├── train_PODNN.py
│   ├── solve_PINN.py
│   └── train_PINN.py
│
├── validation/
│   ├── validate_ROM.py
│   ├── validate_podnn.py
│   └── validate_pinn.py
│
├── data/
│   └── (snapshot files, not tracked by git)
│
├── models/
│   └── (trained weights, not tracked by git)
│
└── results/
    └── (plots and metrics, not tracked by git)
```

## Data
The `data/`, `models/`, and `results/` folders are not tracked by git.
Create them locally before running:
```bash
mkdir data models results
```

Snapshot files for POD-Galerkin and POD-NN are not tracked by git. Download them automatically:
```bash
python download.py
```
## Enriched snapshots

By default, all modes that use training snapshots (`build_basis`, `train_podnn`)
load the enriched snapshot set (`snapshots_train_enriched.npy`,
`parameters_train_enriched.npy`) if available. To use the base training set
instead, pass the `--no_enriched` flag:

```bash
python main.py --mode build_basis --no_enriched
python main.py --mode train_podnn --no_enriched
```
## Usage
Build the POD basis (run once):
```bash
python main.py --mode build_basis
```
Train the POD-NN:
```bash
python main.py --mode train_podnn
```
Train the PINN:
```bash
python main.py --mode train_pinn
```
Validate POD-Galerkin:
```bash
python main.py --mode validate_rom
```
Validate POD-NN:
```bash
python main.py --mode validate_podnn
```
Validate PINN:
```bash
python main.py --mode validate_pinn
```
Plot results:
```bash
python main.py --mode plot --what all
```

All hyperparameters are controlled via `config.yaml`.
