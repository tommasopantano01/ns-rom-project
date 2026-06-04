# ns-rom-comparison

Comparison of reduced order methods for the parametric Navier-Stokes equations on the unit square domain

\[
\Omega = (0,1)^2
\]

with parameters

\[
\mu = (\mu_0, \mu_1) \in [0.1,10] \times [1.0,3.0].
\]

The project compares a Full Order Model, a POD-Galerkin Reduced Order Model, a POD-NN model, and an optional PINN approach.

## Problem

Given \(\mu = (\mu_0,\mu_1)\), find the velocity field \(u(\mu)\) and pressure field \(p(\mu)\) such that

\[
-\mu_0 \nabla \cdot (\nabla u) + (\nabla u)u + \nabla p = f(x;\mu_1)
\quad \text{in } \Omega,
\]

\[
\nabla \cdot u = 0
\quad \text{in } \Omega,
\]

\[
u = 0
\quad \text{on } \partial \Omega.
\]

The pressure is fixed by imposing \(p=0\) at the vertex \((0,0)\). The parameter \(\mu_0\) controls the viscosity, while \(\mu_1\) enters the explicit parametric forcing term \(f(x;\mu_1)\).

## Methods

| Method | Description |
|---|---|
| FOM | Full Order Model based on Taylor-Hood finite elements and Newton iterations |
| POD-Galerkin | Projection-based reduced model using Proper Orthogonal Decomposition and a reduced Newton solver |
| POD-NN | Non-intrusive reduced model where POD coefficients are learned by a feedforward neural network |
| PINN | Physics-Informed Neural Network approach, optional extension |

## Repository structure

```text
ns-rom-comparison/
│
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── download_data.py
│
├── src/
│   ├── fom.py
│   ├── pod.py
│   ├── rom.py
│   ├── podnn.py
│   ├── pinn.py
│   ├── metrics.py
│   ├── plotting.py
│   └── utils.py
│
├── data/
│   └── external snapshot files, not tracked by git
│
└── results/
    ├── figures/
    └── tables/
