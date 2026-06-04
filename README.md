# ns-rom-comparison

Comparison of reduced order methods for the parametric Navier-Stokes equations on the unit square domain.

## Problem setting

The spatial domain is

$$
\Omega = (0,1)^2.
$$

The parameter vector is

$$
\boldsymbol{\mu} = (\mu_0,\mu_1) \in [0.1,10] \times [1.0,3.0].
$$

For each parameter value $\boldsymbol{\mu}$, the goal is to compute the velocity field $u(\boldsymbol{\mu})$ and the pressure field $p(\boldsymbol{\mu})$ satisfying the parametrized Navier-Stokes problem

$$
-\mu_0 \nabla \cdot (\nabla u) + (\nabla u)u + \nabla p = f(x;\mu_1)
\qquad \text{in } \Omega,
$$

$$
\nabla \cdot u = 0
\qquad \text{in } \Omega,
$$

$$
u = 0
\qquad \text{on } \partial \Omega.
$$

The pressure is fixed by imposing

$$
p = 0
\qquad \text{at the vertex } (0,0).
$$

The parameter $\mu_0$ controls the viscosity, while $\mu_1$ enters the explicit parametric forcing term $f(x;\mu_1)$.

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
