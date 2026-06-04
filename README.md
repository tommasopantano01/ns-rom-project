# ns-rom-comparison

Comparison of reduced order methods for the parametric Navier-Stokes equations on the unit square domain

[
\Omega = (0,1)^2
]

with parameters

[
\mu = (\mu_0, \mu_1) \in [0.1,10] \times [1.0,3.0].
]

The project compares a Full Order Model, a POD-Galerkin Reduced Order Model, a POD-NN model, and an optional PINN approach.

## Problem

Given (\mu = (\mu_0,\mu_1)), find the velocity field (u(\mu)) and pressure field (p(\mu)) such that

[
-\mu_0 \nabla \cdot (\nabla u) + (\nabla u)u + \nabla p = f(x;\mu_1)
\quad \text{in } \Omega,
]

[
\nabla \cdot u = 0
\quad \text{in } \Omega,
]

[
u = 0
\quad \text{on } \partial \Omega.
]

The pressure is fixed by imposing (p=0) at the vertex ((0,0)). The parameter (\mu_0) controls the viscosity, while (\mu_1) enters the explicit parametric forcing term (f(x;\mu_1)).

## Methods

| Method       | Description                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------ |
| FOM          | Full Order Model based on Taylor-Hood finite elements and Newton iterations                      |
| POD-Galerkin | Projection-based reduced model using Proper Orthogonal Decomposition and a reduced Newton solver |
| POD-NN       | Non-intrusive reduced model where POD coefficients are learned by a feedforward neural network   |
| PINN         | Physics-Informed Neural Network approach, optional extension                                     |

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
```

## Data

The snapshot arrays are not tracked by git because of their size. They must be downloaded separately and placed inside the `data/` folder.

### Download links

| File                   | Description                                            | Link                                                                               |
| ---------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `snapshots_train.npy`  | 900 FOM training snapshots, 800 uniform + 100 enriched | https://drive.google.com/file/d/16iag1bDzGUrzfbygCK1jyG6-61OmrSfN/view?usp=sharing |
| `snapshots_test.npy`   | 200 FOM test snapshots                                 | https://drive.google.com/file/d/1TucyKJJYN8Thq7HHiGQmr8i_juzho43A/view?usp=sharing |
| `parameters_train.npy` | Training parameters                                    | https://drive.google.com/file/d/1DmJAcsbwwDd0SJBxMECJFCA5t8IsRjX0/view?usp=sharing |
| `parameters_test.npy`  | Test parameters                                        | https://drive.google.com/file/d/1NjYTmP23npkJmTsAi5JmdDZvHcX9z9lz/view?usp=sharing |

After downloading, the folder should look like this:

```text
data/
├── snapshots_train.npy
├── snapshots_test.npy
├── parameters_train.npy
└── parameters_test.npy
```

Alternatively, the data can be regenerated from scratch with

```bash
python main.py --mode generate-data
```

or, if available,

```bash
python src/generate_snapshots.py
```

Snapshot generation requires a working `pypolydim` installation and can be computationally expensive.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If needed, install `pypolydim` explicitly:

```bash
pip install --force-reinstall pypolydim==2.0.17
```

## Usage

The project is script-based. The main entry point is `main.py`.

Run the POD-Galerkin ROM:

```bash
python main.py --mode rom
```

Run the POD-NN model:

```bash
python main.py --mode podnn
```

Run the optional PINN model:

```bash
python main.py --mode pinn
```

Run the full comparison:

```bash
python main.py --mode compare
```

Generate the final plots:

```bash
python main.py --mode plot
```

Run the complete pipeline:

```bash
python main.py --mode all
```

All relevant parameters, paths, tolerances, reduced dimensions, and training options should be specified in `config.yaml`.

## Outputs

Final results are stored in the `results/` folder.

```text
results/
├── figures/
│   ├── pod_energy_decay.png
│   ├── rom_error_vs_ntot.png
│   ├── podnn_error.png
│   ├── rom_vs_podnn_accuracy.png
│   └── computational_time_comparison.png
│
└── tables/
    ├── rom_errors.csv
    ├── podnn_errors.csv
    └── timing_results.csv
```

Only final figures and tables are stored in the repository. Large snapshot files, temporary outputs, model checkpoints, and debugging plots are excluded from git.

## Requirements

* Python 3.10+
* pypolydim
* numpy
* scipy
* matplotlib
* torch
* tqdm
* vtk

## Notes

The POD-Galerkin error is evaluated with respect to the FOM solution. The velocity components are measured using an (H^1)-type relative error, while the pressure is measured using a relative (L^2)-type error.

For very high POD tolerances, the reduced basis may reach the maximum number of available snapshot modes. In that case, further increasing the tolerance does not enrich the reduced space and the error reaches a saturation regime.
