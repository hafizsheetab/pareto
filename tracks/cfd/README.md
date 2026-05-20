# Track: CFD

A finite-difference solver for 2-D incompressible Stokes flow using **Chorin's projection method**, designed as a heterogeneous-computing workflow where each physical step maps to a distinct hardware target. The platform decides where each stage runs; you decide the grid, the Reynolds number, the pressure solver, and how far you push the QPU path.

## Governing Equations

$$\frac{\partial \mathbf{u}}{\partial t} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0$$

The advection term is dropped (low Reynolds number assumption).

## Solution Method

Each time step splits into three hardware-mapped stages:

| Step | Equation | Hardware |
|------|----------|----------|
| **A — Diffusion** | $\mathbf{u}^* = \mathbf{u}^n + \Delta t\, \nu \nabla^2 \mathbf{u}^n$ | GPU / TPU |
| **B — Pressure Poisson** | $\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$ | QPU (classical JAX path) |
| **C — Correction** | $\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p$ | CPU / TPU |

Test case: **lid-driven cavity** at Re = 100.

## Quick Start

```bash
pip install numpy scipy matplotlib   # add jax jaxlib for GPU/TPU
python main.py
```

Output is saved to `results.png` (velocity magnitude, streamlines, pressure).

## Configuration

All parameters live in `config.py`:

```python
N               = 64       # grid resolution (N×N interior nodes)
NU              = 0.01     # kinematic viscosity
PRESSURE_SOLVER = "cg"     # "direct" | "cg" | "vqls"
```

## Swapping the Pressure Solver

The linear system $Ax = b$ is solved via `qpu_linalg.solve()`:

```python
pressure_vector = qpu_linalg.solve(A_matrix, b, method="vqls", tolerance=1e-4)
```

To connect a real QPU, implement `_solve_vqls()` in `qpu_linalg.py`. Classical backends (`"direct"`, `"cg"`) use JAX and run on CPU/GPU/TPU with no code changes.
