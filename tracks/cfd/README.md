# Track: CFD

Computational Fluid Dynamics solves the Navier-Stokes equations on a discrete mesh to predict how fluids move, mix, and react. Three major discretisation families exist: **Finite Element Methods** (FEM) express the solution as a sum of basis functions on unstructured meshes; **Finite Volume Methods** (FVM) enforce conservation laws cell by cell on control volumes — the industry standard in commercial codes (OpenFOAM, Fluent); **Finite Difference Methods** (FDM) approximate derivatives by polynomial stencils on structured Cartesian grids — simpler to implement, regular memory access, and a natural fit for GPU and TPU parallelism. This hackathon uses FDM throughout.

A single FDM time step may combine several distinct computational patterns, such as

- **Dense linear algebra**
- **Sparse symmetric linear systems**
- **Stencil sweeps**

Routing each stage to the hardware it runs fastest on is heterogeneous dispatch — and it is exactly what uniqx does. The full simulation graph is submitted once; the engine schedules each subgraph to CPU, GPU, TPU, or QPU without the user managing data movement between devices.

---

## Starting point

A 2-D incompressible Stokes solver (low Re, advection neglected) on a lid-driven cavity using **Chorin's projection method**. Three stages per step, each targeting a different device class:

| Stage | Equation | Compute pattern |
|-------|----------|-----------------|
| A — Diffusion | u★ = u + Δt·ν·∇²u | Stencil sweep |
| B — Pressure Poisson | ∇²p = (ρ/Δt) ∇·u★ | Dense / sparse linear solve |
| C — Correction | u^{n+1} = u★ − (Δt/ρ) ∇p | Stencil sweep |

### File map

**Shared — both paths use these**

| File | Role |
|------|------|
| `config.py` | Grid size, viscosity, time stepping, solver choice |
| `grid.py` | Geometry and Von Neumann–stable dt |
| `fd_operators.py` | Variable-order FD stencils O(h²)–O(h⁶) |
| `boundary.py` | Lid-driven cavity BCs |
| `step_a_diffusion.py` | Explicit Laplacian update (Step A) |
| `step_b_pressure.py` | Poisson matrix assembly + solver dispatch (Step B) |
| `step_c_correction.py` | Pressure-gradient velocity correction (Step C) |
| `linalg.py` | JAX solver backends: dense LU (`solve_direct`) and CG (`solve_cg`) |
| `visualize.py` | Velocity magnitude, streamlines, pressure contours |

**JAX path — local execution, no gateway**

| File | Role |
|------|------|
| `jax_solve.py` | Chorin's projection loop: runs all steps, collects snapshots, saves figure |
| `jax_main.py` | Entry point — prints diagnostics, delegates everything to `jax_solve.run()` |

**uniqx path — traces IR, submits to gateway**

| File | Role |
|------|------|
| `_traced_ops.py` | Inline stencil helpers for the traced IR body |
| `solver.py` | Fuses all stages into one `fori_loop` IR module |
| `main.py` | Entry point — submits to gateway, parses result, saves figure |

### Run

```bash
pip install -r requirements.txt

# JAX (local, no credentials needed)
python jax_main.py [--solver direct|cg] [--steps N] [--n N]

# uniqx (requires gateway access)
export UNIQX_GATEWAY=<host:port>
export UNIQX_API_KEY=<key>
python main.py [--steps N] [--n N] [--gateway host:port]
```

---

## Challenges

### 1 — Benchmark available hardware — 15 pts ★★☆☆☆

Profile the solver on every backend the gateway exposes for your account. Measure wall time per step, identify the bottlenecks, and compare against the JAX reference implementation.

**Deliverable:** timing bar chart (backend × grid size), short writeup explaining where the cost lives in each stage.

---

### 2 — Extend with native uniqx kernels — 20 pts ★★★☆☆

The `_traced_ops.py` helpers implement Laplacian, gradient, and divergence from raw slice/concat/sub/mul ops at order O(h²). Replace them with native uniqx kernel calls to reach higher-order accuracy and hardware-optimised implementations:

| Kernel | Operation | Module |
|--------|-----------|--------|
| `grid_gradient` | Kronecker-product differentiation matrix | `uniqx.domains.physics.kernels` |
| `grid_laplacian` | Sparse SPD Laplacian matrix | `uniqx.domains.physics.kernels` |
| `grid_divergence` | Row-stacked FD divergence matrix | `uniqx.domains.physics.kernels` |
| `grid_curl` | FD curl operator | `uniqx.domains.physics.kernels` |
| `grid_helmholtz` | Helmholtz operator (∇² + k²) | `uniqx.domains.physics.kernels` |

Replace the manual helpers with native kernel calls. Validate numerical equivalence against the existing output, then benchmark the substitution.

**Deliverable:** modified `_traced_ops.py` (or `_traced_ops_v2.py`), equivalence test, benchmark comparison table.

---

### 3 — Taylor-Green vortex solver — 40 pts ★★★★☆

Implement a 3D Navier-Stokes solver for the **Taylor-Green vortex** (TGV) benchmark. Unlike the Stokes solver, TGV requires the full NS equations (advection term u·∇u is dominant at Re = 1600), triply-periodic boundary conditions, and a time-integration scheme accurate enough for turbulence.

Domain: [0, 2πL]³, Re = V₀L/ν = 1600. Initial conditions:

```
u(x,y,z,0) =  V₀ sin(x/L) cos(y/L) cos(z/L)
v(x,y,z,0) = −V₀ cos(x/L) sin(y/L) cos(z/L)
w(x,y,z,0) =  0
```

Implement the solver classically (JAX or NumPy) first, then provide a uniqx version. Validate by tracking the kinetic energy dissipation rate:

```
ε(t) = ν/V ∫|∇u|² dV
```

and comparing against published data. A resolution of 32 points per dimension is sufficient.

**Deliverable:** Python files implementing TGV, `ε(t)` plot vs. published DNS reference.

---

### 4 — Kernel-fusion design + Python prototype — 25 pts ★★★★★

Design and implement Python uniqx modules that fuse the NS right-hand side into as few traced primitives as possible. Two design questions to address in a short written section before the implementation:

**Time integration scheme.** Which scheme would you choose and why? Consider accuracy order, CFL stability bound, number of temporary arrays (important for GPU memory), and how well the stages fuse into kernel dispatches. Compare at least two options — for example, classical RK4 (4 stages, O(Δt⁴), 4 register arrays) versus low-storage SSP-RK3 (3 stages, O(Δt³), 2 register arrays, strong-stability-preserving property useful for advection-dominated flows).

**Kernel fusion.** The NS-RHS computation requires all 9 partial derivatives ∂uᵢ/∂xⱼ. In a naive implementation each is a separate stencil pass and a separate memory allocation. What is the natural fusion boundary, and how many memory reads/writes does the fused trace need compared to the naive decomposition? A single `ns_rhs` traced module taking (u, v, w) and returning (rhs_u, rhs_v, rhs_w) can compute all gradients and immediately contract them into the advection and viscous terms in one pass — no intermediate gradient tensors materialised at the IR level.

Implement one or more of the following as a single `@uniqx.to_module` trace, validated against a NumPy reference and benchmarked against the Challenge-3 implementation:

- **Periodic gradient** — wrap-around boundary padding instead of zero-pad
- **Vorticity** — fused ω = ∇×u as one traced module
- **NS-RHS** — advection + viscous term in one traced module

**Deliverable:** design document (scheme choice + fusion analysis), the Python modules with a unit test against a NumPy reference, updated `ε(t)` plot showing improvement in throughput or accuracy over the Challenge-3 implementation.

---

## Scoring

| Challenge | Points | Difficulty |
|-----------|--------|------------|
| 1 — Benchmark | 15 | ★★☆☆☆ |
| 2 — Extend primitives | 20 | ★★★☆☆ |
| 3 — TGV solver | 40 | ★★★★☆ |
| 4 — Kernel-fusion design + prototype | 25 | ★★★★★ |
| **Total** | **100** | |

Partial credit is awarded. Challenges build on each other but can be attempted independently. A working TGV solver (Challenge 3) without fused kernels (Challenge 4) still earns substantial credit.

Submit per [docs/submission.md](../../docs/submission.md): copy [templates/submission/](../../templates/submission/) into `submissions/<team-handle>/`, fill in `results.json` / `submission.ipynb` / `preflight_log.txt`, and open one PR per team against this repo before the deadline.
