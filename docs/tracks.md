# Tracks

Three pre-scaffolded tracks plus a fourth "design your own" lane.

All four run in **Studio** with no install (browser, `uniqx` pre-installed, key pre-injected) — or locally via the [quickstart](quickstart.md) if you prefer your own Python environment.

## DFT — density functional theory

**Starter problem**: H₂O at STO-3G. Compute the SCF ground-state energy and the isotropic NMR shielding tensors.

**SDK surface**:
- `uniqx.domains.chemistry.basis.extract_basis(geometry, basis_name)`
- `uniqx.domains.chemistry.hartree_fock.rhf_module(geometry, basis_info)`
- `uniqx.domains.chemistry.nmr_full.nmr_full_module(geometry, basis_info)`

**Where to push**:
- Larger basis sets (6-31G, cc-pVDZ) — watch the cost column climb
- Larger molecules (methane, methanol, alanine) — watch the gateway split the graph into more execution blocks
- Geometry optimization on top of the SCF
- Compare convergence behaviour at different `max_iter` values

**Baseline**: PySCF reference for accuracy comparison. Ships in [tracks/dft/baseline.py](../tracks/dft/baseline.py).

---

## CFD — computational fluid dynamics

**Starter problem**: 2-D incompressible Stokes flow via **Chorin's projection method**. Lid-driven cavity at Re=100 on a 64 × 64 grid. Each time step splits into three hardware-mapped stages:

| Step | Equation | Hardware |
|------|----------|----------|
| **A — Diffusion** | `u* = uⁿ + Δt ν ∇²uⁿ` | GPU / TPU |
| **B — Pressure Poisson** | `∇²p = (ρ/Δt) ∇·u*` | QPU (classical JAX path) |
| **C — Correction** | `uⁿ⁺¹ = u* − (Δt/ρ) ∇p` | CPU / TPU |

**Files**: `main.py` (entry), `solver.py` (time loop), `step_a_diffusion.py` / `step_b_pressure.py` / `step_c_correction.py` (per-stage kernels), `grid.py`, `boundary.py`, `fd_operators.py`, `linalg.py`, `config.py`, `uniqx_client.py`, `visualize.py`.

**Where to push**:
- Scale the grid to 256² or beyond; the Poisson solve dominates
- Cavity at Re=1000 — the Pareto front moves
- Swap the pressure solver (`PRESSURE_SOLVER = "cg" | "direct" | "vqls"`) and connect a real QPU by implementing `_solve_vqls()` in `linalg.py`
- Extend Stokes to Navier-Stokes by reintroducing the advection term

**Baseline**: The classical CG / direct solver paths in `solver.py` serve as the reference. `python main.py` writes `results.png` (velocity, streamlines, pressure).

---

## MD — molecular dynamics (ab-initio)

**Starter problem**: H₂O Born-Oppenheimer molecular dynamics. At every timestep solve the Restricted Hartree-Fock (RHF) SCF equations on STO-3G for the electronic energy, take finite-difference forces, propagate the nuclei with velocity-Verlet. A working NumPy/SciPy implementation is provided as the baseline.

**Two challenge levels**:
- **Level 1** — replace just the RHF-SCF loop with `ux.fori_loop`. Matrices `X`, `g_J`, `g_K`, `H` are precomputed in Python and passed as runtime inputs. The whole loop compiles into one IR module per geometry; the naïve "one submit per SCF iteration" path is slower than NumPy because of network round-trips, so getting the loop architecture right is the core of the challenge.
- **Level 2** — replace the entire energy evaluation with a single call to the precompiled `uniqx.domains.chemistry.scf_module` chemistry kernel. No Python integral engine in the hot loop — one backend submit per geometry.

**Files**: `baseline.py` (entry point), `aimd.py` (integrator), `scf.py` (electronic structure), `integrals.py` (McMurchie-Davidson engine), `basis.py` (STO-3G), `constants.py`, `sto-3g.dat`.

**Where to push**:
- Larger molecules (NH₃, CH₄, H₂O₂) — graph fan-out grows
- Longer trajectories, NVT thermostatting (Langevin, Nosé-Hoover)
- Tighter SCF convergence — read the `max_error_rate` column from `preflight()`

**Baseline**: NumPy/SciPy AIMD in [tracks/md/baseline.py](../tracks/md/baseline.py).

---

## Bring your own

You may submit against a workload that is not one of the three tracks above. To qualify:

1. **State the problem precisely** — one paragraph in `results.json.workload_description`. The judges should understand the scientific question from that paragraph alone.
2. **Provide a baseline** — a NumPy / SciPy / PySCF / domain-standard reference run in `baseline.py`. Without a baseline, the judges cannot score Performance.
3. **Use `uniqx` for the heavy lifting** — at least one module must be traced with `@uniqx.to_module` and submitted through `preflight()` → `submit()`. A submission that doesn't engage the SDK doesn't engage the hackathon.

Custom workloads are scored on the same four criteria as the pre-defined tracks. Originality of the workload itself counts toward Creativity. The bar for Robustness is higher because the judges have no prior reference run to compare against — show your homework.

### Starting points

The fastest way to bootstrap a custom workload is to fork one of the curated examples:

- **30 examples in this repo** — see [`examples/INDEX.md`](../examples/INDEX.md). Each ends with a Validation cell that asserts gateway-vs-classical agreement. Covers foundational tracing, chemistry, physics/PDEs, sampling, ML, optimisation, interop, and real-world demonstrators.
- **Full gallery** — [app.oriqx.com/examples](https://app.oriqx.com/examples). Sign in with your hackathon account to browse and open any notebook in Studio.

Every example follows the same `problem → trace → preflight → run → oracle-compare` skeleton — you replace the problem, keep the skeleton, and you have a runnable submission.
