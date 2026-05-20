# Track: MD

Run a molecular-dynamics step traced through the SDK. The platform places each matmul on whatever hardware the engine picks; you set the lattice, the timestep, and how far you push toward Lennard-Jones.

## What ships

- [`starter.ipynb`](starter.ipynb) — 64 argon-mass particles on a face-centered cubic lattice, harmonic restoring forces against a precomputed coupling matrix, one velocity-Verlet step traced as two matmuls.
- [`harmonic_step.py`](harmonic_step.py) — builds the traced module + runtime inputs.
- [`lj_forces.py`](lj_forces.py) — anharmonic Lennard-Jones force kernel (NumPy reference + a tracing scaffold to extend).
- [`baseline.py`](baseline.py) — NumPy reference integrator for both the harmonic and LJ paths.
- [`aimd/`](aimd/) — Ab-initio sub-track: H₂O Born-Oppenheimer MD. Per-step RHF SCF for the electronic energy, finite-difference forces, velocity-Verlet for the nuclei. Two challenge levels — (L1) compile only the SCF loop with `ux.fori_loop`, (L2) replace the whole energy evaluation per geometry via the precompiled `scf_module` chemistry kernel. See [`aimd/README.md`](aimd/README.md).

## Why harmonic first

True Lennard-Jones forces are non-linear in positions, so a single-matmul trace is not sufficient. The SDK has matrix and `expv` primitives that compose well into harmonic dynamics, and the resulting graph is a clean target for the engine's hardware placer.

Once you can run the harmonic step end-to-job and compare to the NumPy reference, you have the workflow plumbing. Then you extend toward anharmonicity — that is the hackathon work.

## SDK surface

```python
import uniqx
from uniqx import ops
from uniqx.core import tracing
from uniqx.domains.common.spatial import pairwise_distances, neighbor_list
```

`pairwise_distances(positions)` and `neighbor_list(positions, cutoff)` are pure-Python pre-computation. They give you the structure you'll bake into the coupling matrix or feed to a traced force loop.

## Where to push beyond the starter

| Direction | What changes |
|---|---|
| Larger crystal (256, 1024 atoms) | Resize the lattice; the matrix scales `N²`; GPU starts beating CPU |
| Replace harmonic with LJ via per-pair traced ops | Unroll over pair indices using `ops.sub`/`mul`/`pow` at trace time |
| Add Langevin or Nosé-Hoover thermostat | Wraps a stochastic / extended-system step around the Verlet kernel |
| Learned force field (GNN-style) | See `uniqx.domains.ml` for the GNN primitives |
| Multi-step trajectory traced as a single module | Use `ops.control_flow` / `fori_loop` to fold N steps into one trace |

## Reference behaviour

Harmonic crystal, 64 atoms, dt small enough for stability: total energy should drift less than 1% over 100 steps (NVE). If your run drifts more, your dt is too large or your coupling matrix is wrong.

## Important note on the SDK rule

The SDK rule says *no NumPy compute inside `@uniqx.to_module`*. That applies to the traced function body. Pre-computing static structure (the coupling matrix, the pair indices) in Python before tracing is fine — `build_lbm_step_module` in the CFD track does exactly that. The line is: *what runs at submit time must be traced ops*.
