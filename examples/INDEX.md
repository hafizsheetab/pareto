# Examples — bring-your-own starting points

30 notebooks lifted from the upstream `uniqx` gallery and verified end-to-end against the production gateway. Use any of them as scaffolding when designing a custom workload for the "bring-your-own" track.

Every notebook follows the same skeleton: problem definition → trace with `uniqx` → `preflight()` → submit to whichever route the engine recommends → compare to a classical oracle. The user code is identical regardless of the route the engine picks — that is the hardware-agnostic property the hackathon scores you on.

Each notebook ends with a **Validation** cell that asserts gateway-vs-classical agreement, so a regression in the platform or in your fork surfaces immediately rather than printing a silently-wrong number.

## Foundational — read these first

| Notebook | What it teaches |
|---|---|
| [`getting_started.ipynb`](notebooks/getting_started.ipynb) | Vector add, matmul, eigs. Trace + submit + parse round-trip. |
| [`hybrid_cpu_gpu_qpu.ipynb`](notebooks/hybrid_cpu_gpu_qpu.ipynb) | The hackathon's central theme — same code, three hardware routes, `preflight()` shows the tradeoff. |

## Algorithm primer

| Notebook | Algorithm |
|---|---|
| [`algorithm_grover_primer.ipynb`](notebooks/algorithm_grover_primer.ipynb) | Grover amplitude amplification |

## Chemistry — DFT track and beyond

| Notebook | Problem | Classical oracle |
|---|---|---|
| [`real_space_quantum_chemistry.ipynb`](notebooks/real_space_quantum_chemistry.ipynb) | Real-space basis instead of Gaussians | PySCF |
| [`nmr_notebook.ipynb`](notebooks/nmr_notebook.ipynb) | Isotropic shielding tensors | PySCF / NMR prop |
| [`geometry_optimization.ipynb`](notebooks/geometry_optimization.ipynb) | Equilibrium geometry via gradients | PySCF |
| [`conformer_search.ipynb`](notebooks/conformer_search.ipynb) | Conformer enumeration | RDKit |
| [`neb_reaction_path.ipynb`](notebooks/neb_reaction_path.ipynb) | Nudged elastic band | classical NEB |
| [`photoisomerization.ipynb`](notebooks/photoisomerization.ipynb) | Excited-state dynamics | TDDFT |
| [`allosteric_simulation.ipynb`](notebooks/allosteric_simulation.ipynb) | Protein allosteric coupling | MD biophysics |

## Physics and PDEs

| Notebook | Problem | Classical oracle |
|---|---|---|
| [`spin_chain_ground_state.ipynb`](notebooks/spin_chain_ground_state.ipynb) | TFI lowest eigenvalue | Lanczos |
| [`spin_chain_dynamics.ipynb`](notebooks/spin_chain_dynamics.ipynb) | e^{-iHt}·ψ | scipy.expm |
| [`poisson_solve_grid.ipynb`](notebooks/poisson_solve_grid.ipynb) | Lu = b on a 2D grid | LU |

## Sampling and statistics

| Notebook | Problem | Classical oracle |
|---|---|---|
| [`variational_monte_carlo.ipynb`](notebooks/variational_monte_carlo.ipynb) | VMC sampling | PRNG |
| [`thermal_state_sampling.ipynb`](notebooks/thermal_state_sampling.ipynb) | Thermal-state samples | classical MCMC |
| [`random_walk_search.ipynb`](notebooks/random_walk_search.ipynb) | Random-walk search | classical walk |
| [`mcmc_cpu_vs_gpu.ipynb`](notebooks/mcmc_cpu_vs_gpu.ipynb) | Direct CPU vs GPU sampling — a model for reporting a hardware tradeoff. |

## Machine learning

| Notebook | Problem | Classical oracle |
|---|---|---|
| [`dense_neural_network_hybrid.ipynb`](notebooks/dense_neural_network_hybrid.ipynb) | Hybrid dense MLP | classical MLP |
| [`generative_adversarial_step.ipynb`](notebooks/generative_adversarial_step.ipynb) | GAN training step | classical GAN |
| [`reinforcement_learning_step.ipynb`](notebooks/reinforcement_learning_step.ipynb) | RL action sampling | tabular Q-learning |
| [`binary_classification_quantum.ipynb`](notebooks/binary_classification_quantum.ipynb) | Binary classifier | logistic regression |
| [`qml_loss_reduction.ipynb`](notebooks/qml_loss_reduction.ipynb) | QML loss reduction (QAE-as-mean) | numpy reduce |
| [`gradient_variance_diagnostic.ipynb`](notebooks/gradient_variance_diagnostic.ipynb) | Variance of gradients (barren-plateau diag) | numerical gradient |

## Optimisation

| Notebook | Problem | Classical oracle |
|---|---|---|
| [`combinatorial_qubo_optimization.ipynb`](notebooks/combinatorial_qubo_optimization.ipynb) | QUBO solver | tabu / simulated annealing |

## Quantum + classical interop

| Notebook | What it shows |
|---|---|
| [`classical_quantum_interaction.ipynb`](notebooks/classical_quantum_interaction.ipynb) | Round-tripping data between classical and quantum kernels. |
| [`constrained_param_unitary.ipynb`](notebooks/constrained_param_unitary.ipynb) | Parameter constraints on a unitary. |
| [`jax_gap_primitives.ipynb`](notebooks/jax_gap_primitives.ipynb) | JAX-style differentiable primitives mapped to gateway ops. |
| [`oqi_usecases.ipynb`](notebooks/oqi_usecases.ipynb) | Selected industrial use cases driven by `oqi` primitives. |

## Real-world demonstrators

| Notebook | What it shows |
|---|---|
| [`threat_detection.ipynb`](notebooks/threat_detection.ipynb) | Anomaly detection with a hybrid scorer. |
| [`quantum_cryptography.ipynb`](notebooks/quantum_cryptography.ipynb) | QKD-style protocol primitive. |

## How to use these in a "bring your own" submission

1. Pick the notebook closest to the workload you want to build.
2. Copy it into `submissions/<team-handle>/submission.ipynb`.
3. Replace the problem definition with yours. Keep the `preflight()` → `submit()` → oracle-compare skeleton.
4. Fill in `results.json.workload_description` (required for `track: "custom"`).
5. Submit per [docs/submission.md](../docs/submission.md).

The judges score you on the *shape of your Pareto frontier* and the *quality of your justification*, not on which example you started from.
