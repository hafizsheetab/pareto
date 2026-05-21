# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# jax_solve.py — JAX/NumPy solver for the 2D lid-driven cavity.
#
# Implements Chorin's projection method using the step_a/b/c modules:
#   A. Explicit viscous diffusion : u* = u + Δt·ν·∇²u
#   B. Pressure Poisson solve     : ∇²p = (ρ/Δt) ∇·u*
#   C. Helmholtz correction       : u^{n+1} = u* − (Δt/ρ) ∇p
#
# Snapshot collection and figure output are handled here; callers receive
# only the final fields and run statistics.
#
# No uniqx dependency. Pressure solved with JAX (dense LU or CG via linalg.py).
# =============================================================================

import os
import time

import config
import linalg
import numpy as np
from boundary import apply_velocity_bc
from fd_operators import divergence_2d
from grid import Grid
from step_a_diffusion import diffuse
from step_b_pressure import build_poisson_matrix, build_rhs, solve_pressure
from step_c_correction import correct_velocity
from visualize import plot_snapshots

_SOLVER_MAP = {
    "direct": linalg.solve_direct,
    "cg":     linalg.solve_cg,
}


def run(
    grid: Grid,
    n_steps: int = config.N_STEPS,
    solver: str = config.PRESSURE_SOLVER,
    tol: float = config.DIV_TOL,
    U_lid: float = config.U_LID,
    save_path: str | None = None,
) -> dict:
    """Run n_steps of Chorin's projection and optionally save a diagnostic figure.

    Parameters
    ----------
    grid      : Grid
    n_steps   : maximum number of time steps
    solver    : "direct" (JAX dense LU) or "cg" (JAX Conjugate Gradient)
    tol       : early-stop threshold on max|∇·u| after correction
    U_lid     : lid velocity override (defaults to config.U_LID)
    save_path : write the figure to this path; None → skip

    Snapshots and diagnostics are printed/collected every config.PRINT_EVERY steps.

    Returns
    -------
    dict with keys
        u, v      : ndarray (N+2, N+2) — final velocity fields
        p         : ndarray (N, N)     — final interior pressure
        step      : int   — last step executed
        converged : bool
        elapsed   : float — wall-clock seconds
    """
    solver_fn = _SOLVER_MAP.get(solver)
    if solver_fn is None:
        raise ValueError(f"Unknown solver '{solver}'. Choose from: {list(_SOLVER_MAP)}")

    N = grid.N

    # --- Initialise fields ----------------------------------------------------
    u = np.zeros((N + 2, N + 2))
    v = np.zeros((N + 2, N + 2))
    p = np.zeros((N, N))
    apply_velocity_bc(u, v, U_lid)

    # --- Assemble Poisson matrix once (geometry is fixed) ---------------------
    A = build_poisson_matrix(grid, pin="symmetric").toarray()

    snapshots: list = []
    converged = False
    step = 0
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        # Step A — explicit diffusion
        u_star, v_star = diffuse(u, v, grid)
        apply_velocity_bc(u_star, v_star, U_lid)

        # Step B — pressure Poisson
        b = build_rhs(u_star, v_star, grid)
        p = solve_pressure(A, b, grid, solver_fn=solver_fn)

        # Step C — Helmholtz correction
        u, v = correct_velocity(u_star, v_star, p, grid)
        apply_velocity_bc(u, v, U_lid)

        # Convergence: max|∂u/∂x + ∂v/∂y| over interior nodes
        div = divergence_2d(u, v, grid.dx, k=config.GRADIENT_ORDER, dy=grid.dy)
        div_max = float(np.max(np.abs(div[1:-1, 1:-1])))

        if step % config.PRINT_EVERY == 0:
            print(f"  step {step:5d}:  max|∂u/∂x + ∂v/∂y| = {div_max:.4e}")
            snapshots.append((step, u.copy(), v.copy(), p.copy()))

        if div_max < tol:
            converged = True
            if not snapshots or snapshots[-1][0] != step:
                snapshots.append((step, u.copy(), v.copy(), p.copy()))
            break

    elapsed = time.perf_counter() - t0

    # Always include the final state
    if not snapshots or snapshots[-1][0] != step:
        snapshots.append((step, u.copy(), v.copy(), p.copy()))

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plot_snapshots(grid, snapshots, save_path=save_path)

    return {
        "u": u,
        "v": v,
        "p": p,
        "step": step,
        "converged": converged,
        "elapsed": elapsed,
    }
