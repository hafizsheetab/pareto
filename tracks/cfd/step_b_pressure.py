# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# step_b_pressure.py — Step B: Pressure Poisson equation.
#
# Governing equation:
#   ∇²p = (ρ/Δt) · ∇·u*
#
# Reformulated as the sparse linear system  A · x = b  where:
#   A  — discrete Laplacian operator, shape (N², N²)
#        assembled ONCE before the time loop (geometry is fixed)
#   x  — unknown flattened pressure field, shape (N²,)
#   b  — (ρ/Δt) · ∇·u*, shape (N²,), recomputed every step
#
# Solver backends live in linalg.py; callers pass a solver_fn callable to
# solve_pressure():
#   linalg.solve_direct  — dense JAX LU
#   linalg.solve_cg      — Conjugate Gradient JAX
# =============================================================================

import config
import linalg
import numpy as np
import scipy.sparse as sp
from fd_operators import divergence_2d
from grid import Grid

# ---------------------------------------------------------------------------
# Sub-step B.1 — Build the Laplacian matrix A  (called ONCE before the loop)
# ---------------------------------------------------------------------------


def build_poisson_matrix(grid: Grid, pin: str = "symmetric") -> sp.csr_matrix:
    """
    Assemble the N²×N² discrete Laplacian for the interior pressure nodes.

    Global index mapping: node (i, j) → k = i*N + j,  i,j ∈ {0, …, N-1}.

    Stencil (central differences, uniform dx=dy):
        A[k, k]     = -4 / dx²
        A[k, k±1]   = +1 / dx²   (east/west, skip at row edges)
        A[k, k±N]   = +1 / dx²   (north/south)

    Neumann BC at walls: simply omit the off-domain neighbour term —
    equivalent to ∂p/∂n = 0 at every wall.

    Null-space pin at corner k=0:
        pin="row"       — zero row 0 only (standard Dirichlet, fine for LU)
        pin="symmetric" — zero row 0 AND col 0 (symmetric; required for CG)

    Returns
    -------
    sp.csr_matrix, shape (N², N²). Call .toarray() when a dense form is needed.
    """
    N   = grid.N
    M   = N * N
    dx2 = grid.dx ** 2

    A = sp.lil_matrix((M, M))

    for i in range(N):
        for j in range(N):
            k = i * N + j

            A[k, k] = -4.0 / dx2

            if j + 1 < N:                   # east
                A[k, k + 1] = 1.0 / dx2
            if j - 1 >= 0:                  # west
                A[k, k - 1] = 1.0 / dx2
            if i + 1 < N:                   # north
                A[k, k + N] = 1.0 / dx2
            if i - 1 >= 0:                  # south
                A[k, k - N] = 1.0 / dx2

    A[0, :] = 0.0
    A[0, 0] = 1.0
    if pin == "symmetric":
        A[:, 0] = 0.0
        A[0, 0] = 1.0
    elif pin != "row":
        raise ValueError(f"Unknown pin='{pin}'. Use 'symmetric' or 'row'.")

    return A.tocsr()


# ---------------------------------------------------------------------------
# Sub-step B.1 (per-step) — Build the RHS vector b
# ---------------------------------------------------------------------------


def build_rhs(
    u_star: np.ndarray,
    v_star: np.ndarray,
    grid:   Grid,
    k:      int = config.GRADIENT_ORDER,
) -> np.ndarray:
    """
    Compute the RHS vector  b = (ρ/Δt) · ∇·u*  and flatten to shape (N²,).

    The divergence is computed by fd_operators.divergence_2d() at stencil
    order k (default from config.GRADIENT_ORDER).

    Parameters
    ----------
    u_star, v_star : ndarray, shape (N+2, N+2)
    grid           : Grid
    k              : gradient stencil half-width
                     k=1 → 3-pt O(h²),  k=2 → 5-pt O(h⁴),  k=3 → 7-pt O(h⁶)

    Returns
    -------
    b : ndarray, shape (N²,)
    """
    div = divergence_2d(u_star, v_star, grid.dx, k=k, dy=grid.dy)
    # div has shape (N+2, N+2); extract interior (N, N) and flatten
    b = (grid.rho / grid.dt) * div[1:-1, 1:-1].ravel()

    b[0] = 0.0   # pin p[0,0] = 0 to remove the pressure null space
    return b


# ---------------------------------------------------------------------------
# Sub-step B.2 — Solve Ax = b
# ---------------------------------------------------------------------------


def solve_pressure(
    A_dense:   np.ndarray,
    b:         np.ndarray,
    grid:      Grid,
    solver_fn=None,
) -> np.ndarray:
    """
    Solve  A · x = b  and return the 2D pressure field  p  of shape (N, N).

    Parameters
    ----------
    A_dense   : ndarray, shape (N², N²) — from build_poisson_matrix().toarray()
    b         : ndarray, shape (N²,)    — from build_rhs()
    grid      : Grid
    solver_fn : callable(A, b) → x.  Defaults to linalg.solve_direct.

    Returns
    -------
    p : ndarray, shape (N, N) — interior pressure field
    """
    if solver_fn is None:
        solver_fn = linalg.solve_direct
    return np.array(solver_fn(A_dense, b)).reshape(grid.N, grid.N)
