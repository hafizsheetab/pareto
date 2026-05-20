# =============================================================================
# step_b_pressure.py — Step B: Pressure Poisson equation (QPU backend).
#
# Hardware target: CPU/TPU for matrix assembly (Sub-step B.1),
#                 QPU (or classical JAX) for the linear solve (Sub-step B.2).
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
# Solver backends live in linalg.py and are dispatched inside solve_pressure():
#   method="direct"  — dense JAX LU            (linalg.solve_direct)
#   method="cg"      — Conjugate Gradient JAX  (linalg.solve_cg)
#   method="vqls"    — QPU stub                (linalg.solve_vqls)
# =============================================================================

import numpy as np
import scipy.sparse as sp

import linalg
import config
from grid import Grid
from fd_operators import divergence_2d


# ---------------------------------------------------------------------------
# Sub-step B.1 — Build the Laplacian matrix A  (called ONCE before the loop)
# ---------------------------------------------------------------------------

def build_poisson_matrix(grid: Grid) -> np.ndarray:
    """
    Assemble the N²×N² discrete Laplacian for the interior pressure nodes.

    Global index mapping: node (i, j) → k = i*N + j,  i,j ∈ {0, …, N-1}.

    Stencil (central differences, uniform dx=dy):
        A[k, k]     = -4 / dx²
        A[k, k±1]   = +1 / dx²   (east/west, skip at row edges)
        A[k, k±N]   = +1 / dx²   (north/south)

    Neumann BC at walls: simply omit the off-domain neighbour term —
    equivalent to ∂p/∂n = 0 at every wall.

    Dirichlet pin at corner k=0 to remove the pressure null space:
        A[0, :] = 0,  A[0, 0] = 1

    Returns
    -------
    A_dense : ndarray, shape (N², N²)
        Dense matrix (converted from sparse for use with JAX backends).
    """
    N  = grid.N
    M  = N * N
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

    # Dirichlet pin at corner node (0,0) → k=0  to fix the null space
    A[0, :] = 0.0
    A[0, 0] = 1.0

    return A.toarray()          # dense numpy array; JAX wraps this fine


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
# Sub-step B.2 — Solve Ax = b via the configured backend
# ---------------------------------------------------------------------------
import time
def solve_pressure(
    A_dense:   np.ndarray,
    b:         np.ndarray,
    grid:      Grid,
    method:    str   = config.PRESSURE_SOLVER,
    tolerance: float = config.SOLVER_TOLERANCE,
) -> np.ndarray:
    """
    Dispatch  A · x = b  to the chosen solver in linalg.py and return the
    2D pressure field  p  of shape (N, N).

    Parameters
    ----------
    A_dense   : ndarray, shape (N², N²) — from build_poisson_matrix()
    b         : ndarray, shape (N²,)    — from build_rhs()
    grid      : Grid
    method    : "direct" | "cg" | "vqls"
    tolerance : convergence tolerance for iterative / quantum solvers

    Returns
    -------
    p : ndarray, shape (N, N) — interior pressure field
    """
    time_start = time.time()
    if method == "direct":
        pressure_vector = linalg.solve_direct(A_dense, b)
        time_finish = time.time()
        print(f"[solver] Pressure solve time: {time_finish - time_start:.3f} seconds")
        
    elif method == "cg":
        pressure_vector = linalg.solve_cg(A_dense, b, tol=tolerance)

    elif method == "vqls":
        pressure_vector = linalg.solve_vqls(A_dense, b, tol=tolerance)

    else:
        raise ValueError(
            f"Unknown pressure solver '{method}'. "
            "Choose from: 'direct', 'cg', 'vqls'."
        )

    return np.array(pressure_vector).reshape(grid.N, grid.N)
