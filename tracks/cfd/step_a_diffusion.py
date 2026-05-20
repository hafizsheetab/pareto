# =============================================================================
# step_a_diffusion.py — Step A: Explicit viscous diffusion (GPU/TPU backend).
#
# Hardware target: GPU or TPU.
# NumPy slice operations map 1-to-1 onto CuPy or jax.numpy.
#
#   u* = u^n + dt · ν · ∇²u^n
#   v* = v^n + dt · ν · ∇²v^n
#
# The Laplacian is computed by fd_operators.laplacian_2d(), which supports
# variable stencil order (see fd_operators.py for coefficient tables).
# =============================================================================

import numpy as np
import config
from grid import Grid
from fd_operators import laplacian_2d


def diffuse(
    u:    np.ndarray,
    v:    np.ndarray,
    grid: Grid,
    k:    int = config.LAPLACIAN_ORDER,
) -> tuple:
    """
    Return intermediate velocity (u*, v*) after one explicit diffusion step.

    Hardware target: GPU / TPU (NumPy ops swap directly to CuPy / jax.numpy).

    Parameters
    ----------
    u, v : ndarray, shape (N+2, N+2) — current velocity (includes BCs)
    grid : Grid
    k    : Laplacian stencil half-width (default from config.LAPLACIAN_ORDER)
           k=1 → 3-pt O(h²),  k=2 → 5-pt O(h⁴),  k=3 → 7-pt O(h⁶)

    Returns
    -------
    u_star, v_star : ndarray, shape (N+2, N+2)
        Boundary nodes are copied but BCs are not re-applied here — the
        caller must call apply_velocity_bc() after this function.
    """
    lap_u = laplacian_2d(u, grid.dx, k=k, dy=grid.dy)
    lap_v = laplacian_2d(v, grid.dx, k=k, dy=grid.dy)

    u_star = u.copy()
    v_star = v.copy()

    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + grid.dt * grid.nu * lap_u[1:-1, 1:-1]
    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + grid.dt * grid.nu * lap_v[1:-1, 1:-1]

    return u_star, v_star
