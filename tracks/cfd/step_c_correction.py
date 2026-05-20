# =============================================================================
# step_c_correction.py — Step C: Velocity correction (CPU/TPU backend).
#
# Hardware target: CPU or TPU.
#
# Projects u* onto the divergence-free subspace using the pressure field
# returned by Step B:
#
#   u^{n+1} = u* − (Δt/ρ) · ∂p/∂x
#   v^{n+1} = v* − (Δt/ρ) · ∂p/∂y
#
# The pressure gradient is computed by fd_operators.gradient_2d() at the
# same variable stencil order used for the divergence in Step B.
# =============================================================================

import numpy as np
import config
from grid import Grid
from fd_operators import gradient_2d


def correct_velocity(
    u_star: np.ndarray,
    v_star: np.ndarray,
    p:      np.ndarray,
    grid:   Grid,
    k:      int = config.GRADIENT_ORDER,
) -> tuple:
    """
    Apply pressure-gradient correction and return the divergence-free velocity.

    Parameters
    ----------
    u_star, v_star : ndarray, shape (N+2, N+2)
        Intermediate velocity from Step A (BCs already applied).
    p    : ndarray, shape (N, N) — interior pressure from Step B.
    grid : Grid
    k    : gradient stencil half-width (default from config.GRADIENT_ORDER)
           k=1 → 3-pt O(h²),  k=2 → 5-pt O(h⁴),  k=3 → 7-pt O(h⁶)

    Returns
    -------
    u_new, v_new : ndarray, shape (N+2, N+2)
        Corrected velocity; caller must re-apply apply_velocity_bc().
    """
    N = grid.N

    # Embed p in (N+2)×(N+2) with Neumann ghost fill (∂p/∂n = 0 at walls)
    p_full = np.zeros((N + 2, N + 2))
    p_full[1:-1, 1:-1] = p
    p_full[0,  :]  = p_full[1,  :]
    p_full[-1, :]  = p_full[-2, :]
    p_full[:,  0]  = p_full[:,  1]
    p_full[:, -1]  = p_full[:, -2]

    dp_dx, dp_dy = gradient_2d(p_full, grid.dx, k=k, dy=grid.dy)

    u_new = u_star.copy()
    v_new = v_star.copy()

    u_new[1:-1, 1:-1] = u_star[1:-1, 1:-1] - (grid.dt / grid.rho) * dp_dx[1:-1, 1:-1]
    v_new[1:-1, 1:-1] = v_star[1:-1, 1:-1] - (grid.dt / grid.rho) * dp_dy[1:-1, 1:-1]

    return u_new, v_new
