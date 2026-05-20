# =============================================================================
# boundary.py — Enforce boundary conditions for the lid-driven cavity.
#
# Array layout (shape (N+2)×(N+2), row = y-direction):
#   row 0    → y = 0  (bottom wall, no-slip)
#   row N+1  → y = L  (top wall, moving lid at U_lid in x)
#   col 0    → x = 0  (left wall, no-slip)
#   col N+1  → x = L  (right wall, no-slip)
#   rows/cols 1..N    → interior nodes
# =============================================================================

import numpy as np


def apply_velocity_bc(u: np.ndarray, v: np.ndarray, U_lid: float) -> None:
    """Impose Dirichlet velocity BCs on (N+2)×(N+2) arrays (in-place)."""
    # Top lid moves at U_lid; all other walls are no-slip
    u[-1, :] = U_lid   # top wall: u = U_lid
    v[-1, :] = 0.0     # top wall: v = 0

    u[0, :]  = 0.0     # bottom wall
    v[0, :]  = 0.0

    u[:, 0]  = 0.0     # left wall
    v[:, 0]  = 0.0

    u[:, -1] = 0.0     # right wall
    v[:, -1] = 0.0
