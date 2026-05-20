# =============================================================================
# grid.py — Grid geometry and derived simulation parameters.
# Reads constants from config.py; all other modules import Grid from here.
# =============================================================================

from dataclasses import dataclass, field
import numpy as np
import config


@dataclass
class Grid:
    # Primary parameters (default to config values)
    N:     int   = config.N
    L:     float = config.L
    nu:    float = config.NU
    rho:   float = config.RHO
    U_lid: float = config.U_LID

    # Derived geometry (computed in __post_init__)
    dx: float = field(init=False)
    dy: float = field(init=False)
    dt: float = field(init=False)

    # Coordinate arrays (computed in __post_init__)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.dx = self.L / self.N
        self.dy = self.dx

        # Von Neumann stability: dt < dx² / (4·ν)  →  use DT_SAFETY as factor
        self.dt = config.DT_SAFETY * self.dx ** 2 / self.nu

        # All velocity/pressure arrays have shape (N+2)×(N+2).
        # Why N+2 and not N?
        #   N   = number of interior nodes where the PDE is actually solved.
        #   +2  = one boundary node on each side of the domain (walls).
        # Layout along any axis:
        #   index 0        → wall (Dirichlet BC applied here, e.g. no-slip or lid)
        #   indices 1 .. N → interior nodes  ← stencils operate only here
        #   index N+1      → opposite wall
        # Keeping boundary values inside the same array means every interior
        # stencil (e.g. u[i-1], u[i], u[i+1]) is always a valid array access
        # with no special-casing at the edges, and applying BCs is a simple
        # slice assignment (e.g.  u[-1, :] = U_lid).
        self.x = np.linspace(0.0, self.L, self.N + 2)
        self.y = np.linspace(0.0, self.L, self.N + 2)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def __repr__(self):
        return (
            f"Grid(N={self.N}, L={self.L}, dx={self.dx:.4f}, "
            f"dt={self.dt:.2e}, nu={self.nu}, rho={self.rho}, U_lid={self.U_lid})"
        )
