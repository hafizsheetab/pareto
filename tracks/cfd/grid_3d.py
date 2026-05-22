# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# grid.py — Grid geometry and derived simulation parameters.
# Reads constants from config.py; all other modules import Grid from here.
# =============================================================================

from dataclasses import dataclass, field
import math
import config_3d as config


@dataclass
class Grid3d:
    # Primary parameters (default to config values)
    N:   int   = config.N
    L:   float = config.L
    domain: float = 2 * math.pi * config.L
    nu:  float = config.NU
    rho: float = config.RHO

    # Derived geometry (computed in __post_init__)
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self):
        self.dx = self.L / self.N
        self.dy = self.dx
        self.dz = self.dx
        # Von Neumann stability: dt < dx² / (4·ν)  →  use DT_SAFETY as factor
        self.dt = config.DT_SAFETY * self.dx ** 2 / self.nu

    def __repr__(self):
        return (
            f"Grid(N={self.N}, L={self.L}, dx={self.dx:.4f}, "
            f"dt={self.dt:.2e}, nu={self.nu}, rho={self.rho})"
        )
