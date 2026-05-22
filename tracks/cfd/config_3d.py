# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# config.py — Single source of truth for all simulation parameters.
# Every other module imports constants from here.
# =============================================================================
import numpy as np
import math
# --- Domain ---
N = 16          # interior grid points per side (total grid is (N+2)×(N+2))
L = 1        # domain side length [m]
DOMAIN = 2 * math.pi * L
RE = 1600
# --- Fluid properties ---
NU  = 0.01      # kinematic viscosity [m²/s]
RHO = 1.0       # density [kg/m³]
V_0 = (RE * NU)/ L

# --- Boundary condition ---
U_LID = 1.0     # lid (top wall) velocity in x-direction [m/s]
                # => Reynolds number Re = U_LID * L / NU = 100 (Stokes regime)

# --- Time stepping ---
# Von Neumann stability for 2D explicit diffusion: dt < dx² / (4·ν)
# We use DT_SAFETY × that bound as a conservative factor.
DT_SAFETY = 0.25    # safety factor (50 % of the theoretical 2D limit)

# --- Step A: Laplacian stencil order ---
# k = stencil half-width; accuracy = O(h^{2k})
#   k=1 →  3-point  O(h²)   [default, cheapest]
#   k=2 →  5-point  O(h⁴)
#   k=3 →  7-point  O(h⁶)
LAPLACIAN_ORDER = 1   # used in Step A (∇²u)
GRADIENT_ORDER  = 1   # used in Step B (∇·u*) and Step C (∇p)

# --- Solver (Step B) ---
PRESSURE_SOLVER = "direct"  # "direct" | "cg"
CG_TOL          = 1e-6      # CG residual convergence tolerance

# --- Run control ---
N_STEPS     = 100   # maximum number of time steps
DIV_TOL     = 1e-6  # steady-state velocity divergence tolerance (early stop)
PRINT_EVERY = 50    # print diagnostics every this many steps

# --- Output ---
ASSETS_DIR = "assets"                        # all outputs go here (created automatically)
SAVE_PATH  = f"{ASSETS_DIR}/results.png"     # combined snapshot figure
