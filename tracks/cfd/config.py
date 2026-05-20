# =============================================================================
# config.py — Single source of truth for all simulation parameters.
# Every other module imports constants from here.
# =============================================================================

# --- Domain ---
N = 8           # interior grid points per side (total grid is (N+2)×(N+2))
L = 1.0         # domain side length [m]

# --- Fluid properties ---
NU  = 0.01      # kinematic viscosity [m²/s]
RHO = 1.0       # density [kg/m³]

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

# --- Solver (Step B / QPU step) ---
PRESSURE_SOLVER   = "direct"    # "direct" | "cg" | "vqls"
SOLVER_TOLERANCE  = 1e-6    # convergence tolerance for iterative solvers

# --- Run control ---
N_STEPS     = 100  # maximum number of time steps
TOL         = 1e-6  # steady-state divergence tolerance (stop early if reached)
PRINT_EVERY = 1   # print diagnostics every this many steps

# --- Backend selector (for benchmarking) ---
# "jax"   → original JAX implementation
# "uniqx" → Uniqx gateway at localhost:50050
BACKEND = "uniqx" #"jax"#  #

# --- Output ---
ASSETS_DIR = "assets"                        # all outputs go here (created automatically)
SAVE_PATH  = f"{ASSETS_DIR}/results.png"     # combined snapshot figure
