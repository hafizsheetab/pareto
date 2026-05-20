# =============================================================================
# fd_operators.py — Variable-order finite-difference operators for 2D fields.
#
# All operators share the same pad-and-apply engine and support:
#   k=1 →  3-point  O(h²)
#   k=2 →  5-point  O(h⁴)
#   k=3 →  7-point  O(h⁶)
#
# Public API
# ----------
#   laplacian_2d (f,    dx, k, dy)  →  ∇²f           shape (N+2, N+2)
#   gradient_2d  (f,    dx, k, dy)  →  (∂f/∂x, ∂f/∂y)
#   divergence_2d(u, v, dx, k, dy)  →  ∂u/∂x + ∂v/∂y shape (N+2, N+2)
#
# All results are non-zero only at interior nodes [1:-1, 1:-1].
# Near-boundary nodes are handled via edge-padded ghost cells (Neumann
# extension), so the stencil is always applied with full width.
# =============================================================================

import numpy as np


# ── Hardcoded stencil coefficients ────────────────────────────────────────────

def _weights_1st(k: int) -> np.ndarray:
    """
    Central-difference weights for the 1st derivative, unit grid spacing.
        f'(x) ≈ (1/h) · Σ_{j=-k}^{k}  w[j+k] · f(x + j·h)

    Weights are antisymmetric: w[k-j] = -w[k+j],  w[k] = 0.

    k=1  →  3-pt  O(h²)   [-1/2,  0,  1/2]
    k=2  →  5-pt  O(h⁴)   [1/12, -2/3,  0,  2/3, -1/12]
    k=3  →  7-pt  O(h⁶)   [-1/60, 3/20, -3/4,  0,  3/4, -3/20,  1/60]
    """
    if k == 1:
        return np.array([-0.5, 0.0, 0.5])
    elif k == 2:
        return np.array([1.0/12, -2.0/3, 0.0, 2.0/3, -1.0/12])
    elif k == 3:
        return np.array([-1.0/60, 3.0/20, -3.0/4, 0.0, 3.0/4, -3.0/20, 1.0/60])
    else:
        raise ValueError(f"k={k} not supported. Choose k ∈ {{1, 2, 3}}.")


def _weights_2nd(k: int) -> np.ndarray:
    """
    Central-difference weights for the 2nd derivative, unit grid spacing.
        f''(x) ≈ (1/h²) · Σ_{j=-k}^{k}  w[j+k] · f(x + j·h)

    Weights are symmetric: w[k-j] = w[k+j].

    k=1  →  3-pt  O(h²)   [1, -2, 1]
    k=2  →  5-pt  O(h⁴)   [-1, 16, -30, 16, -1] / 12
    k=3  →  7-pt  O(h⁶)   [2, -27, 270, -490, 270, -27, 2] / 180
    """
    if k == 1:
        return np.array([1.0, -2.0, 1.0])
    elif k == 2:
        return np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
    elif k == 3:
        return np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / 180.0
    else:
        raise ValueError(f"k={k} not supported. Choose k ∈ {{1, 2, 3}}.")


# ── Shared stencil engine ─────────────────────────────────────────────────────

def _apply_x(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Apply 1D stencil w along x (columns) at interior nodes, returning the
    raw weighted sum (before dividing by h or h²).

    The stencil half-width k is inferred from len(w) = 2k+1.
    Edge padding (k-1 layers) ensures near-boundary nodes always have full
    stencil coverage.
    """
    k   = len(w) // 2
    N   = f.shape[0] - 2
    pad = max(k - 1, 0)
    fp  = np.pad(f, pad, mode='edge')
    out = np.zeros_like(f)
    for s in range(len(w)):
        shift = s - k
        out[1:-1, 1:-1] += w[s] * fp[k : N+k, k+shift : N+k+shift]
    return out


def _apply_y(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply 1D stencil w along y (rows) at interior nodes (raw sum)."""
    k   = len(w) // 2
    N   = f.shape[0] - 2
    pad = max(k - 1, 0)
    fp  = np.pad(f, pad, mode='edge')
    out = np.zeros_like(f)
    for s in range(len(w)):
        shift = s - k
        out[1:-1, 1:-1] += w[s] * fp[k+shift : N+k+shift, k : N+k]
    return out


# ── Public operators ──────────────────────────────────────────────────────────

def laplacian_2d(
    f:  np.ndarray,
    dx: float,
    k:  int   = 1,
    dy: float = None,
) -> np.ndarray:
    """
    ∇²f = ∂²f/∂x² + ∂²f/∂y²  at all interior nodes.

    Parameters
    ----------
    f  : (N+2, N+2) — full field including boundary nodes
    dx : grid spacing in x (columns)
    k  : stencil half-width  (1 → O(h²),  2 → O(h⁴),  3 → O(h⁶))
    dy : grid spacing in y;  defaults to dx

    Returns
    -------
    ndarray (N+2, N+2), non-zero only at [1:-1, 1:-1]
    """
    if dy is None:
        dy = dx
    w = _weights_2nd(k)
    return _apply_x(f, w) / dx**2 + _apply_y(f, w) / dy**2


def gradient_2d(
    f:  np.ndarray,
    dx: float,
    k:  int   = 1,
    dy: float = None,
) -> tuple:
    """
    (∂f/∂x, ∂f/∂y)  at all interior nodes.

    Returns
    -------
    (df_dx, df_dy) — two ndarrays of shape (N+2, N+2),
    non-zero only at [1:-1, 1:-1]
    """
    if dy is None:
        dy = dx
    w = _weights_1st(k)
    return _apply_x(f, w) / dx, _apply_y(f, w) / dy


def divergence_2d(
    u:  np.ndarray,
    v:  np.ndarray,
    dx: float,
    k:  int   = 1,
    dy: float = None,
) -> np.ndarray:
    """
    ∂u/∂x + ∂v/∂y  at all interior nodes.

    Parameters
    ----------
    u, v : (N+2, N+2) — velocity components
    dx   : grid spacing in x
    k    : stencil half-width
    dy   : grid spacing in y;  defaults to dx

    Returns
    -------
    ndarray (N+2, N+2), non-zero only at [1:-1, 1:-1]
    """
    if dy is None:
        dy = dx
    w = _weights_1st(k)
    return _apply_x(u, w) / dx + _apply_y(v, w) / dy
