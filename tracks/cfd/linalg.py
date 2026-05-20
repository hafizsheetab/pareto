# =============================================================================
# linalg.py — Linear algebra backends for the pressure Poisson solve (Step B).
#
# Two backends are available; select via config.BACKEND:
#   "jax"   — JAX dense LU / CG (CPU/GPU/TPU)
#   "uniqx" — Uniqx gateway at localhost:50050
#
# Public interface (unchanged for callers):
#   solve_direct(A, b)          — exact dense solve
#   solve_cg(A, b, tol)         — Conjugate Gradient
#   solve_vqls(A, b, tol)       — VQLS via QPU (always routes to Uniqx)
# =============================================================================

import warnings
import numpy as np
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla

from uniqx import to_module, fmt_mat, fmt_vec, parse_result, run as ux_run
from uniqx.ops.primitives.solvers import linear_solve
from uniqx.core.modality import Modality
from uniqx_client import get_client, get_or_build
import config


# =============================================================================
# JAX backends
# =============================================================================

def solve_direct_jax(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = jnp.linalg.solve(jnp.array(A), jnp.array(b)) #  A x = b --> x = A⁻¹ b
    return np.array(x)


def solve_cg_jax(A: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    A_jax = jnp.array(A)
    b_jax = jnp.array(b)
    x, info = jsla.cg(lambda v: A_jax @ v, b_jax, tol=tol)
    if info != 0:
        warnings.warn(
            f"CG did not converge (info={info}). "
            "Try increasing SOLVER_TOLERANCE in config.py.",
            stacklevel=2,
        )
    return np.array(x)


# =============================================================================
# Uniqx backends
# =============================================================================

# --- Traced module builders (called once per matrix size M = N²) ---

#@to_module(name="pressure_solve_direct")
def _direct_fn(A_mat, b_vec):
    return linear_solve(A_mat, b_vec, sparse=False, positive_definite=False)

#@to_module(name="pressure_solve_cg")
def _cg_fn(A_mat, b_vec):
    return linear_solve(A_mat, b_vec, sparse=True, precision=config.SOLVER_TOLERANCE)

#@to_module(name="pressure_solve_vqls")
def _vqls_fn(A_mat, b_vec):
    return linear_solve(
        A_mat, b_vec,
        sparse=True,
        precision=config.SOLVER_TOLERANCE,
        modality=Modality.QPU,
    )


def _build_module(fn, M):
    return fn([[0.0] * M for _ in range(M)], [0.0] * M)


def _parse_vector(res) -> np.ndarray:
    payload = res.get("payload") or res.get("result_payload") or b""
    parsed = parse_result(payload, ["x"])
    _, _, vals = parsed["x"]
    return np.array(vals)


def _symmetrize_pinned(A: np.ndarray) -> np.ndarray:
    # DGESV (our Linux LU override) accepts asymmetric matrices, but we still
    # zero col 0 for correctness: the Dirichlet pin sets x[0]=0, so col 0
    # terms vanish. Restoring symmetry also makes the matrix more robust.
    A_sym = A.copy()
    A_sym[:, 0] = 0.0
    A_sym[0, 0] = 1.0
    return A_sym


# Cache the formatted matrix string by object identity — the Poisson matrix
# is fixed for the entire time loop, so we compute fmt_mat once.
_matrix_fmt_cache: dict = {}


def _get_matrix_fmt(A: np.ndarray, M: int) -> str:
    key = id(A)
    if key not in _matrix_fmt_cache:
        _matrix_fmt_cache[key] = fmt_mat(_symmetrize_pinned(A).tolist(), M, M)
    return _matrix_fmt_cache[key]


def solve_direct_uniqx(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    M = len(b)
    return get_or_build(("direct", M), lambda: _build_module(_direct_fn, M))
#    ri = [_get_matrix_fmt(A, M), fmt_vec(b.tolist(), M)]
#    return _parse_vector(ux_run(mod, client=get_client(), runtime_inputs=ri, mode="manual"))


def solve_cg_uniqx(A: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    # tol is baked into the module at trace time from config.SOLVER_TOLERANCE
    M = len(b)
    mod = get_or_build(("cg", M), lambda: _build_module(_cg_fn, M))
    ri = [_get_matrix_fmt(A, M), fmt_vec(b.tolist(), M)]
    return _parse_vector(ux_run(mod, client=get_client(), runtime_inputs=ri, mode="manual"))


def solve_vqls_uniqx(A: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    # tol is baked into the module at trace time from config.SOLVER_TOLERANCE
    M = len(b)
    mod = get_or_build(("vqls", M), lambda: _build_module(_vqls_fn, M))
    ri = [_get_matrix_fmt(A, M), fmt_vec(b.tolist(), M)]
    return _parse_vector(ux_run(mod, client=get_client(), runtime_inputs=ri, mode="manual"))


# =============================================================================
# Public dispatchers
# =============================================================================

def solve_direct(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if config.BACKEND == "uniqx":
        return solve_direct_uniqx(A, b)
    return solve_direct_jax(A, b)


def solve_cg(A: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    if config.BACKEND == "uniqx":
        return solve_cg_uniqx(A, b, tol)
    return solve_cg_jax(A, b, tol)


def solve_vqls(A: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    # VQLS is inherently quantum — always routes to Uniqx regardless of BACKEND
    return solve_vqls_uniqx(A, b, tol)