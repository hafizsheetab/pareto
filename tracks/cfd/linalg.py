# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# linalg.py — JAX linear algebra backends for the pressure Poisson solve (Step B).
#
#   solve_direct(A, b)      — dense LU via jnp.linalg.solve
#   solve_cg(A, b, tol)     — Conjugate Gradient via jax.scipy.sparse.linalg.cg
# =============================================================================

import warnings

import config
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
import numpy as np


def solve_direct(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense LU solve A x = b using jnp.linalg.solve."""
    return np.array(jnp.linalg.solve(jnp.array(A), jnp.array(b)))


def solve_cg(A: np.ndarray, b: np.ndarray, tol: float = config.CG_TOL) -> np.ndarray:
    """Conjugate Gradient solve A x = b using jax.scipy.sparse.linalg.cg.

    Expects A already symmetrically pinned (build_poisson_matrix(pin="symmetric")).
    """
    A_jax = jnp.array(A)
    x, info = jsla.cg(lambda v: A_jax @ v, jnp.array(b), tol=tol)
    if info != 0:
        warnings.warn(
            f"CG did not converge (info={info}). "
            "Try increasing CG_TOL in config.py.",
            stacklevel=2,
        )
    return np.array(x)
