# =============================================================================
# uniqx_traced_ops.py — kernel-based stencil helpers for the TGV solver.
#
# Lives in a SEPARATE file from the @to_module body in uniqx_solve.py. uniqx's
# auto-outliner turns same-file Python calls into IR `call` ops, which the
# gateway pipeline rejects; keeping these helpers here forces them to be
# inlined at trace time, so the whole RK4 + projection step becomes a single
# flat function in the IR.
#
# All operators are constructed from native physics kernels (one IR op each,
# carrying grid attributes — no dense matrix materialised on the wire):
#
#     grid_gradient   (periodic BC, dim=3)   — for ∂x/∂y/∂z and pressure ∇p
#     grid_laplacian  (periodic BC)          — for ν∇²u in the NS-RHS
#     grid_helmholtz  (periodic BC, k=k_reg) — regularized Poisson operator
#                                              ∇² + k_reg²·I used for the
#                                              pressure projection solve
#                                              (lifts the zero eigenvalue
#                                              of the periodic Laplacian)
# =============================================================================

import uniqx as ux
from uniqx.core import types as ut
from uniqx.core.enums import BoundaryCondition
from uniqx.domains.physics.kernels import (
    grid_gradient,
    grid_helmholtz,
    grid_laplacian,
)


# -----------------------------------------------------------------------------
# Operator-matrix builders
# -----------------------------------------------------------------------------
def build_grad_matrix(N: int, h: float):
    """Periodic 3-D gradient operator reshaped to (3·N³, N³)."""
    Nf = N * N * N
    G_flat = grid_gradient(
        nx=N, ny=N, nz=N,
        dx=h, dy=h, dz=h,
        dim=3,
        bc=BoundaryCondition.PERIODIC,
    )
    return ux.reshape(
        G_flat,
        shape=[3 * Nf, Nf],
        result_type=ut.tensor("f64", [3 * Nf, Nf]),
    )


def build_lap_matrix(N: int, h: float):
    """Periodic 3-D Laplacian operator reshaped to (N³, N³)."""
    Nf = N * N * N
    L_flat = grid_laplacian(
        nx=N, ny=N, nz=N,
        dx=h, dy=h, dz=h,
        bc=BoundaryCondition.PERIODIC,
    )
    return ux.reshape(
        L_flat,
        shape=[Nf, Nf],
        result_type=ut.tensor("f64", [Nf, Nf]),
    )


def build_helmholtz_matrix(N: int, h: float, k_reg: float):
    """Regularized periodic Poisson operator  H = ∇² + k_reg²·I.

    Why this instead of grid_laplacian for the projection solve:
    the periodic ∇² is singular (constant mode has zero eigenvalue),
    so `linear_solve(L, b)` is ill-posed. Adding k_reg²·I lifts every
    eigenvalue by k_reg², making H invertible while leaving |∇p|
    essentially unchanged (the gauge constant contributes 0 to ∇p).
    """
    Nf = N * N * N
    H_flat = grid_helmholtz(
        nx=N, ny=N, nz=N,
        dx=h, dy=h, dz=h,
        k=k_reg,
        bc=BoundaryCondition.PERIODIC,
    )
    return ux.reshape(
        H_flat,
        shape=[Nf, Nf],
        result_type=ut.tensor("f64", [Nf, Nf]),
    )


# -----------------------------------------------------------------------------
# Per-step operator applications
# -----------------------------------------------------------------------------
def grad_components(f, G, N: int):
    """Apply gradient operator to scalar (N,N,N) field; return three (N,N,N)
    tensors  (∂f/∂x, ∂f/∂y, ∂f/∂z)."""
    Nf = N * N * N
    flat_t = ut.tensor("f64", [Nf])
    field_t = ut.tensor("f64", [N, N, N])

    f_flat = ux.reshape(f, shape=[Nf], result_type=flat_t)
    stacked = ux.matmul(G, f_flat)  # (3·N³,)

    dx_flat = ux.slice(
        stacked, start_indices=[0], limit_indices=[Nf],
        result_type=flat_t,
    )
    dy_flat = ux.slice(
        stacked, start_indices=[Nf], limit_indices=[2 * Nf],
        result_type=flat_t,
    )
    dz_flat = ux.slice(
        stacked, start_indices=[2 * Nf], limit_indices=[3 * Nf],
        result_type=flat_t,
    )
    return (
        ux.reshape(dx_flat, shape=[N, N, N], result_type=field_t),
        ux.reshape(dy_flat, shape=[N, N, N], result_type=field_t),
        ux.reshape(dz_flat, shape=[N, N, N], result_type=field_t),
    )


def grad_components_flat(f_flat, G, N: int):
    """Same as grad_components but expects a flat (N³,) input and returns
    three flat (N³,) tensors. Used in the projection block where the
    intermediate flat layout is convenient for the divergence sum."""
    Nf = N * N * N
    flat_t = ut.tensor("f64", [Nf])

    stacked = ux.matmul(G, f_flat)
    dx_flat = ux.slice(
        stacked, start_indices=[0], limit_indices=[Nf],
        result_type=flat_t,
    )
    dy_flat = ux.slice(
        stacked, start_indices=[Nf], limit_indices=[2 * Nf],
        result_type=flat_t,
    )
    dz_flat = ux.slice(
        stacked, start_indices=[2 * Nf], limit_indices=[3 * Nf],
        result_type=flat_t,
    )
    return dx_flat, dy_flat, dz_flat


def lap_field(f, L, N: int):
    """Apply Laplacian operator to scalar (N,N,N) field; return (N,N,N)."""
    Nf = N * N * N
    flat_t = ut.tensor("f64", [Nf])
    field_t = ut.tensor("f64", [N, N, N])

    f_flat = ux.reshape(f, shape=[Nf], result_type=flat_t)
    lap_flat = ux.matmul(L, f_flat)
    return ux.reshape(lap_flat, shape=[N, N, N], result_type=field_t)


# -----------------------------------------------------------------------------
# Shape helpers
# -----------------------------------------------------------------------------
def flat(f, N: int):
    """(N,N,N) → (N³,)."""
    Nf = N * N * N
    return ux.reshape(f, shape=[Nf], result_type=ut.tensor("f64", [Nf]))


def unflat(f_flat, N: int):
    """(N³,) → (N,N,N)."""
    return ux.reshape(
        f_flat, shape=[N, N, N],
        result_type=ut.tensor("f64", [N, N, N]),
    )
