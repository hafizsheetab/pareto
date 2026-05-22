# =============================================================================
# qp_traced_ops.py — kernel-based stencil helpers for the fused TGV NS-RHS.
#
# Lives in a SEPARATE file from the @to_module body in qp_solve_task4.py.
# uniqx's auto-outliner turns same-file Python calls into IR `call` ops,
# which the gateway pipeline rejects for fused-kernel deliverables. Keeping
# these helpers here forces them to be inlined at trace time, so the whole
# NS-RHS becomes a single flat function in the IR.
#
# Differentiation strategy
# ------------------------
# Native `physics.grid_gradient` and `physics.grid_laplacian` kernels emit
# *single* ops that the gateway lowers to its own sparse FD representation —
# no dense matrix is materialized on the wire, only the kernel's grid attrs
# (nx, ny, nz, dx, …, bc). We then apply each operator to a flattened (N³,)
# velocity field with `matmul`. The gradient kernel's output stacks
# [∂f/∂x; ∂f/∂y; ∂f/∂z] (3·N³ rows) per its docstring; we slice that back
# into three (N, N, N) tensors for contraction with u/v/w.
# =============================================================================

import uniqx as ux
from uniqx.core import types as ut
from uniqx.core.enums import BoundaryCondition
from uniqx.domains.physics.kernels import grid_gradient, grid_laplacian


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


def grad_components(f, G, N: int):
    """Apply the gradient operator to scalar field f (N,N,N) and split the
    stacked (3·N³,) output back into three (N, N, N) tensors (∂x, ∂y, ∂z)."""
    Nf = N * N * N
    flat_t = ut.tensor("f64", [Nf])
    stacked_t = ut.tensor("f64", [3 * Nf])
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
    # Silence the linter — stacked_t is part of the documented contract.
    _ = stacked_t

    return (
        ux.reshape(dx_flat, shape=[N, N, N], result_type=field_t),
        ux.reshape(dy_flat, shape=[N, N, N], result_type=field_t),
        ux.reshape(dz_flat, shape=[N, N, N], result_type=field_t),
    )


def lap_field(f, L, N: int):
    """Apply the Laplacian operator to scalar field f (N,N,N) and reshape back."""
    Nf = N * N * N
    flat_t = ut.tensor("f64", [Nf])
    field_t = ut.tensor("f64", [N, N, N])

    f_flat = ux.reshape(f, shape=[Nf], result_type=flat_t)
    lap_flat = ux.matmul(L, f_flat)
    return ux.reshape(lap_flat, shape=[N, N, N], result_type=field_t)
