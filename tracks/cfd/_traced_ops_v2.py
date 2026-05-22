# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# _traced_ops.py — uniqx-op helpers used inside the traced iteration body.
#
# These MUST live in a separate file from the @to_module function. uniqx
# tracing inlines helpers that are not in the same source file as the traced
# body; keeping them here forces that inlining at trace time.
# =============================================================================

import uniqx as ux
from uniqx.core import types as ut
import config
import uniqx.domains.physics.kernels as uqk
from typing import Literal
from uniqx.core.enums import BoundaryCondition


def block(f, i0, j0, h, w):
    """f[i0:i0+h, j0:j0+w] as a uniqx slice op (shape (h, w))."""
    return ux.slice(
        f,
        start_indices=[i0, j0],
        limit_indices=[i0 + h, j0 + w],
        result_type=ut.tensor("f64", [h, w]),
    )


def _grid_params(N, dx, dy, dz, dim):
    if dim == "2D":
        nz, dz_val, dim_n = 1, 1.0, 2
    else:
        nz, dz_val, dim_n = N, dz, 3
    n_total = N * N * nz
    return nz, dz_val, dim_n, n_total


def lap(N, dx, dy, dz, dim: Literal["2D", "3D"], bc=BoundaryCondition.DIRICHLET):
    nz, dz_val, _, _ = _grid_params(N, dx, dy, dz, dim)
    return uqk.grid_laplacian(nx=N, ny=N, nz=nz, dx=dx, dy=dy, dz=dz_val, bc=bc)


def div(N, dx, dy, dz, dim: Literal["2D", "3D"], bc=BoundaryCondition.DIRICHLET):
    nz, dz_val, dim_n, _ = _grid_params(N, dx, dy, dz, dim)
    return uqk.grid_divergence(nx=N, ny=N, nz=nz, dx=dx, dy=dy, dz=dz_val, dim=dim_n, bc=bc)


def grad_all(N, dx, dy, dz, dim: Literal["2D", "3D"], bc=BoundaryCondition.DIRICHLET):
    nz, dz_val, dim_n, _ = _grid_params(N, dx, dy, dz, dim)
    return uqk.grid_gradient(nx=N, ny=N, nz=nz, dx=dx, dy=dy, dz=dz_val, dim=dim_n, bc=bc)


def grad_x(N, dx, dy, dz, dim: Literal["2D", "3D"], bc=BoundaryCondition.DIRICHLET):
    nz, dz_val, dim_n, n_total = _grid_params(N, dx, dy, dz, dim)
    G = uqk.grid_gradient(nx=N, ny=N, nz=nz, dx=dx, dy=dy, dz=dz_val, dim=dim_n, bc=bc)
    size = n_total * n_total
    return ux.slice(G, [0], [size], result_type=ut.tensor("f64", [size]))


def grad_y(N, dx, dy, dz, dim: Literal["2D", "3D"], bc=BoundaryCondition.DIRICHLET):
    nz, dz_val, dim_n, n_total = _grid_params(N, dx, dy, dz, dim)
    G = uqk.grid_gradient(nx=N, ny=N, nz=nz, dx=dx, dy=dy, dz=dz_val, dim=dim_n, bc=bc)
    size = n_total * n_total
    return ux.slice(G, [size], [2 * size], result_type=ut.tensor("f64", [size]))


def grad_z(N, dx, dy, dz, dim: Literal["2D", "3D"], bc=BoundaryCondition.DIRICHLET):
    if dim != "3D":
        raise ValueError("grad_z requires 3D dimension")
    nz, dz_val, dim_n, n_total = _grid_params(N, dx, dy, dz, dim)
    G = uqk.grid_gradient(nx=N, ny=N, nz=nz, dx=dx, dy=dy, dz=dz_val, dim=dim_n, bc=bc)
    size = n_total * n_total
    return ux.slice(G, [2 * size], [3 * size], result_type=ut.tensor("f64", [size]))




def embed_velocity(interior, N, top_value):
    """(N, N) interior → (N+2, N+2) with no-slip walls and a lid at the top."""
    zero_col = [[0.0]] * N
    zero_row = [[0.0] * (N + 2)]
    top_row  = [[top_value] * (N + 2)]
    middle = ux.concatenate(
        zero_col, interior, zero_col,
        axis=1,
        result_type=ut.tensor("f64", [N, N + 2]),
    )
    return ux.concatenate(
        zero_row, middle, top_row,
        axis=0,
        result_type=ut.tensor("f64", [N + 2, N + 2]),
    )


def embed_pressure_neumann(p, N):
    """(N, N) pressure → (N+2, N+2) with Neumann ghost cells (∂p/∂n = 0)."""
    left_col  = block(p, 0, 0,     N, 1)
    right_col = block(p, 0, N - 1, N, 1)
    middle = ux.concatenate(
        left_col, p, right_col,
        axis=1,
        result_type=ut.tensor("f64", [N, N + 2]),
    )
    top_row = block(middle, 0,     0, 1, N + 2)
    bot_row = block(middle, N - 1, 0, 1, N + 2)
    return ux.concatenate(
        top_row, middle, bot_row,
        axis=0,
        result_type=ut.tensor("f64", [N + 2, N + 2]),
    )
