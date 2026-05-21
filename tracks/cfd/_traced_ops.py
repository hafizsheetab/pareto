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


def block(f, i0, j0, h, w):
    """f[i0:i0+h, j0:j0+w] as a uniqx slice op (shape (h, w))."""
    return ux.slice(
        f,
        start_indices=[i0, j0],
        limit_indices=[i0 + h, j0 + w],
        result_type=ut.tensor("f64", [h, w]),
    )


def lap(f, N, inv_dx2):
    up    = block(f, 2, 1, N, N)
    down  = block(f, 0, 1, N, N)
    right = block(f, 1, 2, N, N)
    left  = block(f, 1, 0, N, N)
    ctr   = block(f, 1, 1, N, N)
    return (up + down + left + right - ctr * 4.0) * inv_dx2


def div(u, v, N, inv_2dx):
    du_dx = (block(u, 1, 2, N, N) - block(u, 1, 0, N, N)) * inv_2dx
    dv_dy = (block(v, 2, 1, N, N) - block(v, 0, 1, N, N)) * inv_2dx
    return du_dx + dv_dy


def grad_x(f, N, inv_2dx):
    return (block(f, 1, 2, N, N) - block(f, 1, 0, N, N)) * inv_2dx


def grad_y(f, N, inv_2dx):
    return (block(f, 2, 1, N, N) - block(f, 0, 1, N, N)) * inv_2dx


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
