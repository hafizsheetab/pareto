# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# solver.py — Build one Uniqx IR module containing the full Stokes iteration.
#
# Optimizations applied:
#   1. ux.fori_loop with body traced ONCE into a sub-function and a `for` op
#      that references it. Module size is O(1) in N_STEPS instead of O(N_STEPS),
#      so the gateway no longer hits a per-job size ceiling.
#   2. Symmetric pin of the Poisson matrix (row 0 AND col 0 zeroed, A[0,0]=1).
#      linear_solve is told hermitian=True so the gateway can pick a Cholesky-
#      class solver instead of a generic LU.
#   3. Poisson matrix A is a *runtime input*, not an IR const. Tracing time
#      and module text were both O(M²) when A was baked in (N=32 → 765 ms
#      build); promoting A to a module parameter makes build O(1) regardless
#      of N. The matrix is encoded once via `fmt_mat` at submit time.
#   4. Stencil helpers live in _traced_ops.py to bypass uniqx's auto-outliner
#      (same-file Python calls become IR `call` ops, which the gateway CPU
#      pipeline rejects).
#   5. Single-tensor return: (u, v, p) are flattened+concatenated into one
#      output, because the gateway response carries only the first output.
#
# `run()` returns (module, runtime_inputs). Callers submit with:
#   ux.submit(mod, runtime_inputs=runtime_inputs, ...)
# =============================================================================

import config
import uniqx as ux
from _traced_ops import (
    block,
    div,
    embed_pressure_neumann,
    embed_velocity,
    grad_x,
    grad_y,
    lap,
)
import _traced_ops_v2 as v2
from grid import Grid
from step_b_pressure import build_poisson_matrix
from uniqx import fmt_mat, to_module
from uniqx.core import types as ut
from uniqx.core.enums import BoundaryCondition
from uniqx.ops.control_flow import fori_loop
from uniqx.ops.primitives.solvers import linear_solve


def run(grid: Grid, n_steps: int = config.N_STEPS, U_lid: float = config.U_LID):
    """
    Trace and return (module, runtime_inputs) for n_steps of the simulation.

    The Poisson matrix A is a module parameter, not an IR const, so the
    traced module text stays small regardless of grid size.
    """
    N        = grid.N
    Nsq      = N * N
    field    = (N + 2) * (N + 2)
    carry_n  = 2 * field + Nsq                      # u | v | p packed flat

    dt_nu    = grid.dt * grid.nu
    inv_dx2  = 1.0 / (grid.dx ** 2)
    inv_2dx  = 1.0 / (2.0 * grid.dx)
    rho_dt   = grid.rho / grid.dt
    dt_rho   = grid.dt / grid.rho

    A_pinned = build_poisson_matrix(grid, pin="symmetric").toarray()

    # Initial state, baked as a const carry: u/v zero with lid on top, p zero.
    u0 = [[0.0] * (N + 2) for _ in range(N + 2)]
    u0[-1] = [U_lid] * (N + 2)
    v0 = [[0.0] * (N + 2) for _ in range(N + 2)]
    p0 = [[0.0] * N for _ in range(N)]

    field_t       = ut.tensor("f64", [N + 2, N + 2])
    interior_t    = ut.tensor("f64", [N, N])
    flat_t        = ut.tensor("f64", [Nsq])
    double_flat_t = ut.tensor("f64", [2 * Nsq])
    div_op_t      = ut.tensor("f64", [Nsq, 2 * Nsq])
    grad_op_t     = ut.tensor("f64", [2 * Nsq, Nsq])
    lap_pad_t     = ut.tensor("f64", [field, field])
    tail_t        = ut.tensor("f64", [Nsq - 1])
    field_flat_t  = ut.tensor("f64", [field])
    carry_t       = ut.tensor("f64", [carry_n])
    A_t           = ut.tensor("f64", [Nsq, Nsq])

    def _split_carry(carry):
        """carry (264,) → (u (10,10), v (10,10), p (8,8))."""
        u_flat = ux.slice(
            carry, start_indices=[0], limit_indices=[field],
            result_type=field_flat_t,
        )
        v_flat = ux.slice(
            carry, start_indices=[field], limit_indices=[2 * field],
            result_type=field_flat_t,
        )
        p_flat = ux.slice(
            carry, start_indices=[2 * field], limit_indices=[carry_n],
            result_type=flat_t,
        )
        u = ux.reshape(u_flat, shape=[N + 2, N + 2], result_type=field_t)
        v = ux.reshape(v_flat, shape=[N + 2, N + 2], result_type=field_t)
        p = ux.reshape(p_flat, shape=[N, N],         result_type=interior_t)
        return u, v, p

    def _pack_carry(u, v, p):
        u_flat = ux.reshape(u, shape=[field], result_type=field_flat_t)
        v_flat = ux.reshape(v, shape=[field], result_type=field_flat_t)
        p_flat = ux.reshape(p, shape=[Nsq],   result_type=flat_t)
        return ux.concatenate(u_flat, v_flat, p_flat, axis=0, result_type=carry_t)

    @to_module(name="stokes_iterate")
    def iterate(A_param):
        # A_param is the (Nsq, Nsq) Poisson matrix supplied at submit time.
        A_c = A_param

        carry_0 = _pack_carry(ux.const(u0), ux.const(v0), ux.const(p0))

        def body(_i, carry):
            # IMPORTANT: emit kernel operators INSIDE the body. Hoisting them
            # outside `fori_loop` (so they emit once into IR) traces fine
            # locally but the gateway pipeline fails to lower kernel ops
            # captured by an outer scope; jobs come back with "unknown error".
            # Re-emitting per iteration is trace-time only — the gateway is
            # free to dedup; the runtime cost is unchanged.
            #
            # BC choices:
            #   • Laplacian DIRICHLET on the (N+2)² padded grid — the lid
            #     value is already baked into u_padded by embed_velocity,
            #     so the zero-Dirichlet ghosts outside the padded grid only
            #     contaminate rows the inner `block(_, 1, 1, N, N)` discards.
            #   • Divergence DIRICHLET on the N×N interior — the only ghost
            #     the central-diff stencil reaches outside the interior is
            #     v at the lid (=0), which matches DIRICHLET=0 exactly.
            #   • Gradient NEUMANN on the N×N interior — replaces the
            #     `embed_pressure_neumann` ghost-cell scheme one-for-one.
            L_pad = v2.lap(
                N=N + 2, dx=grid.dx, dy=grid.dy, dz=1.0, dim="2D",
                bc=BoundaryCondition.DIRICHLET,
            )
            L_pad = ux.reshape(L_pad, shape=[field, field], result_type=lap_pad_t)

            G_div = v2.div(
                N=N, dx=grid.dx, dy=grid.dy, dz=1.0, dim="2D",
                bc=BoundaryCondition.DIRICHLET,
            )
            G_div_op = ux.reshape(G_div, shape=[Nsq, 2 * Nsq], result_type=div_op_t)

            G_grad = v2.grad_all(
                N=N, dx=grid.dx, dy=grid.dy, dz=1.0, dim="2D",
                bc=BoundaryCondition.NEUMANN,
            )
            G_grad_op = ux.reshape(G_grad, shape=[2 * Nsq, Nsq], result_type=grad_op_t)

            u, v, p = _split_carry(carry)

            # --- A. Diffusion: u* = u + dt·ν·∇²u  (laplacian on padded grid) -
            u_pad_flat = ux.reshape(u, shape=[field], result_type=field_flat_t)
            v_pad_flat = ux.reshape(v, shape=[field], result_type=field_flat_t)

            lap_u_pad_flat = ux.matmul(L_pad, u_pad_flat)
            lap_v_pad_flat = ux.matmul(L_pad, v_pad_flat)

            lap_u_2d = ux.reshape(lap_u_pad_flat, shape=[N + 2, N + 2], result_type=field_t)
            lap_v_2d = ux.reshape(lap_v_pad_flat, shape=[N + 2, N + 2], result_type=field_t)

            u_int_prev = block(u, 1, 1, N, N)
            v_int_prev = block(v, 1, 1, N, N)
            lap_u_int  = block(lap_u_2d, 1, 1, N, N)
            lap_v_int  = block(lap_v_2d, 1, 1, N, N)

            u_star_int = u_int_prev + lap_u_int * dt_nu
            v_star_int = v_int_prev + lap_v_int * dt_nu

            # --- B. Pressure Poisson: A · x = b = (ρ/Δt)·∇·u* ---------------
            u_star_flat = ux.reshape(u_star_int, shape=[Nsq], result_type=flat_t)
            v_star_flat = ux.reshape(v_star_int, shape=[Nsq], result_type=flat_t)
            u_plus_v_flat = ux.concatenate(
                u_star_flat, v_star_flat, axis=0, result_type=double_flat_t,
            )
            div_val_flat = ux.matmul(G_div_op, u_plus_v_flat)

            b = div_val_flat * rho_dt
            tail = ux.slice(
                b, start_indices=[1], limit_indices=[Nsq],
                result_type=tail_t,
            )
            b_pinned = ux.concatenate([0.0], tail, axis=0, result_type=flat_t)
            x = linear_solve(
                A_c, b_pinned,
                sparse=False,
                hermitian=True,
                positive_definite=False,
            )
            p_new = ux.reshape(x, shape=[N, N], result_type=interior_t)

            # --- C. Correction: u^{n+1} = u* − (Δt/ρ)·∇p --------------------
            grad_p_flat = ux.matmul(G_grad_op, x)
            u_v_new_flat = u_plus_v_flat - grad_p_flat * dt_rho

            u_int_flat = ux.slice(
                u_v_new_flat, start_indices=[0], limit_indices=[Nsq],
                result_type=flat_t,
            )
            v_int_flat = ux.slice(
                u_v_new_flat, start_indices=[Nsq], limit_indices=[2 * Nsq],
                result_type=flat_t,
            )
            u_int = ux.reshape(u_int_flat, shape=[N, N], result_type=interior_t)
            v_int = ux.reshape(v_int_flat, shape=[N, N], result_type=interior_t)

            u_new = embed_velocity(u_int, N, U_lid)
            v_new = embed_velocity(v_int, N, 0.0)

            return _pack_carry(u_new, v_new, p_new)

        return fori_loop(0, n_steps, body, carry_0)

    # Hand the tracer an ir.Type directly instead of a sample value — this
    # skips O(M²) shape inference on a placeholder nested list. The actual
    # matrix is shipped via runtime_inputs at submit time.
    mod = iterate(A_t)

    runtime_inputs = [fmt_mat(A_pinned.tolist(), Nsq, Nsq)]
    return mod, runtime_inputs
