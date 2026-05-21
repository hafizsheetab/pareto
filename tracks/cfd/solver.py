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
from grid import Grid
from step_b_pressure import build_poisson_matrix
from uniqx import fmt_mat, to_module
from uniqx.core import types as ut
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

    field_t    = ut.tensor("f64", [N + 2, N + 2])
    interior_t = ut.tensor("f64", [N, N])
    flat_t     = ut.tensor("f64", [Nsq])
    tail_t     = ut.tensor("f64", [Nsq - 1])
    field_flat_t = ut.tensor("f64", [field])
    carry_t    = ut.tensor("f64", [carry_n])
    A_t        = ut.tensor("f64", [Nsq, Nsq])

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
        A_c     = A_param
        carry_0 = _pack_carry(ux.const(u0), ux.const(v0), ux.const(p0))

        def body(_i, carry):
            u, v, p = _split_carry(carry)

            # --- A. Diffusion: u* = u + dt·ν·∇²u  (interior only) -----------
            u_star_int = block(u, 1, 1, N, N) + lap(u, N, inv_dx2) * dt_nu
            v_star_int = block(v, 1, 1, N, N) + lap(v, N, inv_dx2) * dt_nu
            u_star = embed_velocity(u_star_int, N, U_lid)
            v_star = embed_velocity(v_star_int, N, 0.0)

            # --- B. Pressure Poisson: A · x = b = (ρ/Δt)·∇·u* ---------------
            b_field  = div(u_star, v_star, N, inv_2dx) * rho_dt
            b        = ux.reshape(b_field, shape=[Nsq], result_type=flat_t)
            tail     = ux.slice(
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
            p_full = embed_pressure_neumann(p_new, N)
            u_int  = block(u_star, 1, 1, N, N) - grad_x(p_full, N, inv_2dx) * dt_rho
            v_int  = block(v_star, 1, 1, N, N) - grad_y(p_full, N, inv_2dx) * dt_rho
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
