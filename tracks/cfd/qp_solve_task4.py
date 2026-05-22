# =============================================================================
# qp_solve_task4.py — Challenge 4: kernel-fusion design + Python prototype.
#
# Design notes
# ============
#
# Time integration scheme
# -----------------------
# At Re = 1600 the Taylor-Green vortex decay is smooth — no shocks, no sharp
# fronts. Comparing the two natural candidates:
#
#   classical RK4   : 4 stages, O(Δt⁴), 4 register arrays, no SSP guarantee
#   low-storage RK3 : 3 stages, O(Δt³), 2 register arrays, SSP (TVD)
#
# We pick **classical RK4**: O(Δt⁴) accuracy lets us take ~2× larger Δt than
# SSP-RK3 at the same global error, so fewer RHS evaluations per simulated
# time. SSP's monotonicity guarantee is wasted on smooth TGV decay, and the
# 4-array memory cost is trivial at 32³ (4 × 32³ × 8 B ≈ 1 MB — well within
# any modern GPU L2). The *traced* primitive is the NS-RHS itself; RK4 stage
# assembly lives in host code and calls the traced module four times per step.
#
# Kernel fusion via native physics ops
# ------------------------------------
# The NS-RHS needs all 9 partial derivatives ∂uᵢ/∂xⱼ plus ∇²uᵢ. The naive
# layout writes each as its own stencil pass into its own intermediate
# tensor — 15 separate ops, ≥15 × N³ words of memory traffic, 9 named
# gradient tensors held live.
#
# We avoid both Fourier transforms and hand-rolled stencils. Instead we
# use uniqx's native physics kernels:
#
#   physics.grid_gradient   (periodic BC, dim=3)
#   physics.grid_laplacian  (periodic BC)
#
# Each emits a *single* IR op carrying grid attributes (nx, ny, nz, dx, dy,
# dz, bc). The gateway lowers them to its own sparse FD representation —
# no dense matrix is materialised on the wire. Applying them via `matmul`
# yields one stacked gradient (∂x, ∂y, ∂z) and one Laplacian per velocity
# component. The full NS-RHS becomes:
#
#   • 2 kernel ops (G and L), reused across u, v, w
#   • 6 matmuls   (3 gradient + 3 Laplacian)
#   • 9 multiplies + 6 adds for the advection contraction
#   • 3 ν·∇²u multiplies and 3 final subtractions
#   • 3 reshapes + 1 concatenate to pack the output for the gateway
#
# Memory traffic drops to ~6 × N³ on the host wire (3 reads, 3 writes);
# every gradient and Laplacian intermediate stays in gateway scratch.
#
# Why h, nu, and the grid extents are NOT traced parameters
# ---------------------------------------------------------
# Grid spacing, viscosity, and N are baked into the trace as Python values.
# The kernel ops require `nx`/`ny`/`nz`/`dx` etc. as static attributes
# (not TracerValues), and all `slice`/`reshape` ops need shape known at
# trace time. To sweep over (N, h, ν) you build a fresh module per combo.
# =============================================================================

import numpy as np

import config_3d as config
import uniqx as ux
from uniqx import to_module
from uniqx.core import types as ut

from qp_traced_ops import build_grad_matrix, build_lap_matrix, grad_components, lap_field

DEFAULT_GATEWAY = "api.oriqx.com:443"
DEFAULT_API_KEY = "uxk_1bdb37b0f52f9d89260d86f2d21e9513"


# -----------------------------------------------------------------------------
# Traced fused NS-RHS — one @to_module trace, all 9 gradients + contractions.
# -----------------------------------------------------------------------------
def build_ns_rhs(N: int, h: float, nu: float):
    """Build and return the fused NS-RHS IR module.

    The traced function takes three (N, N, N) tensors (u, v, w) and returns
    three (N, N, N) tensors (rhs_u, rhs_v, rhs_w), where

        rhs_i = -u_j ∂_j u_i + ν ∂_j ∂_j u_i

    Pressure projection is applied outside this kernel — keeping the trace
    purely local-stencil keeps it a single fused dispatch.

    Implementation note
    -------------------
    Differentiation uses the native `physics.grid_gradient` and
    `physics.grid_laplacian` kernels (periodic BC). Each emits a single
    kernel op carrying grid attributes; the gateway is free to lower them
    to a sparse stencil sweep without ever materialising the dense matrix.
    All 9 partial derivatives reduce to **two** matmul ops per velocity
    component (one gradient, one Laplacian), six total.
    """
    Nf = N * N * N
    field_t = ut.tensor("f64", [N, N, N])
    flat_t = ut.tensor("f64", [Nf])
    out_t = ut.tensor("f64", [3 * Nf])

    @to_module(name="ns_rhs")
    def fused_ns_rhs(u, v, w):
        # Operator matrices — emitted once and reused across u, v, w.
        G = build_grad_matrix(N, h)
        L = build_lap_matrix(N, h)

        # Gradients ∂u/∂x_j via one matmul each (kernel splits the output
        # into 3 stacked components, then we slice + reshape).
        dudx, dudy, dudz = grad_components(u, G, N)
        dvdx, dvdy, dvdz = grad_components(v, G, N)
        dwdx, dwdy, dwdz = grad_components(w, G, N)

        # Advection (u·∇)u — immediate contraction in physical space.
        adv_u = u * dudx + v * dudy + w * dudz
        adv_v = u * dvdx + v * dvdy + w * dvdz
        adv_w = u * dwdx + v * dwdy + w * dwdz

        # Viscous ν∇²u — one matmul per component.
        visc_u = lap_field(u, L, N) * nu
        visc_v = lap_field(v, L, N) * nu
        visc_w = lap_field(w, L, N) * nu

        rhs_u = visc_u - adv_u
        rhs_v = visc_v - adv_v
        rhs_w = visc_w - adv_w

        # The gateway response carries only the first output, so flatten and
        # concatenate all three RHS components into one tensor of length 3·N³.
        # Host code slices it back into (rhs_u, rhs_v, rhs_w) after parsing.
        rhs_u_f = ux.reshape(rhs_u, shape=[Nf], result_type=flat_t)
        rhs_v_f = ux.reshape(rhs_v, shape=[Nf], result_type=flat_t)
        rhs_w_f = ux.reshape(rhs_w, shape=[Nf], result_type=flat_t)
        return ux.concatenate(rhs_u_f, rhs_v_f, rhs_w_f, axis=0, result_type=out_t)

    return fused_ns_rhs(field_t, field_t, field_t)


# -----------------------------------------------------------------------------
# NumPy reference — same algorithm, executed eagerly. Used by the unit test
# and as the per-stage RHS in a future host-side RK4 loop.
# -----------------------------------------------------------------------------
def ns_rhs_numpy_ref(u, v, w, h, nu):
    inv_2dx = 1.0 / (2.0 * h)
    inv_dx2 = 1.0 / (h * h)

    def d(f, axis):
        return (np.roll(f, -1, axis=axis) - np.roll(f, +1, axis=axis)) * inv_2dx

    def lap(f):
        return (
            np.roll(f, -1, axis=0) + np.roll(f, +1, axis=0)
            + np.roll(f, -1, axis=1) + np.roll(f, +1, axis=1)
            + np.roll(f, -1, axis=2) + np.roll(f, +1, axis=2)
            - 6.0 * f
        ) * inv_dx2

    dudx, dudy, dudz = d(u, 0), d(u, 1), d(u, 2)
    dvdx, dvdy, dvdz = d(v, 0), d(v, 1), d(v, 2)
    dwdx, dwdy, dwdz = d(w, 0), d(w, 1), d(w, 2)

    adv_u = u * dudx + v * dudy + w * dudz
    adv_v = u * dvdx + v * dvdy + w * dvdz
    adv_w = u * dwdx + v * dwdy + w * dwdz

    return (
        nu * lap(u) - adv_u,
        nu * lap(v) - adv_v,
        nu * lap(w) - adv_w,
    )


# -----------------------------------------------------------------------------
# Gateway submission helpers
# -----------------------------------------------------------------------------
def _fmt_3d(arr: np.ndarray) -> str:
    """Encode an (N, N, N) f64 array as a buffer-view string the gateway accepts."""
    a, b, c = arr.shape
    flat = arr.reshape(-1).tolist()
    return f"{a}x{b}x{c}xf64= " + " ".join(repr(x) for x in flat)


def _parse_flat_payload(payload) -> np.ndarray:
    text = payload.decode("latin-1") if isinstance(payload, (bytes, bytearray)) else payload
    _, _, values = text.strip().partition("=")
    return np.fromstring(values, sep=" ", dtype=np.float64)


def submit_and_fetch(mod, u, v, w, N: int,
                     gateway: str = DEFAULT_GATEWAY,
                     api_key: str = DEFAULT_API_KEY):
    """Submit the traced module with (u, v, w) as runtime inputs and return
    three (N, N, N) numpy arrays (rhs_u, rhs_v, rhs_w)."""
    print(f"[task4] connecting to {gateway} …", flush=True)
    client = ux.connect(gateway, api_key=api_key)

    runtime_inputs = [_fmt_3d(u), _fmt_3d(v), _fmt_3d(w), "backend=compiled"]
    print("[task4] submitting fused ns_rhs trace …", flush=True)
    job_id = ux.submit(mod, client=client, runtime_inputs=runtime_inputs)
    print(f"[task4] job_id = {job_id}", flush=True)

    res = ux.get(job_id, client=client, timeout=600.0)
    if res.get("state") != 10:
        payload = res.get("payload") or res.get("result_payload") or b""
        raise SystemExit(f"[task4] job failed (state={res.get('state')}): {payload!r}")

    flat = _parse_flat_payload(res.get("payload") or res.get("result_payload"))
    Nf = N * N * N
    if flat.size != 3 * Nf:
        raise SystemExit(
            f"[task4] expected 3·N³ = {3 * Nf} elements back, got {flat.size}"
        )
    rhs_u = flat[0:Nf].reshape(N, N, N)
    rhs_v = flat[Nf:2 * Nf].reshape(N, N, N)
    rhs_w = flat[2 * Nf:3 * Nf].reshape(N, N, N)
    return rhs_u, rhs_v, rhs_w


# -----------------------------------------------------------------------------
# Unit test:
#   1. trace builds without IR errors
#   2. numpy reference produces sane numbers on the TGV initial condition
#   3. ∇·rhs sanity check (full divergence enforced by pressure projection
#      outside this kernel — non-zero here is expected)
#   4. (optional) submit to the gateway and compare against the reference
# -----------------------------------------------------------------------------
def _unit_test(submit: bool = True):
    from numpy_solve import _generate_initial_condition

    N = config.N
    h = config.DOMAIN / N
    nu = config.NU

    print(f"[task4] building trace  N={N}  h={h:.4g}  ν={nu}")
    mod = build_ns_rhs(N, h, nu)
    n_ops = sum(len(fn.ops) for fn in mod.functions)
    print(f"[task4] traced module   functions={len(mod.functions)}  ops={n_ops}")

    U = _generate_initial_condition(N)
    u, v, w = U[..., 0], U[..., 1], U[..., 2]
    ref_u, ref_v, ref_w = ns_rhs_numpy_ref(u, v, w, h, nu)
    print(
        f"[task4] numpy reference  "
        f"|rhs_u|_∞={np.max(np.abs(ref_u)):.4e}  "
        f"|rhs_v|_∞={np.max(np.abs(ref_v)):.4e}  "
        f"|rhs_w|_∞={np.max(np.abs(ref_w)):.4e}"
    )

    div_rhs = (
        (np.roll(ref_u, -1, 0) - np.roll(ref_u, +1, 0))
        + (np.roll(ref_v, -1, 1) - np.roll(ref_v, +1, 1))
        + (np.roll(ref_w, -1, 2) - np.roll(ref_w, +1, 2))
    ) / (2.0 * h)
    print(
        f"[task4] |∇·rhs|_∞ = {np.max(np.abs(div_rhs)):.4e}  "
        f"(non-zero by design — projection enforces incompressibility outside this kernel)"
    )

    if submit:
        got_u, got_v, got_w = submit_and_fetch(mod, u, v, w, N)
        err_u = float(np.max(np.abs(got_u - ref_u)))
        err_v = float(np.max(np.abs(got_v - ref_v)))
        err_w = float(np.max(np.abs(got_w - ref_w)))
        print(
            f"[task4] gateway vs numpy reference  "
            f"err_u={err_u:.3e}  err_v={err_v:.3e}  err_w={err_w:.3e}"
        )
        tol = 1e-9
        if max(err_u, err_v, err_w) > tol:
            raise SystemExit(
                f"[task4] FAIL — gateway disagrees with numpy ref beyond {tol:g}"
            )
        print("[task4] PASS — gateway output matches numpy reference to 1e-9.")

    return mod


if __name__ == "__main__":
    _unit_test()
