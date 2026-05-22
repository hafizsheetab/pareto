# =============================================================================
# uniqx_solve.py — Challenge 3 TGV solver, kernel-based counterpart of
# `numpy_solve.py`.
#
# Same physics, same diagnostics, same plots — but every spatial derivative
# comes from a native uniqx physics kernel instead of a numpy FFT pass.
#
# Structure
# ---------
# numpy_solve.py was built around the *spectral* loop:
#       Fourier transform → 2/3 dealiased advection → integrating-factor
#       RK4 in Fourier space → inverse FFT.
# uniqx exposes no FFT kernel pair, so this counterpart uses **finite
# differences with periodic boundary conditions** instead:
#
#   • d/dx, d/dy, d/dz, ∇²        ←  grid_gradient / grid_laplacian
#                                    (one IR op each, grid attributes only —
#                                     no dense matrix on the wire)
#   • RK4 time integration        ←  classical 4-stage, fully unrolled
#                                    inside the trace (one fused IR module
#                                    per timestep, ~6 matmul ops × 4 stages)
#   • pressure projection         ←  solve ∇²p = ∇·u*/dt with the
#                                    regularized Helmholtz operator
#                                    ∇² + k_reg²·I via linear_solve
#                                    (k_reg lifts the singular constant mode)
#   • E(t), ε(t), |u|-snapshot    ←  numpy on the host, identical formulas
#                                    to numpy_solve._plot_diagnostics
#
# One traced module is built once and submitted N_STEPS times — host stores
# (u, v, w) between calls, computes diagnostics, then re-submits. This is
# the structural mirror of numpy_solve.run()'s loop; the per-call payload
# is just three (N, N, N) f64 buffers.
# =============================================================================

import math

import matplotlib.pyplot as plt
import numpy as np

import config_3d as config
import uniqx as ux
from grid_3d import Grid3d
from uniqx import to_module
from uniqx.core import types as ut
from uniqx.ops.primitives.solvers import linear_solve
from uniqx_traced_ops import (
    build_grad_matrix,
    build_helmholtz_matrix,
    build_lap_matrix,
    flat,
    grad_components,
    grad_components_flat,
    lap_field,
)

DEFAULT_GATEWAY = "api.oriqx.com:443"
DEFAULT_API_KEY = "uxk_1bdb37b0f52f9d89260d86f2d21e9513"

# Wavenumber used to regularize the singular periodic Laplacian for the
# pressure-projection solve. k_reg²·I shifts every eigenvalue up by k_reg²,
# making H = ∇² + k_reg²·I invertible. ∇p is unaffected by the gauge
# constant absorbed into this shift, so the physical velocity correction
# matches the unregularized problem to O(k_reg² / λ_min(∇²)).
K_REG = 1.0e-3


# -----------------------------------------------------------------------------
# Initial condition — identical to numpy_solve._generate_initial_condition.
# -----------------------------------------------------------------------------
def _generate_initial_condition(n: int) -> np.ndarray:
    grid_indices = np.arange(n)
    scaled_indices = grid_indices * config.DOMAIN / (n * config.L)
    i = scaled_indices[:, np.newaxis, np.newaxis]
    j = scaled_indices[np.newaxis, :, np.newaxis]
    k = scaled_indices[np.newaxis, np.newaxis, :]

    scalar_field_u = np.sin(i) * np.cos(j) * np.cos(k)
    scalar_field_v = np.cos(i) * np.sin(j) * np.cos(k)

    result = np.zeros((n, n, n, 3), dtype=float)
    result[..., 0] = config.V_0 * scalar_field_u
    result[..., 1] = -config.V_0 * scalar_field_v
    return result


# -----------------------------------------------------------------------------
# Gateway data-transport helpers.
# -----------------------------------------------------------------------------
def _fmt_3d(arr: np.ndarray) -> str:
    """Encode (N, N, N) f64 array as a uniqx buffer-view string."""
    a, b, c = arr.shape
    return f"{a}x{b}x{c}xf64= " + " ".join(repr(x) for x in arr.reshape(-1).tolist())


def _parse_flat_payload(payload) -> np.ndarray:
    text = payload.decode("latin-1") if isinstance(payload, (bytes, bytearray)) else payload
    _, _, values = text.strip().partition("=")
    return np.fromstring(values, sep=" ", dtype=np.float64)


# -----------------------------------------------------------------------------
# Traced module: one RK4 step + pressure projection.
# -----------------------------------------------------------------------------
def build_tgv_step(N: int, h: float, nu: float, dt: float, k_reg: float = K_REG):
    """Build and return the fused single-step IR module.

    Module signature
    ----------------
    Inputs : three (N, N, N) f64 tensors  (u, v, w)
    Output : one  (3·N³,)   f64 tensor   packed as  [u_new ‖ v_new ‖ w_new]

    Algorithm (all inside one IR trace)
    -----------------------------------
    1. Build G (3·N³, N³), L (N³, N³), H (N³, N³) once via physics kernels.
    2. RK4 of  dU/dt = -u·∇u + ν·∇²u  — 4 stages, 36 matmuls total.
    3. Compute ∇·u* via three slices of G·{u*, v*, w*}, sum.
    4. Solve  H · p = ∇·u*/dt  with sparse hermitian linear_solve.
       (The constant mode absorbed by k_reg²·I doesn't enter ∇p.)
    5. Subtract  dt·∇p  from u*, v*, w*; pack and return.
    """
    Nf = N * N * N
    field_t = ut.tensor("f64", [N, N, N])
    flat_t = ut.tensor("f64", [Nf])
    out_t = ut.tensor("f64", [3 * Nf])

    @to_module(name="tgv_step")
    def step(u, v, w):
        # Operator matrices — one IR op each, reused across all RK4 stages
        # and the projection. The gateway lowers these to sparse stencil
        # sweeps; no dense N³ × N³ matrix ever materialises on the wire.
        G = build_grad_matrix(N, h)
        L = build_lap_matrix(N, h)
        H = build_helmholtz_matrix(N, h, k_reg)

        def rhs(uu, vv, ww):
            """NS-RHS:  -u·∇u  +  ν·∇²u  per component, all (N,N,N)."""
            dudx, dudy, dudz = grad_components(uu, G, N)
            dvdx, dvdy, dvdz = grad_components(vv, G, N)
            dwdx, dwdy, dwdz = grad_components(ww, G, N)

            adv_u = uu * dudx + vv * dudy + ww * dudz
            adv_v = uu * dvdx + vv * dvdy + ww * dvdz
            adv_w = uu * dwdx + vv * dwdy + ww * dwdz

            visc_u = lap_field(uu, L, N) * nu
            visc_v = lap_field(vv, L, N) * nu
            visc_w = lap_field(ww, L, N) * nu

            return (visc_u - adv_u, visc_v - adv_v, visc_w - adv_w)

        # --- RK4 stages -------------------------------------------------
        K1u, K1v, K1w = rhs(u, v, w)

        h2 = 0.5 * dt
        K2u, K2v, K2w = rhs(
            u + K1u * h2,
            v + K1v * h2,
            w + K1w * h2,
        )
        K3u, K3v, K3w = rhs(
            u + K2u * h2,
            v + K2v * h2,
            w + K2w * h2,
        )
        K4u, K4v, K4w = rhs(
            u + K3u * dt,
            v + K3v * dt,
            w + K3w * dt,
        )

        s = dt / 6.0
        u_star = u + (K1u + K2u * 2.0 + K3u * 2.0 + K4u) * s
        v_star = v + (K1v + K2v * 2.0 + K3v * 2.0 + K4v) * s
        w_star = w + (K1w + K2w * 2.0 + K3w * 2.0 + K4w) * s

        # --- Pressure projection ---------------------------------------
        # Flatten for the linear algebra leg.
        u_star_f = flat(u_star, N)
        v_star_f = flat(v_star, N)
        w_star_f = flat(w_star, N)

        # ∇·u* — only one component of each gradient is needed, so we use
        # grad_components_flat and discard the unused slices.
        dudx_f, _, _ = grad_components_flat(u_star_f, G, N)
        _, dvdy_f, _ = grad_components_flat(v_star_f, G, N)
        _, _, dwdz_f = grad_components_flat(w_star_f, G, N)
        div_f = dudx_f + dvdy_f + dwdz_f

        b_f = div_f * (1.0 / dt)

        # Solve  H · p = b  with H = ∇² + k_reg²·I  (regularized periodic
        # Poisson). hermitian=True lets the gateway pick a symmetric solver;
        # sparse=True keeps the kernel-emitted operator in its sparse form.
        p_f = linear_solve(H, b_f, sparse=True, hermitian=True)

        dpdx_f, dpdy_f, dpdz_f = grad_components_flat(p_f, G, N)

        u_new_f = u_star_f - dpdx_f * dt
        v_new_f = v_star_f - dpdy_f * dt
        w_new_f = w_star_f - dpdz_f * dt

        # Single-tensor return — the gateway response carries only the
        # first output. Host code splits this back into (u, v, w).
        _ = flat_t  # silence linter: flat_t is part of the documented shape contract
        return ux.concatenate(
            u_new_f, v_new_f, w_new_f,
            axis=0,
            result_type=out_t,
        )

    return step(field_t, field_t, field_t)


# -----------------------------------------------------------------------------
# One step on the gateway.
# -----------------------------------------------------------------------------
def submit_step(mod, u: np.ndarray, v: np.ndarray, w: np.ndarray, N: int, client):
    runtime = [_fmt_3d(u), _fmt_3d(v), _fmt_3d(w), "backend=compiled"]
    job_id = ux.submit(mod, client=client, runtime_inputs=runtime)
    res = ux.get(job_id, client=client, timeout=600.0)
    if res.get("state") != 10:
        payload = res.get("payload") or res.get("result_payload") or b""
        raise SystemExit(f"[uniqx-tgv] job failed (state={res.get('state')}): {payload!r}")

    flat_out = _parse_flat_payload(res.get("payload") or res.get("result_payload"))
    Nf = N * N * N
    if flat_out.size != 3 * Nf:
        raise SystemExit(
            f"[uniqx-tgv] expected 3·N³={3 * Nf} elements back, got {flat_out.size}"
        )
    return (
        flat_out[0:Nf].reshape(N, N, N),
        flat_out[Nf:2 * Nf].reshape(N, N, N),
        flat_out[2 * Nf:3 * Nf].reshape(N, N, N),
    )


# -----------------------------------------------------------------------------
# Diagnostics (host-side, physical space).
# -----------------------------------------------------------------------------
def compute_energy_dissipation_phys(u, v, w, dx, nu, V):
    """Physical-space E(t) and ε(t), using the same central-difference
    operator family that the trace uses for ∇ on the gateway.

        E   = (1/V) · ½ Σ |u|² · dx³
        eps = (1/V) ·  ν Σ |∇u|² · dx³

    With V = (2πL)³ this matches the standard TGV non-dimensionalization,
    so the curve can be overlaid on published Re=1600 DNS data.
    """
    cell_vol = dx ** 3
    abs_sq = u ** 2 + v ** 2 + w ** 2
    E_k = 0.5 * np.sum(abs_sq) * cell_vol / V

    inv_2dx = 1.0 / (2.0 * dx)
    grad_sq = np.zeros_like(u)
    for f in (u, v, w):
        for axis in range(3):
            df = (np.roll(f, -1, axis=axis) - np.roll(f, +1, axis=axis)) * inv_2dx
            grad_sq = grad_sq + df * df
    eps = nu * np.sum(grad_sq) * cell_vol / V
    return E_k, eps


def _plot_diagnostics(times, energy, snaps, grid: Grid3d):
    """Same three plots numpy_solve emits, suffixed `_uniqx` so both paths
    can be eyeballed side-by-side in the assets folder."""
    times = np.asarray(times)
    E_k = np.array([e[0] for e in energy])
    eps = np.array([e[1] for e in energy])

    # --- Energy + dissipation ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, E_k, "b-")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("kinetic energy  E(t)")
    axes[0].set_title("Energy  [uniqx-kernel TGV]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, eps, "r-")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"dissipation  $\varepsilon(t)$")
    axes[1].set_title("Dissipation rate  [uniqx-kernel TGV]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{config.ASSETS_DIR}/energy_uniqx.png", dpi=120)
    plt.close(fig)

    # --- Energy balance: dE/dt vs -eps --------------------------------
    dEdt = np.gradient(E_k, times)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, dEdt, "b-",  label=r"$dE/dt$ (finite diff)")
    axes[0].plot(times, -eps,  "r--", label=r"$-\varepsilon(t)$")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("rate")
    axes[0].set_title("Energy balance  [uniqx-kernel]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    denom = np.maximum(np.abs(eps), 1e-300)
    rel_err = np.abs(dEdt + eps) / denom
    axes[1].semilogy(times, rel_err, "k-")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$|dE/dt + \varepsilon| / |\varepsilon|$")
    axes[1].set_title("Relative imbalance (log scale)")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{config.ASSETS_DIR}/energy_balance_uniqx.png", dpi=120)
    plt.close(fig)

    mask = E_k > E_k[0] * 1e-10
    if mask.any():
        worst = rel_err[mask].max()
        print(f"[energy balance] max |dE/dt + eps|/|eps| over meaningful "
              f"window: {worst:.3e}  ({mask.sum()} points)")

    # --- Velocity magnitude snapshots --------------------------------
    n_snap = min(6, len(snaps))
    snap_idx = np.linspace(0, len(snaps) - 1, n_snap).astype(int)
    z_mid = grid.N // 2

    mags = []
    for idx in snap_idx:
        u, v, w = snaps[idx]
        mags.append(np.sqrt(u ** 2 + v ** 2 + w ** 2)[:, :, z_mid])
    vmax_global = max(m.max() for m in mags)

    fig, axes = plt.subplots(1, n_snap, figsize=(3 * n_snap, 3.2))
    if n_snap == 1:
        axes = [axes]
    for ax, idx, mag in zip(axes, snap_idx, mags):
        im = ax.imshow(mag, origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax_global)
        ax.set_title(f"t = {times[idx]:.3g}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(r"|u| on mid-z slice — uniqx-kernel TGV")
    fig.savefig(f"{config.ASSETS_DIR}/velocity_snapshots_uniqx.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Top-level driver — structural mirror of numpy_solve.run().
# -----------------------------------------------------------------------------
def run(grid: Grid3d,
        t_final: float | None = None,
        cfl: float = 0.5,
        gateway: str = DEFAULT_GATEWAY,
        api_key: str = DEFAULT_API_KEY,
        n_steps_override: int | None = None,
        save_every: int = 1):
    """Run the TGV simulation on the uniqx gateway and emit the same plots
    `numpy_solve.run` does (suffixed `_uniqx`)."""
    # Time step: the trace is purely explicit, so we need both the convective
    # CFL bound *and* the Von Neumann viscous bound. The Grid3d default is
    # the viscous bound; we tighten to whichever is more restrictive.
    dt_cfl = cfl * grid.dx / config.V_0
    grid.dt = min(grid.dt, dt_cfl)

    if t_final is None:
        # Same target window as numpy_solve: ~ 2× past peak dissipation.
        t_final = 20.0 * config.L / config.V_0
    n_steps = int(np.ceil(t_final / grid.dt))
    if n_steps_override is not None:
        n_steps = n_steps_override

    # Standard TGV non-dimensionalization volume — domain is [0, 2πL]³.
    V_dom = (2.0 * math.pi * config.L) ** 3

    print(f"[uniqx-tgv] N={grid.N}  dx={grid.dx:.4g}  dt={grid.dt:.4g}  "
          f"n_steps={n_steps}  t_final={n_steps * grid.dt:.3g}")

    # Trace the per-step module once.
    print("[uniqx-tgv] tracing step module …", flush=True)
    mod = build_tgv_step(grid.N, grid.dx, grid.nu, grid.dt)
    n_ops = sum(len(fn.ops) for fn in mod.functions)
    print(
        f"[uniqx-tgv] traced module: functions={len(mod.functions)}  ops={n_ops}",
        flush=True,
    )

    print(f"[uniqx-tgv] connecting to {gateway} …", flush=True)
    client = ux.connect(gateway, api_key=api_key)

    # Initial condition.
    U = _generate_initial_condition(grid.N)
    u = U[..., 0].copy()
    v = U[..., 1].copy()
    w = U[..., 2].copy()

    times: list[float] = []
    energy: list[tuple[float, float]] = []
    snaps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    t = 0.0

    for i in range(n_steps):
        if i % save_every == 0:
            E_k, eps = compute_energy_dissipation_phys(u, v, w, grid.dx, grid.nu, V_dom)
            snaps.append((u.copy(), v.copy(), w.copy()))
            times.append(t)
            energy.append((E_k, eps))
            if i % max(1, n_steps // 20) == 0:
                print(
                    f"[uniqx-tgv] step {i:4d}/{n_steps}  t={t:.4f}  "
                    f"E={E_k:.4e}  eps={eps:.4e}",
                    flush=True,
                )

        u, v, w = submit_step(mod, u, v, w, grid.N, client)
        t += grid.dt

    # Final snapshot so the last plotted point is the actual end state.
    E_k, eps = compute_energy_dissipation_phys(u, v, w, grid.dx, grid.nu, V_dom)
    snaps.append((u.copy(), v.copy(), w.copy()))
    times.append(t)
    energy.append((E_k, eps))

    _plot_diagnostics(times, energy, snaps, grid)
    return times, energy, snaps


if __name__ == "__main__":
    run(Grid3d())
