# =============================================================================
# qp_run_task4.py — Challenge 4 driver: fused NS-RHS module + full TGV run.
#
# Closes out the Task-4 deliverable by wrapping `qp_solve_task4.build_ns_rhs`
# (single `@to_module` trace returning (rhs_u, rhs_v, rhs_w)) with the loop
# infrastructure the README asks for:
#
#   • RK4 (4 stages) driven on the host: each stage either submits the fused
#     trace to the gateway *or* evaluates the validated NumPy reference. Both
#     paths reuse the same algorithm — the reference exists so the curve can
#     be generated even when the gateway is unreachable.
#   • Pressure projection via FFT-based Helmholtz–Hodge decomposition (host).
#     Projection lives outside the fused kernel by design (qp_solve_task4
#     documents this); doing it on the host keeps the trace a single dispatch.
#   • ε(t) computed in physical space (central-difference FD) so the curve is
#     directly comparable to the spectral Challenge-3 baseline once both are
#     evaluated through the same diagnostic.
#   • Overlay plot `assets/energy_uniqx_fused.png` — Task 4 fused-kernel ε(t)
#     vs Task 3 spectral ε(t) computed on the same volume/dx normalization.
#
# Run:
#   python qp_run_task4.py                       # numpy-only (no gateway)
#   python qp_run_task4.py --gateway             # 4 submissions per RK4 step
#   python qp_run_task4.py --n-steps 200         # cap step count for quick demo
# =============================================================================

import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np

import config_3d as config
import numpy_solve as ns
from grid_3d import Grid3d
from qp_solve_task4 import (
    DEFAULT_API_KEY,
    DEFAULT_GATEWAY,
    _fmt_3d,
    _parse_flat_payload,
    build_ns_rhs,
    ns_rhs_numpy_ref,
)
from uniqx_solve import compute_energy_dissipation_phys


# -----------------------------------------------------------------------------
# Host-side periodic Helmholtz–Hodge projection.
#
# Used at the end of each RK4 step:  u^{n+1} = u* − ∇φ  where ∇²φ = ∇·u*.
# Closed-form in Fourier space:  φ̂ = i(k·û*)/|k|²,  ûproj = û* − ik φ̂.
# Constant mode forced to zero (gauge).
# -----------------------------------------------------------------------------
def project_periodic_fft(u, v, w, h):
    N = u.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=h)
    kx = k.reshape(N, 1, 1)
    ky = k.reshape(1, N, 1)
    kz = k.reshape(1, 1, N)
    K2 = kx ** 2 + ky ** 2 + kz ** 2
    K2[0, 0, 0] = 1.0

    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)

    kdotu = kx * u_hat + ky * v_hat + kz * w_hat
    factor = kdotu / K2

    u_hat -= factor * kx
    v_hat -= factor * ky
    w_hat -= factor * kz
    u_hat[0, 0, 0] = 0.0
    v_hat[0, 0, 0] = 0.0
    w_hat[0, 0, 0] = 0.0

    return (
        np.real(np.fft.ifftn(u_hat)),
        np.real(np.fft.ifftn(v_hat)),
        np.real(np.fft.ifftn(w_hat)),
    )


# -----------------------------------------------------------------------------
# RHS evaluation — switchable between gateway and NumPy reference.
# -----------------------------------------------------------------------------
def _gateway_rhs(mod, u, v, w, N, client):
    """One submission of the fused NS-RHS module; returns three (N,N,N) arrays."""
    import uniqx as ux

    runtime = [_fmt_3d(u), _fmt_3d(v), _fmt_3d(w), "backend=compiled"]
    job_id = ux.submit(mod, client=client, runtime_inputs=runtime)
    res = ux.get(job_id, client=client, timeout=600.0)
    if res.get("state") != 10:
        payload = res.get("payload") or res.get("result_payload") or b""
        raise SystemExit(f"[qp-task4] job failed (state={res.get('state')}): {payload!r}")

    flat = _parse_flat_payload(res.get("payload") or res.get("result_payload"))
    Nf = N * N * N
    return (
        flat[0:Nf].reshape(N, N, N),
        flat[Nf:2 * Nf].reshape(N, N, N),
        flat[2 * Nf:3 * Nf].reshape(N, N, N),
    )


def rk4_step(u, v, w, dt, h, nu, *, rhs_fn):
    """Classical RK4 step using a swappable RHS function.

    `rhs_fn(u, v, w) -> (Ku, Kv, Kw)` is the only thing that varies between
    the gateway-driven path (4 submissions of the fused trace per step) and
    the NumPy-only path (4 calls to ns_rhs_numpy_ref). Same algorithm, same
    arithmetic — the host arithmetic is identical so any divergence in the
    final state isolates the fused-trace numerics.
    """
    K1u, K1v, K1w = rhs_fn(u, v, w)

    h2 = 0.5 * dt
    K2u, K2v, K2w = rhs_fn(u + h2 * K1u, v + h2 * K1v, w + h2 * K1w)
    K3u, K3v, K3w = rhs_fn(u + h2 * K2u, v + h2 * K2v, w + h2 * K2w)
    K4u, K4v, K4w = rhs_fn(u + dt * K3u, v + dt * K3v, w + dt * K3w)

    s = dt / 6.0
    u_star = u + s * (K1u + 2 * K2u + 2 * K3u + K4u)
    v_star = v + s * (K1v + 2 * K2v + 2 * K3v + K4v)
    w_star = w + s * (K1w + 2 * K2w + 2 * K3w + K4w)
    return u_star, v_star, w_star


# -----------------------------------------------------------------------------
# Task-4 simulation — host RK4 around the fused NS-RHS, FFT projection.
# -----------------------------------------------------------------------------
def run_task4(grid: Grid3d,
              n_steps_max: int | None = None,
              use_gateway: bool = False):
    dt_cfl = 0.5 * grid.dx / config.V_0
    grid.dt = min(grid.dt, dt_cfl)
    t_final = 20.0 * config.L / config.V_0
    n_steps = int(np.ceil(t_final / grid.dt))
    if n_steps_max is not None:
        n_steps = min(n_steps, n_steps_max)

    V_dom = (2.0 * math.pi * config.L) ** 3
    backend = "gateway" if use_gateway else "numpy-ref"
    print(f"[task4] N={grid.N}  dx={grid.dx:.4g}  dt={grid.dt:.4g}  "
          f"n_steps={n_steps}  backend={backend}")

    # Pick the RHS implementation. Both paths run the same algorithm; the
    # gateway path costs 4 submissions per RK4 step but exercises the fused
    # trace; the numpy path is the validated reference (the unit test in
    # qp_solve_task4._unit_test asserts they agree to 1e-9 at the single-step
    # level).
    mod = None
    client = None
    if use_gateway:
        import uniqx as ux

        print("[task4] tracing fused ns_rhs module …", flush=True)
        mod = build_ns_rhs(grid.N, grid.dx, grid.nu)
        n_ops = sum(len(fn.ops) for fn in mod.functions)
        print(f"[task4] traced module: functions={len(mod.functions)}  ops={n_ops}",
              flush=True)
        print(f"[task4] connecting to {DEFAULT_GATEWAY} …", flush=True)
        client = ux.connect(DEFAULT_GATEWAY, api_key=DEFAULT_API_KEY)

        def rhs_fn(u_, v_, w_):
            return _gateway_rhs(mod, u_, v_, w_, grid.N, client)
    else:
        def rhs_fn(u_, v_, w_):
            return ns_rhs_numpy_ref(u_, v_, w_, grid.dx, grid.nu)

    # TGV initial condition (identical to numpy_solve.py).
    U = ns._generate_initial_condition(grid.N)
    u = U[..., 0].copy()
    v = U[..., 1].copy()
    w = U[..., 2].copy()
    u, v, w = project_periodic_fft(u, v, w, grid.dx)

    times: list[float] = []
    eps_list: list[float] = []
    E_list: list[float] = []
    t = 0.0
    t0 = time.time()
    report_every = max(1, n_steps // 20)

    for i in range(n_steps):
        E_k, eps = compute_energy_dissipation_phys(u, v, w, grid.dx, grid.nu, V_dom)
        times.append(t)
        E_list.append(E_k)
        eps_list.append(eps)
        if i % report_every == 0:
            print(f"[task4] step {i:4d}/{n_steps}  t={t:.4f}  "
                  f"E={E_k:.4e}  eps={eps:.4e}", flush=True)

        u_star, v_star, w_star = rk4_step(u, v, w, grid.dt, grid.dx, grid.nu, rhs_fn=rhs_fn)
        u, v, w = project_periodic_fft(u_star, v_star, w_star, grid.dx)
        t += grid.dt

    # Final snapshot — last plotted point is the actual end state.
    E_k, eps = compute_energy_dissipation_phys(u, v, w, grid.dx, grid.nu, V_dom)
    times.append(t)
    E_list.append(E_k)
    eps_list.append(eps)

    runtime = time.time() - t0
    n_rhs_calls = 4 * n_steps  # RK4 has 4 stages
    print(f"[task4] done — wall {runtime:.2f}s  "
          f"({n_rhs_calls} RHS evaluations, {runtime / n_rhs_calls * 1e3:.1f} ms/RHS)")

    return {
        "times": np.asarray(times),
        "energy": np.asarray(E_list),
        "eps": np.asarray(eps_list),
        "runtime": runtime,
        "n_rhs_calls": n_rhs_calls,
        "backend": backend,
    }


# -----------------------------------------------------------------------------
# Challenge-3 reference (spectral) — ε(t) computed *with the same physical-
# space diagnostic* as the Task-4 run, so the two curves share normalization.
# -----------------------------------------------------------------------------
def run_task3_spectral(grid: Grid3d, n_steps_max: int | None = None):
    dt_cfl = 0.5 * grid.dx / config.V_0
    g = Grid3d(N=grid.N)
    g.dt = min(g.dt, dt_cfl)
    t_final = 20.0 * config.L / config.V_0
    n_steps = int(np.ceil(t_final / g.dt))
    if n_steps_max is not None:
        n_steps = min(n_steps, n_steps_max)

    V_dom = (2.0 * math.pi * config.L) ** 3
    print(f"[task3] N={g.N}  dx={g.dx:.4g}  dt={g.dt:.4g}  n_steps={n_steps}  backend=spectral")

    U = ns._generate_initial_condition(g.N)
    (kx, ky, kz) = ns._get_kx(g.N, grid=g)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    u_hat, v_hat, w_hat = ns._fourier_transform(U)

    times: list[float] = []
    eps_list: list[float] = []
    E_list: list[float] = []
    t = 0.0
    t0 = time.time()
    N = g.N
    report_every = max(1, n_steps // 20)

    for i in range(n_steps):
        # Inverse-FFT to physical space and apply the SAME ε formula used
        # by the Task-4 run — this is what makes the curves comparable.
        u_p = np.fft.irfftn(u_hat, s=(N, N, N), axes=(0, 1, 2))
        v_p = np.fft.irfftn(v_hat, s=(N, N, N), axes=(0, 1, 2))
        w_p = np.fft.irfftn(w_hat, s=(N, N, N), axes=(0, 1, 2))
        E_k, eps = compute_energy_dissipation_phys(u_p, v_p, w_p, g.dx, g.nu, V_dom)
        times.append(t)
        E_list.append(E_k)
        eps_list.append(eps)
        if i % report_every == 0:
            print(f"[task3] step {i:4d}/{n_steps}  t={t:.4f}  "
                  f"E={E_k:.4e}  eps={eps:.4e}", flush=True)

        u_hat, v_hat, w_hat, t = ns.rk4_step(k2, t, u_hat, v_hat, w_hat, grid=g, N=N)

    u_p = np.fft.irfftn(u_hat, s=(N, N, N), axes=(0, 1, 2))
    v_p = np.fft.irfftn(v_hat, s=(N, N, N), axes=(0, 1, 2))
    w_p = np.fft.irfftn(w_hat, s=(N, N, N), axes=(0, 1, 2))
    E_k, eps = compute_energy_dissipation_phys(u_p, v_p, w_p, g.dx, g.nu, V_dom)
    times.append(t)
    E_list.append(E_k)
    eps_list.append(eps)

    runtime = time.time() - t0
    print(f"[task3] done — wall {runtime:.2f}s")
    return {
        "times": np.asarray(times),
        "energy": np.asarray(E_list),
        "eps": np.asarray(eps_list),
        "runtime": runtime,
        "backend": "spectral",
    }


# -----------------------------------------------------------------------------
# Comparison plot.
# -----------------------------------------------------------------------------
def plot_comparison(grid: Grid3d, t4: dict, t3: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(t4["times"], t4["eps"], "b-", lw=1.6,
                 label=f"Task 4 — fused NS-RHS ({t4['backend']})")
    axes[0].plot(t3["times"], t3["eps"], "r--", lw=1.2,
                 label="Task 3 — spectral (numpy_solve)")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"dissipation rate  $\varepsilon(t)$")
    axes[0].set_title(f"TGV $\\varepsilon(t)$  Re={config.RE}  N={grid.N}$^3$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    eps_peak_t4 = float(np.max(t4["eps"]))
    eps_peak_t3 = float(np.max(t3["eps"]))
    axes[1].plot(t4["times"], t4["eps"] / eps_peak_t4, "b-", lw=1.6,
                 label="Task 4 (fused) / peak")
    axes[1].plot(t3["times"], t3["eps"] / eps_peak_t3, "r--", lw=1.2,
                 label="Task 3 (spectral) / peak")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$\varepsilon(t) / \varepsilon_{\rm peak}$")
    axes[1].set_title("Shape comparison (peak-normalized)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {save_path}")


# -----------------------------------------------------------------------------
# Single-step gateway equivalence check — closes the loop on the unit test.
# -----------------------------------------------------------------------------
def validate_gateway_against_numpy(grid: Grid3d, atol: float = 1e-9) -> bool:
    """Submit one fused NS-RHS evaluation to the gateway on the TGV IC and
    confirm it matches `ns_rhs_numpy_ref` element-wise."""
    import uniqx as ux

    print("[validate] tracing fused ns_rhs module …", flush=True)
    mod = build_ns_rhs(grid.N, grid.dx, grid.nu)
    n_ops = sum(len(fn.ops) for fn in mod.functions)
    print(f"[validate] module ops = {n_ops}", flush=True)

    U = ns._generate_initial_condition(grid.N)
    u, v, w = U[..., 0], U[..., 1], U[..., 2]
    ref_u, ref_v, ref_w = ns_rhs_numpy_ref(u, v, w, grid.dx, grid.nu)

    print(f"[validate] connecting to {DEFAULT_GATEWAY} …", flush=True)
    client = ux.connect(DEFAULT_GATEWAY, api_key=DEFAULT_API_KEY)
    got_u, got_v, got_w = _gateway_rhs(mod, u, v, w, grid.N, client)

    err = max(
        float(np.max(np.abs(got_u - ref_u))),
        float(np.max(np.abs(got_v - ref_v))),
        float(np.max(np.abs(got_w - ref_w))),
    )
    print(f"[validate] gateway vs numpy reference  max err = {err:.3e}")
    if err > atol:
        raise SystemExit(f"[validate] FAIL — gateway disagrees with numpy ref beyond {atol:g}")
    print(f"[validate] PASS — gateway output matches numpy reference to {atol:g}.")
    return True


# -----------------------------------------------------------------------------
# Entry point.
# -----------------------------------------------------------------------------
def main(use_gateway: bool, n_steps_max: int | None) -> None:
    grid = Grid3d()
    print("=" * 64)
    print("  ORIQX CFD — Challenge 4: fused NS-RHS driving a TGV run")
    print("=" * 64)
    print(grid, "\n")

    if use_gateway:
        validate_gateway_against_numpy(grid)
        print()

    t4 = run_task4(grid, n_steps_max=n_steps_max, use_gateway=use_gateway)
    print()
    t3 = run_task3_spectral(grid, n_steps_max=n_steps_max)
    print()

    save_path = f"{config.ASSETS_DIR}/energy_uniqx_fused.png"
    plot_comparison(grid, t4, t3, save_path)

    eps_peak_t4 = float(np.max(t4["eps"]))
    eps_peak_t3 = float(np.max(t3["eps"]))
    t_peak_t4 = float(t4["times"][int(np.argmax(t4["eps"]))])
    t_peak_t3 = float(t3["times"][int(np.argmax(t3["eps"]))])
    rel = abs(eps_peak_t4 - eps_peak_t3) / eps_peak_t3
    print("\n" + "=" * 64)
    print("  Summary")
    print("=" * 64)
    print(f"  Task 4 (fused NS-RHS, {t4['backend']:>10}):  "
          f"peak ε = {eps_peak_t4:.4e} @ t = {t_peak_t4:.3f}  "
          f"wall = {t4['runtime']:.2f}s  "
          f"({t4['n_rhs_calls']} RHS calls)")
    print(f"  Task 3 (spectral, numpy_solve     ):  "
          f"peak ε = {eps_peak_t3:.4e} @ t = {t_peak_t3:.3f}  "
          f"wall = {t3['runtime']:.2f}s")
    print(f"  Peak-ε relative gap: {rel:.2%}")
    print(f"  Plot:  {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4 — fused NS-RHS TGV driver")
    parser.add_argument("--gateway", action="store_true",
                        help="submit RHS to the uniqx gateway (4 calls per RK4 step)")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="cap on n_steps (default: full TGV run to t* ≈ 20)")
    args = parser.parse_args()
    main(use_gateway=args.gateway, n_steps_max=args.n_steps)
