import numpy as np
import matplotlib.pyplot as plt
import config_3d as config
import math
from grid_3d import Grid3d



def _generate_initial_condition(n):
    import numpy as np

    grid_indices = np.arange(n)
    scaled_indices = grid_indices * config.DOMAIN / (n * config.L)

    i = scaled_indices[:, np.newaxis, np.newaxis]
    j = scaled_indices[np.newaxis, :, np.newaxis]
    k = scaled_indices[np.newaxis, np.newaxis, :]
    # print(i)
    scalar_field_u = np.sin(i) * np.cos(j) * np.cos(k)
    scalar_field_v = np.cos(i) * np.sin(j) * np.cos(k)

    result = np.zeros((n, n, n, 3), dtype=float)
    result[..., 0] = config.V_0 * scalar_field_u
    result[..., 1] = -config.V_0 * scalar_field_v
    return result

def _fourier_transform(U):
    u = U[..., 0]
    v = U[..., 1]
    w = U[..., 2]
    u_hat = np.fft.rfftn(u)
    v_hat = np.fft.rfftn(v)
    w_hat = np.fft.rfftn(w)
    return (u_hat, v_hat, w_hat)
    # return(np.stack(, axis=-1))
    # return result_hat
def _pad_hat(x_hat):
    N = x_hat.shape[0]
    M = int(1.5 * N)
    half_N = N // 2

    x_hat_padded = np.zeros((M, M, M // 2 + 1), dtype=x_hat.dtype)

    x_hat_padded[:half_N, :half_N, :half_N + 1] = x_hat[:half_N, :half_N, :]
    x_hat_padded[-half_N:, :half_N, :half_N + 1] = x_hat[-half_N:, :half_N, :]
    x_hat_padded[:half_N, -half_N:, :half_N + 1] = x_hat[:half_N, -half_N:, :]
    x_hat_padded[-half_N:, -half_N:, :half_N + 1] = x_hat[-half_N:, -half_N:, :]
    return x_hat_padded

def _get_kx(n, grid: Grid3d):
    kx = np.fft.fftfreq(n, d=grid.dx) * 2 * math.pi
    ky = np.fft.fftfreq(n, d=grid.dy) * 2 * math.pi
    kz = np.fft.rfftfreq(n, d=grid.dz) * 2 * math.pi
    kx = kx.reshape(n, 1, 1)
    ky = ky.reshape(1, n, 1)
    kz = kz.reshape(1, 1, -1)
    return (kx, ky, kz)

def _truncate_rfftn_shift(f_hat_M, N, M):
    """
    f_hat_M : complex array of shape (M, M, M//2+1)
    N : original grid size (even)
    M : fine grid size (M = 3*N//2, also even)
    Returns complex array of shape (N, N, N//2+1)
    """
    # 1. Shift axes 0 and 1 to bring zero frequency to the centre
    f_shift = np.fft.ifftshift(f_hat_M, axes=(0,1))
    # Now for axes 0 and 1: index M//2 corresponds to k=0
    # Axis 2 is unchanged: index 0 is k_z=0, positive up to M//2.

    # 2. Determine the start index and slice out the central N
    start = (M - N) // 2
    f_trunc_shift = f_shift[start:start+N, start:start+N, :N//2+1]
    # shape now (N, N, N//2+1)

    # 3. Shift back to standard FFT ordering
    f_trunc = np.fft.fftshift(f_trunc_shift, axes=(0,1))
    return f_trunc
def _projector(ft_stacked, N, grid: Grid3d, k2):
    (u_hat_n, v_hat_n, w_hat_n) = ft_stacked
    
    (u_hat_m, v_hat_m, w_hat_m) = (_pad_hat(u_hat_n), _pad_hat(v_hat_n), _pad_hat(w_hat_n))
    (kx_m, ky_m, kz_m) = _get_kx(int(1.5*N), grid=grid)
    # x‑derivatives
    dudx = np.fft.irfftn(1j * kx_m * u_hat_m)
    dvdx = np.fft.irfftn(1j * kx_m * v_hat_m)
    dwdx = np.fft.irfftn(1j * kx_m * w_hat_m)
    
    # y‑derivatives
    dudy = np.fft.irfftn(1j * ky_m * u_hat_m)
    dvdy = np.fft.irfftn(1j * ky_m * v_hat_m)
    dwdy = np.fft.irfftn(1j * ky_m * w_hat_m)
    
    # z‑derivatives
    dudz = np.fft.irfftn(1j * kz_m * u_hat_m)
    dvdz = np.fft.irfftn(1j * kz_m * v_hat_m)
    dwdz = np.fft.irfftn(1j * kz_m * w_hat_m)
    
    u = np.fft.irfftn(u_hat_m)
    v = np.fft.irfftn(v_hat_m)
    w = np.fft.irfftn(w_hat_m)
    
    Nx = u * dudx + v * dudy + w * dudz
    Ny = u * dvdx + v * dvdy + w * dvdz
    Nz = u * dwdx + v * dwdy + w * dwdz
    
    M = 3*N//2
    Nx_hat = _truncate_rfftn_shift(np.fft.rfftn(Nx), N, M)
    Ny_hat = _truncate_rfftn_shift(np.fft.rfftn(Ny), N, M)
    Nz_hat = _truncate_rfftn_shift(np.fft.rfftn(Nz), N, M)
    # return (-Nx_hat, -Ny_hat, -Nz_hat)
    (kx, ky, kz) = _get_kx(N, grid=grid)
    # Don't mutate the caller's k2 — use a safe local copy with 1 at the DC bin.
    k2_safe = k2.copy()
    k2_safe[0, 0, 0] = 1.0
    (px_hat, py_hat, pz_hat) = project(-Nx_hat, -Ny_hat, -Nz_hat, kx, ky, kz, k2_safe)
    return (px_hat, py_hat, pz_hat)

def project(fx_hat, fy_hat, fz_hat, kx, ky, kz, k2):
    """
    Apply the divergence-free projection to a vector field in Fourier space.
    All inputs are complex arrays of shape (N, N, N//2+1).
    Returns three complex arrays of the same shape.
    """
    # Dot product k·f
    kdotf = kx * fx_hat + ky * fy_hat + kz * fz_hat
    
    # Compute (k·f)/|k|^2
    factor = kdotf / k2
    
    # Subtract the parallel component
    px_hat = fx_hat - factor * kx
    py_hat = fy_hat - factor * ky
    pz_hat = fz_hat - factor * kz
    
    # Zero out the mean mode (optional, but recommended)
    px_hat[0,0,0] = 0.0
    py_hat[0,0,0] = 0.0
    pz_hat[0,0,0] = 0.0
    
    return px_hat, py_hat, pz_hat
    
def rhs_U(k2, s, Vx, Vy, Vz, N, grid: Grid3d):
    # s is the local time within the current RK4 step, s in [0, dt].
    # Using s (not absolute t) keeps exp(+/- NU*k2*s) bounded.
    exp_neg = np.exp(-config.NU * k2 * s)
    u_hat = exp_neg * Vx
    v_hat = exp_neg * Vy
    w_hat = exp_neg * Vz
    (Px, Py, Pz) = _projector((u_hat, v_hat, w_hat), N=N, grid=grid, k2=k2)
    exp_pos = np.exp(config.NU * k2 * s)
    return exp_pos * Px, exp_pos * Py, exp_pos * Pz

def rk4_step(k2, t_stage, u_hat, v_hat, w_hat, grid: Grid3d, N):
    """RK4 step in Fourier space using a step-local integrating factor.

    Takes the physical Fourier coefficients u_hat at time t_stage and returns
    them at t_stage + dt. Internally works with V(s) = exp(NU*k2*s) * u_hat(t+s)
    where s in [0, dt], so the integrating factor never accumulates across steps.
    """
    dt = grid.dt
    # V(0) = u_hat (physical Fourier mode at start of step)
    Vx, Vy, Vz = u_hat, v_hat, w_hat

    # Stage 1: s = 0
    K1x, K1y, K1z = rhs_U(k2, 0.0, Vx, Vy, Vz, N, grid=grid)

    # Stage 2: s = dt/2
    s_half = 0.5 * dt
    Vx2 = Vx + s_half * K1x
    Vy2 = Vy + s_half * K1y
    Vz2 = Vz + s_half * K1z
    K2x, K2y, K2z = rhs_U(k2, s_half, Vx2, Vy2, Vz2, N, grid=grid)

    # Stage 3: s = dt/2
    Vx3 = Vx + s_half * K2x
    Vy3 = Vy + s_half * K2y
    Vz3 = Vz + s_half * K2z
    K3x, K3y, K3z = rhs_U(k2, s_half, Vx3, Vy3, Vz3, N, grid=grid)

    # Stage 4: s = dt
    Vx4 = Vx + dt * K3x
    Vy4 = Vy + dt * K3y
    Vz4 = Vz + dt * K3z
    K4x, K4y, K4z = rhs_U(k2, dt, Vx4, Vy4, Vz4, N, grid=grid)

    # V(dt)
    Vx_new = Vx + (dt / 6.0) * (K1x + 2*K2x + 2*K3x + K4x)
    Vy_new = Vy + (dt / 6.0) * (K1y + 2*K2y + 2*K3y + K4y)
    Vz_new = Vz + (dt / 6.0) * (K1z + 2*K2z + 2*K3z + K4z)

    # Convert back: u_hat(t+dt) = exp(-NU*k2*dt) * V(dt)
    exp_neg_dt = np.exp(-config.NU * k2 * dt)
    u_hat_new = exp_neg_dt * Vx_new
    v_hat_new = exp_neg_dt * Vy_new
    w_hat_new = exp_neg_dt * Vz_new

    return (u_hat_new, v_hat_new, w_hat_new, t_stage + dt)
def run(grid: Grid3d, t_final: float | None = None, cfl: float = 0.5):
    # The integrating factor handles diffusion exactly, so the timestep is
    # bounded by the convective CFL condition, not viscous stability.
    dt_cfl = cfl * grid.dx / config.V_0
    grid.dt = min(grid.dt, dt_cfl)

    # TGV Re=1600 dissipation peaks at t* = t·V_0/L ≈ 9.  Run to t* ≈ 20.
    if t_final is None:
        t_final = 20.0 * config.L / config.V_0
    n_steps = int(np.ceil(t_final / grid.dt))

    print(f"[run] N={grid.N}  dx={grid.dx:.4g}  dt={grid.dt:.4g}  "
          f"n_steps={n_steps}  t_final={n_steps*grid.dt:.3g}")

    u = _generate_initial_condition(grid.N)
    (kx, ky, kz) = _get_kx(int(grid.N), grid=grid)
    k2 = kx**2 + ky**2 + kz**2
    t = 0.0
    (u_hat, v_hat, w_hat) = _fourier_transform(u)
    energy = []
    small_us = []
    times = []
    for _ in range(n_steps):
        small_us.append((u_hat, v_hat, w_hat))
        times.append(t)
        energy.append(compute_energy_dissipation(u_hat, v_hat, w_hat, k2, config.NU))
        (u_hat, v_hat, w_hat, t) = rk4_step(k2, t, u_hat, v_hat, w_hat, grid=grid, N=grid.N)

    _plot_diagnostics(times, energy, small_us, grid)
    return times, energy, small_us


def _plot_diagnostics(times, energy, small_us, grid: Grid3d):
    times = np.asarray(times)
    E_k = np.array([e[0] for e in energy])
    eps = np.array([e[1] for e in energy])

    # --- Energy & dissipation vs time ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, E_k, "b-")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("kinetic energy  E(t)")
    axes[0].set_title("Energy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, eps, "r-")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"dissipation  $\varepsilon(t)$")
    axes[1].set_title("Dissipation rate")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{config.ASSETS_DIR}/energy.png", dpi=120)
    plt.close(fig)

    # --- Energy balance check: dE/dt should equal -eps ---
    # Central difference on E(t) (forward/backward at the ends)
    dEdt = np.gradient(E_k, times)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, dEdt, "b-",  label=r"$dE/dt$ (finite diff)")
    axes[0].plot(times, -eps,  "r--", label=r"$-\varepsilon(t)$")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("rate")
    axes[0].set_title("Energy balance:  dE/dt  vs  -" + r"$\varepsilon$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Relative error |dE/dt + eps| / max(|eps|, tiny)
    denom = np.maximum(np.abs(eps), 1e-300)
    rel_err = np.abs(dEdt + eps) / denom
    axes[1].semilogy(times, rel_err, "k-")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$|dE/dt + \varepsilon| / |\varepsilon|$")
    axes[1].set_title("Relative imbalance (log scale)")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{config.ASSETS_DIR}/energy_balance.png", dpi=120)
    plt.close(fig)

    # Print a one-line summary on a window where signal is meaningful
    # (skip the very end where E has decayed to floating-point noise)
    mask = E_k > E_k[0] * 1e-10
    if mask.any():
        worst = rel_err[mask].max()
        print(f"[energy balance] max |dE/dt + eps|/|eps| over meaningful "
              f"window: {worst:.3e}  ({mask.sum()} points)")

    # --- Velocity magnitude snapshots (mid-z slice), SHARED color scale ---
    n_snap = min(6, len(small_us))
    snap_idx = np.linspace(0, len(small_us) - 1, n_snap).astype(int)
    z_mid = grid.N // 2

    # First pass: compute all slices to find a global vmax (so we can see decay).
    mags = []
    for idx in snap_idx:
        u_hat, v_hat, w_hat = small_us[idx]
        u = np.fft.irfftn(u_hat, s=(grid.N, grid.N, grid.N), axes=(0, 1, 2))
        v = np.fft.irfftn(v_hat, s=(grid.N, grid.N, grid.N), axes=(0, 1, 2))
        w = np.fft.irfftn(w_hat, s=(grid.N, grid.N, grid.N), axes=(0, 1, 2))
        mags.append(np.sqrt(u**2 + v**2 + w**2)[:, :, z_mid])
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
    fig.suptitle(r"|u| on mid-z slice (shared color scale)")
    fig.savefig(f"{config.ASSETS_DIR}/velocity_snapshots.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)


# For diagnostics
def compute_energy_dissipation(u_hat, v_hat, w_hat, k2, nu, V=config.L**3):
    """Return (E_k, eps) from Fourier coefficients."""
    # rfftn last axis has length N//2 + 1, so recover N from it.
    N_grid = 2 * (k2.shape[2] - 1)
    # Proper weighting for R2C half‑complex
    # This weight accounts for the fact that kz>0 modes represent both + and - kz
    weight = np.ones_like(k2)
    weight[:, :, 1:] = 2.0          # double count for kz>0
    # Nyquist mode (last) is its own conjugate if N even -> weight = 1
    if N_grid % 2 == 0:
        weight[:, :, -1] = 1.0

    abs_sq = (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
    E_k = 0.5 * np.sum(weight * abs_sq) / V

    eps = nu * np.sum(weight * k2 * abs_sq) / V
    return E_k, eps


run(Grid3d(N=64))

# _generate_initial_condition(4)
