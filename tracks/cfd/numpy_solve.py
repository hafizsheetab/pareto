import numpy as np
import config_3d as config
import math
from grid_3d import Grid3d

def main(N=16):
    u0 = _generate_initial_condition(N)


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
    grid = Grid3d(N=n)
    return (result, grid)

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
def _projector(ft_stacked, N, grid: Grid3d):
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
    (kx, ky, kz) = _get_kx(int(N), grid=grid)
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1.0   # avoid division by zero
    (px_hat, py_hat, pz_hat) = project(-Nx_hat, -Ny_hat, -Nz_hat, kx, ky, kz, k2)
    print(print(np.shape(px_hat)))

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
    
    # kx = np.fft.fftfreq()
    # M = 3/2 * N
    # u_hat

    # print(np.array(u_hat))
    # print(len(u_hat), len(v_hat), len(w_hat))

# _projector(_generate_initial_condition(16), 4)

def run(grid: Grid3d,
    n_steps: int = config.N_STEPS):
    for i in range(n_steps):
        return
    return
N = 16
(initial_condition, grid) = _generate_initial_condition(N)
ft_stacked = _fourier_transform(initial_condition)
_projector(ft_stacked, N,grid=grid)