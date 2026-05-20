# =============================================================================
# main.py — Entry point for the 2D Stokes flow solver.
#
# Run:
#   python main.py
#
# Builds one Uniqx IR module that runs the full Stokes iteration server-side,
# submits it to the gateway, fetches the result, reshapes (u, v, p), and saves
# a snapshot figure to assets/.
# =============================================================================

import os
import numpy as np

import uniqx as ux

import config
from grid import Grid
from solver import run
from visualize import plot_snapshots


def _parse_flat_payload(payload: bytes) -> np.ndarray:
    """Parse a single `Nxf64=v0 v1 …` payload into a 1-D numpy array."""
    text = payload.decode("latin-1") if isinstance(payload, (bytes, bytearray)) else payload
    _, _, values = text.strip().partition("=")
    return np.fromstring(values, sep=" ", dtype=np.float64)


def _split_uvp(flat: np.ndarray, N: int):
    """Reverse of solver.iterate's concat: split into (u, v, p)."""
    field = (N + 2) * (N + 2)
    u = flat[0:field].reshape(N + 2, N + 2)
    v = flat[field:2 * field].reshape(N + 2, N + 2)
    p = flat[2 * field:2 * field + N * N].reshape(N, N)
    return u, v, p


def main():
    print("=" * 60)
    print(" ORIQX CFD — 2D Incompressible Stokes Flow Solver")
    print(" Chorin's Projection Method  •  single-module server-side run")
    print("=" * 60)

    grid = Grid()
    print(f"\n{grid}\n")

    mod, runtime_inputs = run(grid)
    print(f"[main] module built — submitting to gateway…", flush=True)

    gateway = os.environ.get("UNIQX_GATEWAY", "api.oriqx.com:443")
    api_key = os.environ.get("UNIQX_API_KEY")
    client = ux.connect(gateway, api_key=api_key)

    job_id = ux.submit(mod, client=client, runtime_inputs=runtime_inputs)
    print(f"[main] job_id = {job_id}", flush=True)

    res = ux.get(job_id, client=client, timeout=600.0)
    if res.get("state") != 10:
        payload = res.get("payload") or res.get("result_payload") or b""
        raise SystemExit(f"[main] job failed (state={res.get('state')}): {payload!r}")

    payload = res.get("payload") or res.get("result_payload")
    flat = _parse_flat_payload(payload)
    u, v, p = _split_uvp(flat, grid.N)
    print(f"[main] received  u{u.shape}  v{v.shape}  p{p.shape}", flush=True)
    print(f"[main]   max|u|={np.max(np.abs(u)):.6e}  "
          f"max|v|={np.max(np.abs(v)):.6e}  "
          f"max|p|={np.max(np.abs(p)):.6e}")

    # Single end-of-run snapshot for visualization.
    assets = config.ASSETS_DIR
    os.makedirs(assets, exist_ok=True)
    plot_snapshots(grid, [(config.N_STEPS, u, v, p)],
                   save_path=f"{assets}/results.png")
    np.savez(f"{assets}/snapshots.npz",
             **{f"u_{config.N_STEPS:04d}": u,
                f"v_{config.N_STEPS:04d}": v,
                f"p_{config.N_STEPS:04d}": p})

    print("[main] Done.")


if __name__ == "__main__":
    main()
