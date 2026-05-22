# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# main.py — Entry point for the uniqx 2D Stokes flow solver.
#
# Run:
#   python main.py [--steps N] [--n N] [--gateway host:port]
#
# Builds one Uniqx IR module that runs the full Stokes iteration server-side,
# submits it to the gateway, fetches the result, reshapes (u, v, p), and saves
# a snapshot figure to assets/.
# =============================================================================

import argparse
import os

import config
import numpy as np
import uniqx as ux
from grid import Grid
from solver import run
from visualize import plot_snapshots
import graphviz


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


def main(n: int = config.N, n_steps: int = config.N_STEPS, gateway: str | None = None) -> None:
    print("=" * 60)
    print("  ORIQX CFD — 2D Incompressible Stokes Flow Solver [uniqx]")
    print("  Chorin projection  |  single-module server-side run")
    print("=" * 60)

    grid = Grid(N=n)
    print(f"\n{grid}\n")

    mod, runtime_inputs = run(grid, n_steps=n_steps)
    runtime_inputs.append("backend=compiled")
    print("[main] module built — submitting to gateway…", flush=True)
    gateway =  "api.oriqx.com:443"
    api_key = "uxk_291c86ad1347b417e96d897d0d655a19"
    print("API_KEYYYYYYY", api_key)
    client = ux.connect(gateway, api_key=api_key)
    job_id = ux.submit(mod, client=client, runtime_inputs=runtime_inputs)
    print(job_id, "YAYYYY")
    # print(f"[main] job_id = {job_id}", flush=True)

    res = ux.get(job_id, client=client, timeout=600.0)
    print(res.keys())
    if res.get("state") != 10:
        payload = res.get("payload") or res.get("result_payload") or b""
        raise SystemExit(f"[main] job failed (state={res.get('state')}): {payload!r}")

    payload = res.get("payload") or res.get("result_payload")
    flat = _parse_flat_payload(payload)
    u, v, p = _split_uvp(flat, grid.N)
    print(f"[main] received  u{u.shape}  v{v.shape}  p{p.shape}", flush=True)

    # Single end-of-run snapshot for visualization.
    assets = config.ASSETS_DIR
    os.makedirs(assets, exist_ok=True)
    plot_snapshots(grid, [(n_steps, u, v, p)], save_path=f"{assets}/results.png")

    print("[main] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Stokes solver — uniqx gateway")
    parser.add_argument("--n", type=int, default=config.N,
                        help="interior grid points per side (default: %(default)s)")
    parser.add_argument("--steps", type=int, default=config.N_STEPS,
                        help="number of time steps baked into the IR module (default: %(default)s)")
    parser.add_argument("--gateway", default=None,
                        help="gateway address, overrides UNIQX_GATEWAY env var")
    args = parser.parse_args()
    main(n=args.n, n_steps=args.steps, gateway=args.gateway)
