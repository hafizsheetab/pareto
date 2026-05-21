# Copyright (c) 2026 ORIQX AG. MIT licensed.
# =============================================================================
# jax_main.py — Entry point for the JAX/NumPy lid-driven cavity solver.
#
# Run:
#   python jax_main.py [--solver direct|cg] [--steps N] [--n N]
#
# No uniqx gateway required. All computation runs locally via JAX and NumPy.
# Results are saved to assets/results_jax.png.
# =============================================================================

import argparse
import os

import config
from grid import Grid
from jax_solve import run


def main(
    solver: str = config.PRESSURE_SOLVER,
    n_steps: int = config.N_STEPS,
    n: int = config.N,
) -> None:
    print("=" * 60)
    print("  ORIQX CFD — 2D Lid-Driven Cavity [JAX]")
    print("  Chorin projection  |  FDM O(h²)  |  no gateway")
    print("=" * 60)

    grid = Grid(N=n)
    print(f"\n{grid}")
    Re = config.U_LID * grid.L / grid.nu
    print(f"  solver : {solver}  |  steps : {n_steps}  |  Re = {Re:.0f}\n")

    save_path = os.path.join(config.ASSETS_DIR, "results_jax.png")
    result = run(grid, n_steps=n_steps, solver=solver, save_path=save_path)

    step    = result["step"]
    elapsed = result["elapsed"]

    print(f"{'converged' if result['converged'] else 'max steps reached'} "
          f"after {step} steps  ({elapsed:.2f}s, {elapsed / step * 1e3:.1f} ms/step)")
    print(f"\n  figure saved → {save_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D lid-driven cavity — JAX solver")
    parser.add_argument("--solver", choices=["direct", "cg"], default=config.PRESSURE_SOLVER,
                        help="pressure Poisson solver (default: %(default)s)")
    parser.add_argument("--steps", type=int, default=config.N_STEPS,
                        help="maximum time steps (default: %(default)s)")
    parser.add_argument("--n", type=int, default=config.N,
                        help="interior grid points per side (default: %(default)s)")
    args = parser.parse_args()
    main(solver=args.solver, n_steps=args.steps, n=args.n)
