import os
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import uniqx as ux
from grid import Grid
import jax_solve
import solver_copy as uniqx_solver
import config

# Default gateway and key from main.py
DEFAULT_GATEWAY = "api.oriqx.com:443"
DEFAULT_API_KEY = "uxk_f1fbb49476f7e4606c87f1dccc0a7df5"

def run_uniqx_benchmark(grid, n_steps, gateway, api_key, backend="compiled"):
    """Executes the Uniqx solver and returns the elapsed wall time."""
    mod, runtime_inputs = uniqx_solver.run(grid, n_steps=n_steps)
    # backend = "Auto"
    runtime_inputs.append(f"backend={backend}")
    
    client = ux.connect(gateway, api_key=api_key)
    
    t0 = time.perf_counter()
    job_id = ux.submit(mod, client=client, runtime_inputs=runtime_inputs)
    # print(job_id)
    res = ux.get(job_id, client=client, timeout=900.0)
    elapsed = time.perf_counter() - t0
    
    if res.get("state") != 10:
        print(f"  [Uniqx] Warning: Job failed for N={grid.N} with backend={backend}")
        return None
    
    return elapsed

def benchmark_scaling(start_n=32, end_n=64, step_size=4, n_steps=50, backend="compiled"):
    """
    Benchmarks both JAX and Uniqx solver performance across different grid sizes.
    """
    n_values = list(range(start_n, end_n + 1, step_size))
    jax_results = []
    uniqx_results = []

    print("=" * 70)
    print(f"  CFD Scaling Benchmark: N = {start_n} to {end_n} | Backend: {backend}")
    print("=" * 70)
    print(f"{'N':>5} | {'JAX (ms/step)':>15} | {f'Uniqx-{backend} (ms/step)':>25}")
    print("-" * 70)

    # Ensure assets directory exists
    os.makedirs(config.ASSETS_DIR, exist_ok=True)

    for n in n_values:
        grid = Grid(N=n)
        
        # --- JAX Benchmark ---
        # Warm-up
        jax_solve.run(grid, n_steps=1)
        # Run
        jax_res = jax_solve.run(grid, n_steps=n_steps)
        jax_ms = (jax_res["elapsed"] / jax_res["step"]) * 1000
        jax_results.append(jax_ms)

        # --- Uniqx Benchmark ---
        # Note: Uniqx doesn't need a local warm-up for the remote run in the same way,
        # but the first run might include server-side cold start. 
        # We'll do one warm-up submission anyway.
        run_uniqx_benchmark(grid, n_steps=1, gateway=DEFAULT_GATEWAY, api_key=DEFAULT_API_KEY, backend=backend)
        
        uniqx_elapsed = run_uniqx_benchmark(grid, n_steps=n_steps, gateway=DEFAULT_GATEWAY, api_key=DEFAULT_API_KEY, backend=backend)
        if uniqx_elapsed is not None:
            uniqx_ms = (uniqx_elapsed / n_steps) * 1000
            uniqx_results.append(uniqx_ms)
        else:
            uniqx_results.append(np.nan)
        
        print(f"{n:5d} | {jax_ms:15.3f} | {uniqx_results[-1]:25.3f}")

    # --- Save Data to CSV ---
    csv_path = f"scaling_results_{backend}_task_2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "jax_ms_per_step", f"uniqx_{backend}_ms_per_step"])
        for i, n in enumerate(n_values):
            writer.writerow([n, jax_results[i], uniqx_results[i]])
    
    # --- Create Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, jax_results, 'o-', linewidth=2, label='JAX (Local)')
    plt.plot(n_values, uniqx_results, 's--', linewidth=2, label=f'Uniqx ({backend})')
    
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('Wall Time per Step (ms)', fontsize=12)
    plt.title(f'Solver Performance Scaling: JAX vs. Uniqx ({backend})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot_path = os.path.join(config.ASSETS_DIR, f"scaling_plot_{backend}_task_2.png")
    plt.savefig(plot_path)
    
    print("-" * 70)
    print(f"Benchmark Complete.")
    print(f"  - Data saved to: {csv_path}")
    print(f"  - Plot saved to: {plot_path}")

if __name__ == "__main__":
    # You can now easily change the backend here
    benchmark_scaling(start_n=16, end_n=64, step_size=8, n_steps=50, backend="compiled")
