# =============================================================================
# visualize.py — Post-processing: velocity heatmap, streamlines, pressure.
#
# Two public functions:
#   plot_snapshots(grid, snapshots, save_path)
#       Multi-row figure — one row per snapshot, three columns.
#       This is the primary output used by main.py.
#
#   plot_results(grid, u, v, p, save_path)
#       Single-row convenience wrapper (wraps plot_snapshots).
# =============================================================================

import warnings
import numpy as np
import matplotlib.pyplot as plt
from grid import Grid


# ── Internal helpers ──────────────────────────────────────────────────────────

def _embed_pressure(p: np.ndarray, N: int) -> np.ndarray:
    """Embed N×N interior pressure into (N+2)×(N+2) with Neumann ghost fill."""
    p_full = np.zeros((N + 2, N + 2))
    p_full[1:-1, 1:-1] = p
    p_full[0,  :]  = p_full[1,  :]
    p_full[-1, :]  = p_full[-2, :]
    p_full[:,  0]  = p_full[:,  1]
    p_full[:, -1]  = p_full[:, -2]
    return p_full


def _render_row(
    fig,
    axes,
    grid: Grid,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    step: int,
    row_label: bool = True,
) -> None:
    """
    Fill one row of three axes with (velocity magnitude | streamlines | pressure).
    Called once per snapshot by plot_snapshots.
    """
    X, Y   = grid.X, grid.Y
    speed  = np.sqrt(u**2 + v**2)
    p_full = _embed_pressure(p, grid.N)

    # --- Panel 1: velocity magnitude heatmap ---
    ax = axes[0]
    im = ax.imshow(
        speed,
        origin="lower",
        extent=[0, grid.L, 0, grid.L],
        cmap="hot",
        aspect="equal",
        vmin=0,
    )
    fig.colorbar(im, ax=ax, label="|u| [m/s]", shrink=0.85)
    ax.set_title("Velocity Magnitude")
    ax.set_xlabel("x");  ax.set_ylabel("y")
    if row_label:
        ax.set_ylabel(f"step {step}\ny", labelpad=4)

    # --- Panel 2: streamlines ---
    ax = axes[1]
    if speed.max() > 1e-10:          # skip if field is still all-zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strm = ax.streamplot(
                grid.x, grid.y, u, v,
                density=1.2, color=speed, cmap="cool", linewidth=0.8,
            )
            fig.colorbar(strm.lines, ax=ax, label="|u| [m/s]", shrink=0.85)
    ax.set_title("Streamlines")
    ax.set_xlabel("x");  ax.set_ylabel("y")
    ax.set_xlim(0, grid.L);  ax.set_ylim(0, grid.L)
    ax.set_aspect("equal")

    # --- Panel 3: pressure contours ---
    ax = axes[2]
    cf = ax.contourf(X, Y, p_full, levels=20, cmap="RdBu_r")
    ax.contour(X, Y, p_full, levels=20, colors="k", linewidths=0.4, alpha=0.4)
    fig.colorbar(cf, ax=ax, label="p [Pa]", shrink=0.85)
    ax.set_title("Pressure Field")
    ax.set_xlabel("x");  ax.set_ylabel("y")
    ax.set_aspect("equal")


# ── Primary output function ───────────────────────────────────────────────────

def plot_snapshots(
    grid:      Grid,
    snapshots: list,
    save_path: str = None,
) -> None:
    """
    Produce a (n_snapshots × 3) figure — one row per time snapshot.

    Parameters
    ----------
    grid      : Grid
    snapshots : list of (step, u, v, p) tuples, as returned by solver.run()
    save_path : str or None — write to file; None → plt.show()
    """
    n_rows = len(snapshots)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15, 4.5 * n_rows),
        squeeze=False,              # always (n_rows, 3) even for n_rows=1
    )
    fig.suptitle(
        "2D Incompressible Stokes Flow — Lid-Driven Cavity\n"
        "(one row per snapshot)",
        fontsize=12, y=1.01,
    )

    for row, (step, u, v, p) in enumerate(snapshots):
        _render_row(fig, axes[row], grid, u, v, p, step, row_label=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[visualize] Figure saved to '{save_path}'  "
              f"({n_rows} rows × 3 columns)")
    else:
        plt.show()

    plt.close(fig)


# ── Single-frame convenience wrapper ─────────────────────────────────────────

def plot_results(
    grid:      Grid,
    u:         np.ndarray,
    v:         np.ndarray,
    p:         np.ndarray,
    save_path: str = None,
    step:      int = -1,
) -> None:
    """Single-frame plot (delegates to plot_snapshots)."""
    plot_snapshots(grid, [(step, u, v, p)], save_path=save_path)
