from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_reproductions.brunton2016_wake.wake_sindy import (
    load_velocity_state_matrix,
    simulate_sindy,
    snapshot_pod,
)


OUT_DIR = ROOT / "paper_reproductions" / "brunton2016_wake"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"


def load_results():
    coeffs = np.load(RES_DIR / "wake_pod_coefficients.npz")
    df3 = pd.read_csv(RES_DIR / "wake_sindy_coefficients_3d.csv")
    xi3 = df3[["xdot", "ydot", "zdot"]].to_numpy(dtype=np.float64)
    return coeffs, df3, xi3


def set_shared_limits(axes, arrays):
    stacked = np.vstack(arrays)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins)
    for ax in axes:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-52)


def plot_colored_trajectory(ax, xyz, times, cmap="viridis", s=12, alpha=0.95):
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.75", lw=0.8, alpha=0.45)
    pts = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=times, cmap=cmap, s=s, alpha=alpha)
    return pts


def make_figure8(coeffs3, rollout3, times):
    fig = plt.figure(figsize=(11.5, 5.2), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    pts1 = plot_colored_trajectory(ax1, coeffs3, times)
    pts2 = plot_colored_trajectory(ax2, rollout3, times)
    ax1.set_title("Full Simulation")
    ax2.set_title("Identified System")
    set_shared_limits([ax1, ax2], [coeffs3, rollout3])

    cbar = fig.colorbar(pts2, ax=[ax1, ax2], fraction=0.035, pad=0.04)
    cbar.set_label("Simulation time")
    fig.suptitle("Figure 8 Style Wake Trajectory in Reduced Coordinates")
    fig.savefig(FIG_DIR / "wake_figure8_paper_style.png", dpi=220)
    plt.close(fig)


def make_table5(df3):
    active = df3[(df3[["xdot", "ydot", "zdot"]].abs() > 1e-12).any(axis=1)].copy()
    display = active.copy()
    for col in ["xdot", "ydot", "zdot"]:
        display[col] = display[col].map(lambda x: f"{x:.4g}")

    fig_h = max(4.0, 0.38 * len(display) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("Table 5 Style Sparse Wake Coefficients", pad=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wake_table5_style.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_analogue_figure(path: Path, coeffs3, traj, initial_point, title_left: str, title_right: str, times):
    fig = plt.figure(figsize=(11.5, 5.2), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.plot(coeffs3[:, 0], coeffs3[:, 1], coeffs3[:, 2], color="0.65", lw=1.0, alpha=0.9)
    ax1.scatter(coeffs3[:, 0], coeffs3[:, 1], coeffs3[:, 2], color="0.82", s=6, alpha=0.35)
    ax1.scatter(*initial_point, color="#d62728", s=90, label="Initial state")
    ax1.legend(loc="upper right", frameon=False)

    pts = plot_colored_trajectory(ax2, traj, times)
    ax2.scatter(*initial_point, color="#d62728", s=90)

    ax1.set_title(title_left)
    ax2.set_title(title_right)
    set_shared_limits([ax1, ax2], [coeffs3, traj, initial_point[None, :]])

    cbar = fig.colorbar(pts, ax=[ax1, ax2], fraction=0.035, pad=0.04)
    cbar.set_label("Simulation time")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def contour_panel(ax, x, y, values, title, vmin=None, vmax=None):
    tri = mtri.Triangulation(x, y)
    levels = np.linspace(vmin if vmin is not None else float(values.min()),
                         vmax if vmax is not None else float(values.max()), 19)
    ax.tricontourf(tri, values, levels=levels, cmap="RdBu_r", extend="both")
    ax.tricontour(tri, values, levels=levels, colors="k", linewidths=0.35, alpha=0.55)
    ax.add_patch(Circle((0.0, 0.0), 0.5, facecolor="0.35", edgecolor="black", lw=1.0, zorder=10))
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 8.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks(np.arange(-1, 9, 1))
    ax.set_yticks(np.arange(-2, 3, 1))
    ax.set_title(title, loc="left", fontsize=12)


def make_figure7(coeffs, raw):
    state = raw["state"].astype(np.float64)
    times = raw["times"].astype(np.float64)
    x_all = raw["x"].astype(np.float64)
    y_all = raw["y"].astype(np.float64)
    pod = snapshot_pod(state, times, x_all, y_all, n_modes=3)

    coeffs3 = pod.coefficients[:, :3].astype(np.float64)
    n_nodes = x_all.size
    u_snapshots = raw["u"].astype(np.float64)

    A_idx = int(np.argmax(coeffs3[:, 0]))
    B_state = pod.mean_state
    z_c = float(np.min(coeffs3[:, 2]))
    C_state = pod.mean_state + z_c * pod.modes[:, 2]

    zoom_mask = (x_all >= -1.0) & (x_all <= 8.0) & (y_all >= -2.5) & (y_all <= 2.5)
    x = x_all[zoom_mask]
    y = y_all[zoom_mask]

    A_u = u_snapshots[A_idx][zoom_mask]
    B_u = B_state[:n_nodes][zoom_mask]
    C_u = C_state[:n_nodes][zoom_mask]
    mode1_u = pod.modes[:n_nodes, 0][zoom_mask]
    mode2_u = pod.modes[:n_nodes, 1][zoom_mask]
    mode3_u = pod.modes[:n_nodes, 2][zoom_mask]

    fields = [A_u, B_u, C_u, mode1_u, mode2_u, mode3_u]
    amp = max(float(np.max(np.abs(f))) for f in fields)

    fig = plt.figure(figsize=(13.5, 6.8), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.15, 1.0, 1.0])

    ax_geom = fig.add_subplot(gs[:, 0], projection="3d")
    ax_A = fig.add_subplot(gs[0, 1])
    ax_B = fig.add_subplot(gs[1, 1])
    ax_C = fig.add_subplot(gs[2, 1])
    ax_m1 = fig.add_subplot(gs[0, 2])
    ax_m2 = fig.add_subplot(gs[1, 2])
    ax_m3 = fig.add_subplot(gs[2, 2])

    # Left schematic: actual reduced trajectory plus an approximate slow manifold cone.
    B = np.array([0.0, 0.0, 0.0])
    A = coeffs3[A_idx]
    r0 = float(np.median(np.sqrt(coeffs3[:, 0] ** 2 + coeffs3[:, 1] ** 2)))
    theta = np.linspace(0, 2 * np.pi, 70)
    z_vals = np.linspace(z_c, 0.0, 40)
    Theta, Z = np.meshgrid(theta, z_vals)
    R = r0 * (Z - z_c) / max(1e-9, (0.0 - z_c))
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    ax_geom.plot_surface(X, Y, Z, color="#d8d7cc", alpha=0.42, linewidth=0, shade=True)
    ax_geom.plot(coeffs3[:, 0], coeffs3[:, 1], coeffs3[:, 2], color="#1f2a6d", lw=2.0)
    ax_geom.plot(r0 * np.cos(theta), r0 * np.sin(theta), np.zeros_like(theta), color="#1f2a6d", lw=1.5)
    ax_geom.scatter(*A, color="#8b1d1d", s=34)
    ax_geom.scatter(*B, color="#8b1d1d", s=26)
    ax_geom.scatter(0.0, 0.0, z_c, color="#8b1d1d", s=30)
    ax_geom.text(*(A + np.array([4.0, 4.0, 2.0])), "A", fontsize=12, fontstyle="italic")
    ax_geom.text(*(B + np.array([2.5, 1.5, 2.0])), "B", fontsize=12, fontstyle="italic")
    ax_geom.text(1.5, 1.0, z_c - 1.5, "C*", fontsize=12, fontstyle="italic")
    ax_geom.text(-0.9 * r0, -0.9 * r0, 2.0, "Limit cycle", fontsize=11)
    ax_geom.text(-0.85 * r0, -0.65 * r0, 0.5 * z_c, "Slow\nmanifold", fontsize=11)
    ax_geom.set_xlabel("x")
    ax_geom.set_ylabel("y")
    ax_geom.set_zlabel("z")
    ax_geom.set_title("Low-dimensional wake geometry", pad=6)
    ax_geom.view_init(elev=17, azim=-58)
    ax_geom.set_xlim(-1.2 * r0, 1.2 * r0)
    ax_geom.set_ylim(-1.2 * r0, 1.2 * r0)
    ax_geom.set_zlim(z_c - 2.0, max(6.0, np.max(coeffs3[:, 2]) + 2.0))

    contour_panel(ax_A, x, y, A_u, "A - vortex shedding", vmin=-amp, vmax=amp)
    contour_panel(ax_B, x, y, B_u, "B - mean flow", vmin=-amp, vmax=amp)
    contour_panel(ax_C, x, y, C_u, "C* - shift-like reconstruction", vmin=-amp, vmax=amp)

    mode_amp = max(float(np.max(np.abs(mode1_u))), float(np.max(np.abs(mode2_u))), float(np.max(np.abs(mode3_u))))
    contour_panel(ax_m1, x, y, mode1_u, r"$u_x$ - POD mode 1", vmin=-mode_amp, vmax=mode_amp)
    contour_panel(ax_m2, x, y, mode2_u, r"$u_y$ - POD mode 2", vmin=-mode_amp, vmax=mode_amp)
    contour_panel(ax_m3, x, y, mode3_u, r"$u_z$ - approx. shift mode", vmin=-mode_amp, vmax=mode_amp)

    for ax in [ax_A, ax_B, ax_m1, ax_m2]:
        ax.set_xticklabels([])
    for ax in [ax_m1, ax_m2, ax_m3]:
        ax.set_yticklabels([])

    fig.suptitle("Figure 7 Analogue: Low-rank wake structure from free Re=100 DNS data", fontsize=14)
    fig.savefig(FIG_DIR / "wake_figure7_analogue.png", dpi=220)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    coeffs, df3, xi3 = load_results()
    times = coeffs["times"].astype(np.float64)
    coeffs3 = coeffs["coefficients_3d"].astype(np.float64)
    rollout3 = coeffs["rollout_3d"].astype(np.float64)
    raw = load_velocity_state_matrix(ROOT / "paper_reproductions" / "brunton2016_wake" / "data" / "fixed_cylinder_atRe100", dtype=np.float32)

    make_figure8(coeffs3, rollout3, times)
    make_table5(df3)
    make_figure7(coeffs, raw)

    mean_cycle = coeffs3.mean(axis=0)
    traj_mean = simulate_sindy(mean_cycle, times, xi3, degree=5, feature_names=["x", "y", "z"])
    make_analogue_figure(
        FIG_DIR / "wake_figure9_analogue.png",
        coeffs3,
        traj_mean,
        mean_cycle,
        "Reference attractor",
        "Analogue from mean state",
        times,
    )

    doubled = 2.0 * coeffs3[0]
    traj_doubled = simulate_sindy(doubled, times, xi3, degree=5, feature_names=["x", "y", "z"])
    make_analogue_figure(
        FIG_DIR / "wake_figure10_analogue.png",
        coeffs3,
        traj_doubled,
        doubled,
        "Reference attractor",
        "Analogue from doubled initial state",
        times,
    )


if __name__ == "__main__":
    main()
