from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_reproductions.brunton2016_wake.wake_sindy import (
    coeff_table_rows,
    fit_sindy,
    load_velocity_state_matrix,
    snapshot_pod,
    simulate_sindy,
)


RAW_PATH = ROOT / "paper_reproductions" / "brunton2016_wake" / "data" / "fixed_cylinder_atRe100"
OUT_DIR = ROOT / "paper_reproductions" / "brunton2016_wake"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "results"


def save_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_energy(energy_fraction: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    cumulative = np.cumsum(energy_fraction)
    idx = np.arange(1, len(energy_fraction) + 1)
    plt.plot(idx, 100 * cumulative, marker="o")
    plt.xlabel("Mode")
    plt.ylabel("Cumulative energy (%)")
    plt.title("Wake POD energy capture")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_mode_timeseries(times: np.ndarray, coeffs: np.ndarray, path: Path) -> None:
    labels = ["x", "y", "z"]
    plt.figure(figsize=(9, 5))
    for i in range(min(3, coeffs.shape[1])):
        plt.plot(times, coeffs[:, i], label=labels[i])
    plt.xlabel("Time")
    plt.ylabel("Coefficient")
    plt.title("First POD coefficients")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_figure8_like(times: np.ndarray, coeffs_true: np.ndarray, coeffs_model: np.ndarray, path: Path) -> None:
    fig = plt.figure(figsize=(11, 5))
    cmap = plt.get_cmap("viridis")

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    p1 = ax1.scatter(coeffs_true[:, 0], coeffs_true[:, 1], coeffs_true[:, 2], c=times, cmap=cmap, s=18)
    ax1.set_title("DNS POD trajectory")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    p2 = ax2.scatter(coeffs_model[:, 0], coeffs_model[:, 1], coeffs_model[:, 2], c=times, cmap=cmap, s=18)
    ax2.set_title("Identified SINDy trajectory")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    cbar = fig.colorbar(p2, ax=[ax1, ax2], shrink=0.8)
    cbar.set_label("Time")
    fig.suptitle("Figure 8 style wake comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def select_best_fit(fits):
    stable = [fit for fit in fits if fit.stable]
    if stable:
        return min(stable, key=lambda fit: (fit.rollout_rmse, fit.n_active))
    return min(fits, key=lambda fit: (fit.rollout_rmse, fit.n_active))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading wake DNS data...")
    data = load_velocity_state_matrix(RAW_PATH, dtype=np.float32)
    print(f"Loaded state matrix: {data['state'].shape}")

    print("Computing POD...")
    pod = snapshot_pod(data["state"], data["times"], data["x"], data["y"], n_modes=6)

    coeffs3 = pod.coefficients[:, :3].astype(np.float64)
    coeffs2 = pod.coefficients[:, :2].astype(np.float64)
    times = pod.times.astype(np.float64)

    thresholds = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    print("Fitting 3D wake SINDy models...")
    fits3 = [fit_sindy(coeffs3, times, degree=5, threshold=thr, feature_names=["x", "y", "z"]) for thr in thresholds]
    best3 = select_best_fit(fits3)

    print("Fitting 2D wake SINDy models...")
    fits2 = [fit_sindy(coeffs2, times, degree=5, threshold=thr, feature_names=["x", "y"]) for thr in thresholds]
    best2 = select_best_fit(fits2)

    rollout3 = simulate_sindy(coeffs3[0], times, best3.coefficients, degree=5, feature_names=["x", "y", "z"])
    rollout2 = simulate_sindy(coeffs2[0], times, best2.coefficients, degree=5, feature_names=["x", "y"])

    save_csv(coeff_table_rows(best3, ["xdot", "ydot", "zdot"]), DATA_DIR / "wake_sindy_coefficients_3d.csv")
    save_csv(coeff_table_rows(best2, ["xdot", "ydot"]), DATA_DIR / "wake_sindy_coefficients_2d.csv")

    results = {
        "dataset": {
            "n_times": int(len(times)),
            "n_nodes": int(len(data["x"])),
            "dt": float(pod.dt),
            "time_start": float(times[0]),
            "time_end": float(times[-1]),
        },
        "pod": {
            "mode_energy_fraction": [float(x) for x in pod.energy_fraction.tolist()],
            "mode_energy_cumulative": [float(x) for x in np.cumsum(pod.energy_fraction).tolist()],
            "mode_std": [float(x) for x in np.std(pod.coefficients[:, :6], axis=0).tolist()],
        },
        "sindy_3d": {
            "best_threshold": best3.threshold,
            "n_active": best3.n_active,
            "derivative_rmse": best3.derivative_rmse,
            "rollout_rmse": best3.rollout_rmse,
            "stable": best3.stable,
            "threshold_sweep": [
                {
                    "threshold": fit.threshold,
                    "n_active": fit.n_active,
                    "derivative_rmse": fit.derivative_rmse,
                    "rollout_rmse": fit.rollout_rmse,
                    "stable": fit.stable,
                }
                for fit in fits3
            ],
        },
        "sindy_2d": {
            "best_threshold": best2.threshold,
            "n_active": best2.n_active,
            "derivative_rmse": best2.derivative_rmse,
            "rollout_rmse": best2.rollout_rmse,
            "stable": best2.stable,
            "threshold_sweep": [
                {
                    "threshold": fit.threshold,
                    "n_active": fit.n_active,
                    "derivative_rmse": fit.derivative_rmse,
                    "rollout_rmse": fit.rollout_rmse,
                    "stable": fit.stable,
                }
                for fit in fits2
            ],
        },
    }

    with (DATA_DIR / "wake_sindy_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_energy(pod.energy_fraction, FIG_DIR / "wake_pod_energy.png")
    plot_mode_timeseries(times, coeffs3, FIG_DIR / "wake_pod_timeseries.png")
    plot_figure8_like(times, coeffs3, rollout3, FIG_DIR / "wake_figure8_like_3d.png")

    np.savez_compressed(
        DATA_DIR / "wake_pod_coefficients.npz",
        times=times,
        coefficients_3d=coeffs3,
        rollout_3d=rollout3,
        coefficients_2d=coeffs2,
        rollout_2d=rollout2,
        energy_fraction=pod.energy_fraction,
        singular_values=pod.singular_values,
    )

    print("Done.")
    print(json.dumps(
        {
            "3d_best_threshold": best3.threshold,
            "3d_n_active": best3.n_active,
            "3d_rollout_rmse": best3.rollout_rmse,
            "2d_best_threshold": best2.threshold,
            "2d_n_active": best2.n_active,
            "2d_rollout_rmse": best2.rollout_rmse,
            "energy_first3": np.cumsum(pod.energy_fraction[:3]).tolist(),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
