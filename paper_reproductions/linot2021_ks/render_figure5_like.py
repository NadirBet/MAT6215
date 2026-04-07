from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_reproductions" / "linot2021_ks"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"

L_VALUE = 22.0
DT = 0.25
STATE_DIM = 64
U_CLIM = 3.0


def load_core_results(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def best_shift_l2(true_state: np.ndarray, pred_state: np.ndarray) -> tuple[int, float]:
    best_shift = 0
    best_norm = float("inf")
    for shift in range(true_state.size):
        shifted = np.roll(pred_state, shift)
        cur = float(np.linalg.norm(true_state - shifted))
        if cur < best_norm:
            best_norm = cur
            best_shift = shift
    return best_shift, best_norm


def build_error_curves(true_traj: np.ndarray, pred_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unshifted = np.linalg.norm(pred_traj - true_traj, axis=1)
    shifted = np.zeros_like(unshifted)
    for i in range(len(true_traj)):
        _, shifted[i] = best_shift_l2(true_traj[i], pred_traj[i])
    return unshifted, shifted


def plot_figure5_like(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    *,
    model_label: str,
    output_path: Path,
    summary_path: Path,
    time_max: float = 100.0,
    dt: float = DT,
    L: float = L_VALUE,
) -> None:
    n_steps_total = min(len(true_traj), len(pred_traj))
    n_steps = min(n_steps_total, int(round(time_max / dt)) + 1)

    true_plot = np.asarray(true_traj[:n_steps])
    pred_plot = np.asarray(pred_traj[:n_steps])
    abs_err = np.abs(pred_plot - true_plot)
    unshifted, shifted = build_error_curves(true_plot, pred_plot)

    t_vals = np.arange(n_steps) * dt
    x_coords = np.linspace(-L / 2.0, L / 2.0, true_plot.shape[1], endpoint=False)

    err_vmax = max(np.percentile(abs_err, 99), 1e-6)

    fig = plt.figure(figsize=(5.6, 6.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.9], hspace=0.22)

    panels = [
        ("a)", true_plot.T, "u", (-U_CLIM, U_CLIM), "RdBu_r"),
        ("b)", pred_plot.T, "u", (-U_CLIM, U_CLIM), "RdBu_r"),
        ("c)", abs_err.T, "|u - û|", (0.0, err_vmax), "Reds"),
    ]

    for row, (tag, field, cbar_label, clim, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(
            field,
            origin="lower",
            aspect="auto",
            extent=(t_vals[0], t_vals[-1], x_coords[0], x_coords[-1]),
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            interpolation="nearest",
        )
        ax.text(-0.11, 0.92, tag, transform=ax.transAxes, fontsize=11)
        ax.set_ylabel("x")
        if row < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("t")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label(cbar_label, rotation=90)

    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t_vals, unshifted, label="Unshifted", color="#1565c0", lw=1.6)
    ax.plot(t_vals, shifted, label="Shifted", color="#ef6c00", lw=1.6, ls="--")
    ax.text(-0.11, 0.92, "d)", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|u-\hat{u}\|$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle(f"Figure 5-style trajectory panel ({model_label}, L=22)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model": model_label,
        "time_max": float(t_vals[-1]),
        "n_steps": int(n_steps),
        "unshifted_error_mean": float(np.mean(unshifted)),
        "unshifted_error_final": float(unshifted[-1]),
        "shifted_error_mean": float(np.mean(shifted)),
        "shifted_error_final": float(shifted[-1]),
        "raw_abs_error_max": float(abs_err.max()),
        "raw_abs_error_p99": float(np.percentile(abs_err, 99)),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("latent", "physical", "fourier"), default="latent")
    parser.add_argument("--time-max", type=float, default=100.0)
    parser.add_argument(
        "--input",
        type=Path,
        default=RES_DIR / "figure3_l22_core.pkl",
    )
    args = parser.parse_args()

    results = load_core_results(args.input)
    true_traj = np.asarray(results["true_traj"])
    pred_traj = np.asarray(results[f"{args.model}_traj"])

    output_path = FIG_DIR / f"figure5_like_l22_{args.model}.png"
    summary_path = RES_DIR / f"figure5_like_l22_{args.model}.json"
    plot_figure5_like(
        true_traj,
        pred_traj,
        model_label=args.model.capitalize(),
        output_path=output_path,
        summary_path=summary_path,
        time_max=args.time_max,
    )
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
