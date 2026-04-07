from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x).ravel())
    y = np.sort(np.asarray(y).ravel())
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    n = max(len(x), len(y))
    p = np.linspace(0.0, 1.0, n, endpoint=True)
    xp = np.interp(p, np.linspace(0.0, 1.0, len(x), endpoint=True), x)
    yp = np.interp(p, np.linspace(0.0, 1.0, len(y), endpoint=True), y)
    return float(np.mean(np.abs(xp - yp)))


def compare_angle_distributions(
    reference_angles: np.ndarray,
    surrogate_angles: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict:
    per_pair = []
    for idx, pair in enumerate(pairs):
        dist = wasserstein_1d(reference_angles[:, idx], surrogate_angles[:, idx])
        per_pair.append(
            {
                "pair": tuple(int(v) for v in pair),
                "wasserstein_1d": dist,
                "reference_mean": float(np.mean(reference_angles[:, idx])),
                "surrogate_mean": float(np.mean(surrogate_angles[:, idx])),
            }
        )
    return {
        "per_pair": per_pair,
        "mean_wasserstein": float(np.mean([entry["wasserstein_1d"] for entry in per_pair])),
    }


def aligned_reference_spectrum(reference: np.ndarray, *, neutral_drop: int = 2) -> np.ndarray:
    reference = np.asarray(reference)
    if neutral_drop <= 0 or len(reference) <= neutral_drop:
        return reference
    neutral_idx = np.argsort(np.abs(reference))[:neutral_drop]
    mask = np.ones(len(reference), dtype=bool)
    mask[neutral_idx] = False
    return reference[mask]


def plot_reconstruction_curve(curves: dict, path: Path, *, title: str) -> None:
    latent_dim = np.asarray(curves["latent_dim"])
    val_mse = np.asarray(curves["val_mse"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(latent_dim, val_mse, marker="o", lw=1.8)
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Validation Reconstruction MSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lyapunov_spectrum(
    reference: np.ndarray,
    surrogate: np.ndarray,
    path: Path,
    *,
    title: str,
    surrogate_label: str = "CAE-ESN",
    skip_reference_neutral: int = 0,
    max_modes: int = 28,
) -> dict:
    ref = np.asarray(reference)
    if skip_reference_neutral > 0:
        ref = aligned_reference_spectrum(ref, neutral_drop=skip_reference_neutral)
    sur = np.asarray(surrogate)
    n = min(max_modes, len(ref), len(sur))
    ref = ref[:n]
    sur = sur[:n]
    mae = float(np.mean(np.abs(ref - sur)))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(1, n + 1)
    ax.plot(x, ref, marker="o", lw=1.6, label="Reference")
    ax.plot(x, sur, marker="s", lw=1.6, label=surrogate_label)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("Exponent Index")
    ax.set_ylabel("Lyapunov Exponent")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.98,
        f"MAE = {mae:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"n_compared": n, "mae": mae}


def plot_angle_histograms(
    reference_angles: np.ndarray,
    surrogate_angles: np.ndarray,
    pairs: list[tuple[int, int]],
    path: Path,
    *,
    title: str,
    surrogate_label: str = "CAE-ESN",
    extra_series: list[tuple[str, np.ndarray]] | None = None,
) -> dict:
    extra_series = extra_series or []
    comparison = compare_angle_distributions(reference_angles, surrogate_angles, pairs)

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5.0 * n_pairs, 3.9), squeeze=False)
    axes = axes[0]
    bins = np.linspace(0.0, 180.0, 51)

    for idx, pair in enumerate(pairs):
        ax = axes[idx]
        ax.hist(
            reference_angles[:, idx],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            label="Reference",
        )
        ax.hist(
            surrogate_angles[:, idx],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            label=surrogate_label,
        )
        for label, values in extra_series:
            ax.hist(
                values[:, idx],
                bins=bins,
                density=True,
                histtype="step",
                lw=1.4,
                label=label,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"CLV({pair[0]+1},{pair[1]+1})\nW1={comparison['per_pair'][idx]['wasserstein_1d']:.3f}"
        )
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return comparison
