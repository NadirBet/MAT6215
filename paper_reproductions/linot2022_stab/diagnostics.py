from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import numpy as np

from ks_solver import KSSolver
from paper_reproductions.linot2022_stab.data import add_band_limited_noise, sample_rollout_start_indices
from paper_reproductions.linot2022_stab.models import KSEContext, ModelBundle

base_diagnostics = importlib.import_module("diagnostics")


@dataclass(frozen=True)
class DiagnosticConfig:
    spacetime_steps: int = 2000
    ensemble_size: int = 100
    short_time_steps: int = 440
    spectrum_steps: int = 2000
    joint_pdf_steps: int = 4000
    noise_steps: int = 1200
    tau_l: float = 22.0
    noise_levels: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_spacetime_panels(systems: list[tuple[str, np.ndarray]], x_grid: np.ndarray, path: Path) -> None:
    n_rows = len(systems)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9.5, 2.2 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = [axes]
    mappable = None
    u_abs_max = float(np.max(np.abs(systems[0][1])))
    for ax, (title, traj) in zip(axes, systems):
        mappable = ax.imshow(
            traj.T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-u_abs_max,
            vmax=u_abs_max,
            extent=[0, traj.shape[0], x_grid[0], x_grid[-1]],
            interpolation="nearest",
        )
        ax.set_ylabel("x")
        ax.set_title(title)
    axes[-1].set_xlabel("Step")
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("u")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_energy_spectrum(spectra: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, (q, energy) in spectra.items():
        ax.loglog(q[1:], energy[1:], label=label)
    ax.set_xlabel("Wavenumber q")
    ax.set_ylabel("E(q)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_joint_pdf_panels(pdf_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    labels = list(pdf_payload.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(4.1 * len(labels), 4.2), sharex=True, sharey=True)
    if len(labels) == 1:
        axes = [axes]
    vmax = max(float(pdf.max()) for _, _, pdf in pdf_payload.values())
    mappable = None
    for ax, label in zip(axes, labels):
        ux_edges, uxx_edges, pdf = pdf_payload[label]
        mappable = ax.pcolormesh(ux_edges, uxx_edges, pdf.T, shading="auto", cmap="magma", vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel(r"$u_x$")
    axes[0].set_ylabel(r"$u_{xx}$")
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("PDF")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ensemble_error(curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, (t_norm, mean_err, std_err) in curves.items():
        ax.plot(t_norm, mean_err, label=label)
        ax.fill_between(t_norm, mean_err - std_err, mean_err + std_err, alpha=0.15)
    ax.set_xlabel(r"$t / \tau_L$")
    ax.set_ylabel("Relative error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_noise_robustness(panels: list[tuple[str, np.ndarray]], x_grid: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 2.1 * len(panels)), sharex=True, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    mappable = None
    u_abs_max = max(float(np.max(np.abs(traj))) for _, traj in panels)
    for ax, (label, traj) in zip(axes, panels):
        mappable = ax.imshow(
            traj.T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-u_abs_max,
            vmax=u_abs_max,
            extent=[0, traj.shape[0], x_grid[0], x_grid[-1]],
            interpolation="nearest",
        )
        ax.set_ylabel("x")
        ax.set_title(label)
    axes[-1].set_xlabel("Step")
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("u")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_paper_diagnostics(
    *,
    solver: KSSolver,
    context: KSEContext,
    test_states: np.ndarray,
    trained_models: dict[str, dict],
    figure_dir: Path,
    data_dir: Path,
    config: DiagnosticConfig = DiagnosticConfig(),
) -> dict:
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    x_grid = solver.x_grid()
    u0 = np.asarray(test_states[0])

    spacetime_true = np.asarray(test_states[1:1 + config.spacetime_steps])
    spacetime_models = {}
    for artifact in trained_models.values():
        bundle: ModelBundle = artifact["bundle"]
        params = artifact["params"]
        spacetime_models[bundle.name] = np.asarray(bundle.rollout(params, u0, config.spacetime_steps))

    plot_spacetime_panels(
        [
            ("True KSE", spacetime_true),
            ("Standard NODE", spacetime_models["nonlinear"]),
            ("Fixed-linear NODE", spacetime_models["fixed_linear"]),
            ("CNN NODE", spacetime_models["cnn"]),
        ],
        x_grid,
        figure_dir / "fig6_spacetime.png",
    )

    ensemble_indices = sample_rollout_start_indices(
        test_states,
        n_windows=config.ensemble_size,
        window_steps=config.short_time_steps,
    )
    true_ensemble = np.stack(
        [test_states[idx + 1:idx + 1 + config.short_time_steps] for idx in ensemble_indices],
        axis=0,
    )
    error_curves = {}
    for artifact in trained_models.values():
        bundle: ModelBundle = artifact["bundle"]
        params = artifact["params"]
        initial_conditions = np.asarray(test_states[ensemble_indices])
        batched_rollout = jax.jit(
            jax.vmap(
                lambda u_init: bundle.rollout(params, u_init, config.short_time_steps),
                in_axes=0,
                out_axes=0,
            )
        )
        pred_ensemble = np.asarray(batched_rollout(initial_conditions))
        error_curves[bundle.display_name] = base_diagnostics.ensemble_error(
            true_ensemble,
            pred_ensemble,
            dt=context.tau,
            lyapunov_time=config.tau_l,
        )
    plot_ensemble_error(error_curves, figure_dir / "fig7_ensemble_error.png")

    long_true = np.asarray(test_states[:config.spectrum_steps])
    long_rollouts = {"True KSE": long_true}
    for artifact in trained_models.values():
        bundle: ModelBundle = artifact["bundle"]
        params = artifact["params"]
        long_rollouts[bundle.display_name] = np.asarray(bundle.rollout(params, u0, config.spectrum_steps))

    spectra = {
        label: base_diagnostics.spatial_power_spectrum(traj, L=context.L)
        for label, traj in long_rollouts.items()
    }
    plot_energy_spectrum(spectra, figure_dir / "fig_energy_spectrum.png")

    pdf_true = np.asarray(test_states[:config.joint_pdf_steps])
    pdf_rollouts = {"True KSE": pdf_true}
    for artifact in trained_models.values():
        bundle: ModelBundle = artifact["bundle"]
        params = artifact["params"]
        pdf_rollouts[bundle.display_name] = np.asarray(bundle.rollout(params, u0, config.joint_pdf_steps))
    pdf_payload = {
        label: base_diagnostics.joint_pdf_derivatives(traj, L=context.L, n_bins=200)
        for label, traj in pdf_rollouts.items()
    }
    plot_joint_pdf_panels(pdf_payload, figure_dir / "fig8_joint_pdf.png")

    cnn_bundle = trained_models["cnn"]["bundle"]
    cnn_params = trained_models["cnn"]["params"]
    noise_panels = []
    for idx, eps in enumerate(config.noise_levels):
        noisy_u0 = add_band_limited_noise(u0, epsilon=eps, seed=idx)
        noise_panels.append(
            (
                f"CNN NODE, epsilon={eps:g}",
                np.asarray(cnn_bundle.rollout(cnn_params, noisy_u0, config.noise_steps)),
            )
        )
    plot_noise_robustness(noise_panels, x_grid, figure_dir / "fig9_noise_robustness.png")

    summary = {
        "config": asdict(config),
        "spacetime_steps": config.spacetime_steps,
        "ensemble_indices": ensemble_indices.tolist(),
        "spectra_labels": list(spectra.keys()),
        "joint_pdf_labels": list(pdf_payload.keys()),
        "bonus_outputs": ["fig_energy_spectrum.png"],
    }
    write_json(data_dir / "paper_diagnostics_summary.json", summary)
    return summary
