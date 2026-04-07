from __future__ import annotations

# ============================================================
# EXTENSION BEYOND LINOT 2022
# The paper does not compute Lyapunov spectra.
# We add dynamical-systems diagnostics on top of the paper's
# statistical and trajectory-based evaluation.
# ============================================================

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from ks_solver import KSSolver
from paper_reproductions.linot2022_stab.models import KSEContext, ModelBundle

base_lyapunov = importlib.import_module("lyapunov")

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class DynamicsConfig:
    n_steps: int = 4000
    n_exponents: int = 20
    reorthonormalize_every: int = 1


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def compute_discrete_map_spectrum(
    step_fn,
    params,
    u0: np.ndarray,
    *,
    dt: float,
    n_steps: int,
    n_exponents: int,
    reorthonormalize_every: int = 1,
) -> np.ndarray:
    if reorthonormalize_every != 1:
        raise NotImplementedError(
            "The current Linot 2022 extension assumes QR reorthonormalization every step."
        )

    u0_j = jnp.asarray(u0)
    n_state = u0_j.shape[0]
    q0 = jnp.eye(n_state, n_exponents, dtype=jnp.float64)
    log0 = jnp.zeros((n_exponents,), dtype=jnp.float64)

    def scan_step(carry, _):
        u, q_frame, log_sum = carry
        q_raw = jax.vmap(
            lambda q: jax.jvp(lambda x: step_fn(params, x), (u,), (q,))[1],
            in_axes=1,
            out_axes=1,
        )(q_frame)
        u_next = step_fn(params, u)
        q_next, r = jnp.linalg.qr(q_raw)
        diag_r = jnp.diag(r)
        signs = jnp.where(diag_r >= 0.0, 1.0, -1.0)
        q_next = q_next * signs[None, :]
        log_sum = log_sum + jnp.log(jnp.abs(diag_r))
        return (u_next, q_next, log_sum), None

    (_, _, log_total), _ = jax.lax.scan(
        scan_step,
        (u0_j, q0, log0),
        None,
        length=n_steps,
    )
    total_time = n_steps * dt
    return np.asarray(log_total / max(total_time, 1e-12))


def summarize_spectrum(exponents: np.ndarray, label: str) -> dict:
    n_pos = int(np.sum(exponents > 0.0))
    return {
        "label": label,
        "lambda_1": float(exponents[0]),
        "n_positive": n_pos,
        "kaplan_yorke_dim": float(base_lyapunov.kaplan_yorke_dimension(exponents)),
        "ks_entropy": float(base_lyapunov.ks_entropy(exponents)),
        "exponents": [float(x) for x in exponents],
    }


def plot_spectra(summaries: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label, summary in summaries.items():
        exponents = np.asarray(summary["exponents"])
        ax.plot(np.arange(1, len(exponents) + 1), exponents, marker="o", ms=3, label=label)
    ax.axhline(0.0, color="black", lw=1, alpha=0.5)
    ax.set_xlabel("Exponent index")
    ax.set_ylabel("Lyapunov exponent")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_bars(summaries: dict[str, dict], path: Path) -> None:
    labels = list(summaries.keys())
    metrics = ["lambda_1", "n_positive", "kaplan_yorke_dim", "ks_entropy"]
    titles = [r"$\lambda_1$", r"$n_{pos}$", r"$D_{KY}$", r"$h_{KS}$"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 4.2))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [summaries[label][metric] for label in labels]
        ax.bar(labels, vals)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_dynamics_extension(
    *,
    solver: KSSolver,
    context: KSEContext,
    test_states: np.ndarray,
    trained_models: dict[str, dict],
    data_dir: Path,
    figure_dir: Path,
    config: DynamicsConfig = DynamicsConfig(),
) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    u0 = np.asarray(test_states[0])
    true_exponents, _, _ = base_lyapunov.compute_lyapunov_spectrum_jit(
        solver,
        jnp.asarray(u0),
        n_steps=config.n_steps,
        n_modes=config.n_exponents,
    )
    summaries = {"True KSE": summarize_spectrum(np.asarray(true_exponents), "True KSE")}

    for artifact in trained_models.values():
        bundle: ModelBundle = artifact["bundle"]
        params = artifact["params"]
        exponents = compute_discrete_map_spectrum(
            bundle.step,
            params,
            u0,
            dt=context.tau,
            n_steps=config.n_steps,
            n_exponents=config.n_exponents,
            reorthonormalize_every=config.reorthonormalize_every,
        )
        summaries[bundle.display_name] = summarize_spectrum(exponents, bundle.display_name)

    write_json(
        data_dir / "dynamics_extension_summary.json",
        {
            "config": asdict(config),
            "systems": summaries,
        },
    )
    plot_spectra(summaries, figure_dir / "fig_ext_lyapunov_spectrum.png")
    plot_metric_bars(summaries, figure_dir / "fig_ext_metrics_bars.png")
    return summaries
