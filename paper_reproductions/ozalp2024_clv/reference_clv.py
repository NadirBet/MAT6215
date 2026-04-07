from __future__ import annotations

import importlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ks_solver import KSSolver
from paper_reproductions.ozalp2024_clv.data import OzalpPreset

base_lyapunov = importlib.import_module("lyapunov")

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ReferenceCLVConfig:
    n_steps: int = 12000
    n_clv: int = 12
    seed: int = 0
    n_warmup: int = 2000


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def default_angle_pairs(n_positive: int) -> list[tuple[int, int]]:
    if n_positive >= 3:
        return [(0, 2), (0, 4), (2, 4)]
    if n_positive >= 2:
        return [(0, 1), (0, 2), (1, 2)]
    return [(0, 1)]


def compute_reference_clv(
    preset: OzalpPreset,
    *,
    config: ReferenceCLVConfig,
    data_dir: Path,
) -> dict:
    solver = KSSolver(L=preset.domain_length, N=preset.state_dim, dt=preset.dt)
    key = jax.random.PRNGKey(config.seed)
    u0_hat = solver.random_ic(key)
    u0_hat = solver.warmup(u0_hat, n_warmup=config.n_warmup)
    u0 = jnp.fft.ifft(u0_hat).real

    exponents, _, _ = base_lyapunov.compute_lyapunov_spectrum_jit(
        solver,
        u0,
        n_steps=config.n_steps,
        n_modes=config.n_clv,
    )
    n_positive = int(np.sum(np.asarray(exponents) > 0.0))
    pairs = default_angle_pairs(n_positive)

    clv_comp = base_lyapunov.CLVComputer(solver, n_clv=config.n_clv)
    clv_result = clv_comp.run(u0, n_steps=config.n_steps, key=key)
    angles = clv_comp.compute_clv_angles(clv_result["CLVs"], pairs)

    result = {
        "preset": asdict(preset),
        "config": asdict(config),
        "exponents": np.asarray(exponents),
        "n_positive": n_positive,
        "pairs": pairs,
        "angles": np.asarray(angles),
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / f"{preset.name}_reference_spectrum.npy", result["exponents"])
    np.savez(data_dir / f"{preset.name}_reference_clv_angles.npz", angles=result["angles"], pairs=np.asarray(pairs))
    write_json(
        data_dir / f"{preset.name}_reference_clv_summary.json",
        {
            "preset": asdict(preset),
            "config": asdict(config),
            "lambda_1": float(result["exponents"][0]),
            "n_positive": n_positive,
            "pairs": pairs,
        },
    )
    save_pickle(data_dir / f"{preset.name}_reference_clv.pkl", result)
    return result
