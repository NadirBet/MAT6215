from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from paper_reproductions.ozalp2024_clv.data import OzalpPreset
from paper_reproductions.ozalp2024_clv.esn import (
    closed_loop_step,
    esn_output_tangent,
    esn_tangent_step,
    warm_start_state,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class LatentCLVConfig:
    n_steps: int = 4000
    n_clv: int = 12
    warmup_steps: int = 400
    seed: int = 0
    angle_space: str = "latent"
    store_q_dtype: str = "float32"


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


def _pair_angles(vectors: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    angles = np.zeros((len(pairs),), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        vi = vectors[:, i]
        vj = vectors[:, j]
        denom = np.linalg.norm(vi) * np.linalg.norm(vj) + 1.0e-14
        cos_theta = float(np.clip(np.abs(np.dot(vi, vj)) / denom, 0.0, 1.0))
        angles[idx] = np.degrees(np.arccos(cos_theta))
    return angles


def _memmap_paths(data_dir: Path, preset: OzalpPreset) -> dict[str, Path]:
    prefix = f"{preset.name}_esn_clv"
    return {
        "q_history": data_dir / f"{prefix}_q_history.dat",
        "r_history": data_dir / f"{prefix}_r_history.dat",
    }


def _as_esn_params(esn_artifact: dict) -> dict:
    params = esn_artifact["best"]["params"] if "best" in esn_artifact else esn_artifact["params"]
    return {
        "config": params["config"],
        "W": jnp.asarray(params["W"]),
        "W_in": jnp.asarray(params["W_in"]),
        "bias": jnp.asarray(params["bias"]),
        "W_out": jnp.asarray(params["W_out"]),
    }


def compute_esn_clv(
    preset: OzalpPreset,
    *,
    esn_artifact: dict,
    z_reference: np.ndarray,
    config: LatentCLVConfig,
    data_dir: Path,
    pairs: list[tuple[int, int]] | None = None,
    keep_qr_cache: bool = False,
) -> dict:
    esn = _as_esn_params(esn_artifact)
    n_res = int(esn["W"].shape[0])
    n_clv = min(config.n_clv, n_res)

    if len(z_reference) < max(config.warmup_steps, 2):
        raise ValueError("Need at least warmup_steps latent states for ESN warm start.")

    data_dir.mkdir(parents=True, exist_ok=True)
    mm_paths = _memmap_paths(data_dir, preset)
    q_dtype = np.float32 if config.store_q_dtype == "float32" else np.float64
    q_history = np.memmap(
        mm_paths["q_history"],
        dtype=q_dtype,
        mode="w+",
        shape=(config.n_steps, n_res, n_clv),
    )
    r_history = np.memmap(
        mm_paths["r_history"],
        dtype=np.float64,
        mode="w+",
        shape=(config.n_steps, n_clv, n_clv),
    )

    r = np.asarray(warm_start_state(esn, z_reference[: config.warmup_steps]))
    q_frame = np.eye(n_res, n_clv, dtype=np.float64)
    log_stretches = np.zeros((n_clv,), dtype=np.float64)

    for step_idx in range(config.n_steps):
        q_raw = np.asarray(esn_tangent_step(esn, jnp.asarray(r), jnp.asarray(q_frame)))
        r = np.asarray(closed_loop_step(esn, jnp.asarray(r)))
        q_next, r_upper = np.linalg.qr(q_raw)
        signs = np.sign(np.diag(r_upper))
        signs = np.where(signs == 0.0, 1.0, signs)
        q_next = q_next * signs[None, :]
        r_upper = r_upper * signs[:, None]
        q_history[step_idx] = q_next.astype(q_dtype)
        r_history[step_idx] = r_upper
        log_stretches += np.log(np.abs(np.diag(r_upper)) + 1.0e-14)
        q_frame = q_next

    q_history.flush()
    r_history.flush()

    exponents = log_stretches / (config.n_steps * preset.dt)
    n_positive = int(np.sum(exponents > 0.0))
    pairs = default_angle_pairs(n_positive) if pairs is None else pairs

    rng = np.random.default_rng(config.seed)
    coeff = rng.normal(size=(n_clv, n_clv))
    coeff = np.triu(coeff)
    coeff = coeff / (np.linalg.norm(coeff, axis=0, keepdims=True) + 1.0e-14)

    angle_series = np.zeros((config.n_steps, len(pairs)), dtype=np.float64)
    for step_idx in range(config.n_steps - 1, -1, -1):
        r_upper = np.asarray(r_history[step_idx], dtype=np.float64)
        coeff = np.linalg.solve(r_upper, coeff)
        coeff = coeff / (np.linalg.norm(coeff, axis=0, keepdims=True) + 1.0e-14)
        q_step = np.asarray(q_history[step_idx], dtype=np.float64)
        clv_vectors = q_step @ coeff
        if config.angle_space == "latent":
            angle_basis = np.asarray(esn_output_tangent(esn, jnp.asarray(clv_vectors)))
        elif config.angle_space == "reservoir":
            angle_basis = clv_vectors
        else:
            raise ValueError(f"Unsupported angle_space: {config.angle_space}")
        angle_series[step_idx] = _pair_angles(angle_basis, pairs)

    result = {
        "preset": asdict(preset),
        "config": asdict(config),
        "exponents": exponents,
        "lambda_1": float(exponents[0]),
        "n_positive": n_positive,
        "pairs": pairs,
        "angles": angle_series,
        "angle_space": config.angle_space,
    }

    np.save(data_dir / f"{preset.name}_esn_spectrum.npy", exponents)
    np.savez(
        data_dir / f"{preset.name}_esn_clv_angles.npz",
        angles=angle_series,
        pairs=np.asarray(pairs, dtype=np.int64),
    )
    write_json(
        data_dir / f"{preset.name}_esn_clv_summary.json",
        {
            "preset": asdict(preset),
            "config": asdict(config),
            "lambda_1": float(exponents[0]),
            "n_positive": n_positive,
            "pairs": pairs,
            "angle_space": config.angle_space,
        },
    )
    save_pickle(data_dir / f"{preset.name}_esn_clv.pkl", result)

    if not keep_qr_cache:
        for path in mm_paths.values():
            if path.exists():
                path.unlink()

    return result
