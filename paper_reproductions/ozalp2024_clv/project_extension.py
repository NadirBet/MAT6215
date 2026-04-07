from __future__ import annotations

import glob
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class LinotExtensionConfig:
    n_steps: int = 4000
    n_clv: int = 8
    seed: int = 0
    checkpoint_path: str | None = None
    dt: float = 0.25


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def find_linot_checkpoint(root: Path, explicit: str | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        return path if path.exists() else None
    patterns = [
        root / "paper_reproductions" / "linot2021_ks" / "results" / "*latent*params*.pkl",
        root / "paper_reproductions" / "linot2021_ks" / "results" / "*latent*checkpoint*.pkl",
    ]
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(str(pattern))]
        if matches:
            return matches[0]
    return None


def mlp_forward(params, x):
    for idx, (w, b) in enumerate(params):
        x = x @ jnp.asarray(w) + jnp.asarray(b)
        if idx < len(params) - 1:
            x = jax.nn.sigmoid(x)
    return x


def rk4_step(params, x, dt: float) -> jnp.ndarray:
    rhs = lambda z: mlp_forward(params, z)
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def default_angle_pairs(n_positive: int) -> list[tuple[int, int]]:
    if n_positive >= 3:
        return [(0, 2), (0, 4), (2, 4)]
    if n_positive >= 2:
        return [(0, 1), (0, 2), (1, 2)]
    return [(0, 1)]


def _pair_angles(vectors: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    values = np.zeros((len(pairs),), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        vi = vectors[:, i]
        vj = vectors[:, j]
        denom = np.linalg.norm(vi) * np.linalg.norm(vj) + 1.0e-14
        cos_theta = float(np.clip(np.abs(np.dot(vi, vj)) / denom, 0.0, 1.0))
        values[idx] = np.degrees(np.arccos(cos_theta))
    return values


def run_linot_extension(*, root: Path, config: LinotExtensionConfig, output_dir: Path) -> dict:
    checkpoint = find_linot_checkpoint(root, explicit=config.checkpoint_path)
    if checkpoint is None:
        result = {
            "status": "skipped",
            "reason": (
                "No archived Linot 2021 latent-node checkpoint with parameters was found. "
                "Existing figure3_l22_core artifacts only store trajectories and histories."
            ),
            "config": asdict(config),
        }
        write_json(output_dir / "linot2021_extension_summary.json", result)
        return result

    with checkpoint.open("rb") as f:
        artifact = pickle.load(f)

    if "params" not in artifact:
        result = {
            "status": "skipped",
            "reason": f"Checkpoint {checkpoint.name} does not contain a 'params' entry.",
            "config": asdict(config),
        }
        write_json(output_dir / "linot2021_extension_summary.json", result)
        return result

    params = artifact["params"]
    latent_dim = int(artifact.get("latent_dim", len(artifact["params"][-1][1])))
    n_clv = min(config.n_clv, latent_dim)
    dt = float(artifact.get("dt", config.dt))
    x = jnp.zeros((latent_dim,), dtype=jnp.float64)
    q_frame = np.eye(latent_dim, n_clv, dtype=np.float64)
    q_history = np.zeros((config.n_steps, latent_dim, n_clv), dtype=np.float64)
    r_history = np.zeros((config.n_steps, n_clv, n_clv), dtype=np.float64)
    log_stretches = np.zeros((n_clv,), dtype=np.float64)

    step_fn = lambda state: rk4_step(params, state, dt)

    for step_idx in range(config.n_steps):
        q_raw = np.asarray(
            jax.vmap(lambda q: jax.jvp(step_fn, (x,), (q,))[1], in_axes=1, out_axes=1)(jnp.asarray(q_frame))
        )
        x = step_fn(x)
        q_next, r_upper = np.linalg.qr(q_raw)
        signs = np.sign(np.diag(r_upper))
        signs = np.where(signs == 0.0, 1.0, signs)
        q_next = q_next * signs[None, :]
        r_upper = r_upper * signs[:, None]
        q_history[step_idx] = q_next
        r_history[step_idx] = r_upper
        log_stretches += np.log(np.abs(np.diag(r_upper)) + 1.0e-14)
        q_frame = q_next

    exponents = log_stretches / (config.n_steps * dt)
    n_positive = int(np.sum(exponents > 0.0))
    pairs = default_angle_pairs(n_positive)
    rng = np.random.default_rng(config.seed)
    coeff = rng.normal(size=(n_clv, n_clv))
    coeff = np.triu(coeff)
    coeff = coeff / (np.linalg.norm(coeff, axis=0, keepdims=True) + 1.0e-14)
    angles = np.zeros((config.n_steps, len(pairs)), dtype=np.float64)

    for step_idx in range(config.n_steps - 1, -1, -1):
        coeff = np.linalg.solve(r_history[step_idx], coeff)
        coeff = coeff / (np.linalg.norm(coeff, axis=0, keepdims=True) + 1.0e-14)
        clv_vectors = q_history[step_idx] @ coeff
        angles[step_idx] = _pair_angles(clv_vectors, pairs)

    result = {
        "status": "completed",
        "checkpoint": str(checkpoint),
        "config": asdict(config),
        "lambda_1": float(exponents[0]),
        "n_positive": n_positive,
        "pairs": pairs,
        "exponents": exponents,
        "angles": angles,
    }
    np.save(output_dir / "linot2021_extension_spectrum.npy", exponents)
    np.savez(output_dir / "linot2021_extension_angles.npz", angles=angles, pairs=np.asarray(pairs, dtype=np.int64))
    write_json(
        output_dir / "linot2021_extension_summary.json",
        {
            "status": "completed",
            "checkpoint": str(checkpoint),
            "config": asdict(config),
            "lambda_1": float(exponents[0]),
            "n_positive": n_positive,
            "pairs": pairs,
        },
    )
    save_pickle(output_dir / "linot2021_extension.pkl", result)
    return result
