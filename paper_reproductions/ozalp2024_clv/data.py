from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from ks_solver import KSSolver


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = ROOT / "paper_reproductions" / "ozalp2024_clv"
DATA_DIR = THIS_DIR / "data"
FIG_DIR = THIS_DIR / "figures"
LINOT2021_DATA_DIR = ROOT / "paper_reproductions" / "linot2021_ks" / "data"


@dataclass(frozen=True)
class OzalpPreset:
    name: str
    domain_length: float
    state_dim: int
    dt: float
    n_train: int
    n_val: int
    n_test: int
    latent_dim: int
    reservoir_dim: int
    source: str


PAPER_PRESET = OzalpPreset(
    name="paper",
    domain_length=float(2.0 * np.pi * 10.0),
    state_dim=128,
    dt=0.25,
    n_train=50000,
    n_val=20000,
    n_test=50000,
    latent_dim=24,
    reservoir_dim=5000,
    source="generate",
)

PROJECT_PRESET = OzalpPreset(
    name="project",
    domain_length=22.0,
    state_dim=64,
    dt=0.25,
    n_train=16000,
    n_val=4000,
    n_test=8000,
    latent_dim=10,
    reservoir_dim=500,
    source="linot2021_cache",
)


def get_preset(name: str) -> OzalpPreset:
    if name == "paper":
        return PAPER_PRESET
    if name == "project":
        return PROJECT_PRESET
    raise ValueError(f"Unknown Ozalp preset: {name}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def dataset_paths(preset: OzalpPreset) -> dict[str, Path]:
    prefix = f"ozalp_{preset.name}_l{int(round(preset.domain_length))}_n{preset.state_dim}"
    return {
        "train": DATA_DIR / f"{prefix}_train.npy",
        "val": DATA_DIR / f"{prefix}_val.npy",
        "test": DATA_DIR / f"{prefix}_test.npy",
        "norm": DATA_DIR / f"{prefix}_normalization.json",
        "meta": DATA_DIR / f"{prefix}_meta.json",
    }


def linot2021_paths() -> dict[str, Path]:
    prefix = "ks_l22_n64_dt025"
    return {
        "train": LINOT2021_DATA_DIR / f"{prefix}_train.npy",
        "test": LINOT2021_DATA_DIR / f"{prefix}_test.npy",
        "meta": LINOT2021_DATA_DIR / f"{prefix}_meta.json",
    }


def load_cached_dataset(preset: OzalpPreset) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None:
    paths = dataset_paths(preset)
    if not all(path.exists() for path in paths.values()):
        return None
    train = np.load(paths["train"])
    val = np.load(paths["val"])
    test = np.load(paths["test"])
    with paths["meta"].open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return train, val, test, meta


def save_dataset(
    preset: OzalpPreset,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    meta: dict,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(preset)
    np.save(paths["train"], train)
    np.save(paths["val"], val)
    np.save(paths["test"], test)
    write_json(paths["norm"], {"mean": mean.tolist(), "std": std.tolist()})
    write_json(paths["meta"], meta)


def normalize_splits(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std


def generate_contiguous_trajectory(
    *,
    L: float,
    N: int,
    dt: float,
    seed: int,
    n_warmup: int,
    n_steps: int,
) -> np.ndarray:
    solver = KSSolver(L=L, N=N, dt=dt)
    key = jax.random.PRNGKey(seed)
    u0_hat = solver.random_ic(key)
    u0_hat = solver.warmup(u0_hat, n_warmup=n_warmup)
    return np.asarray(solver.integrate(u0_hat, n_steps=n_steps))


def ensure_dataset(
    preset: OzalpPreset,
    *,
    allow_generate: bool = False,
    seed: int = 0,
    n_warmup: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cached = load_cached_dataset(preset)
    if cached is not None:
        return cached

    if preset.source == "linot2021_cache":
        paths = linot2021_paths()
        if not all(path.exists() for path in paths.values()):
            raise FileNotFoundError("Missing linot2021_ks cached L=22 dataset.")
        train_src = np.load(paths["train"])
        test_src = np.load(paths["test"])
        full = np.concatenate([train_src, test_src], axis=0)
        train = full[:preset.n_train]
        val = full[preset.n_train:preset.n_train + preset.n_val]
        test = full[preset.n_train + preset.n_val:preset.n_train + preset.n_val + preset.n_test]
        train_n, val_n, test_n, mean, std = normalize_splits(train, val, test)
        meta = {
            "preset": asdict(preset),
            "source": preset.source,
            "seed": seed,
            "note": "Re-split contiguous linot2021_ks cached trajectory into train/val/test.",
            "train_shape": list(train_n.shape),
            "val_shape": list(val_n.shape),
            "test_shape": list(test_n.shape),
        }
        save_dataset(preset, train_n, val_n, test_n, mean, std, meta)
        return train_n, val_n, test_n, meta

    if not allow_generate:
        raise FileNotFoundError(
            f"Missing cached dataset for preset '{preset.name}'. Call with allow_generate=True to build it."
        )

    n_total = preset.n_train + preset.n_val + preset.n_test
    full = generate_contiguous_trajectory(
        L=preset.domain_length,
        N=preset.state_dim,
        dt=preset.dt,
        seed=seed,
        n_warmup=n_warmup,
        n_steps=n_total,
    )
    train = full[:preset.n_train]
    val = full[preset.n_train:preset.n_train + preset.n_val]
    test = full[preset.n_train + preset.n_val:preset.n_train + preset.n_val + preset.n_test]
    train_n, val_n, test_n, mean, std = normalize_splits(train, val, test)
    meta = {
        "preset": asdict(preset),
        "source": "generated",
        "seed": seed,
        "n_warmup": n_warmup,
        "train_shape": list(train_n.shape),
        "val_shape": list(val_n.shape),
        "test_shape": list(test_n.shape),
    }
    save_dataset(preset, train_n, val_n, test_n, mean, std, meta)
    return train_n, val_n, test_n, meta


def one_step_pairs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return states[:-1], states[1:]
