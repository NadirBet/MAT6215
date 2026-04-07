from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import numpy as np

from ks_solver import KSSolver, generate_training_data


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_reproductions" / "linot2021_ks"
DATA_DIR = OUT_DIR / "data"
RES_DIR = OUT_DIR / "results"


def linot_data_prefix(L: float) -> str:
    return f"ks_l{int(round(L))}_n64_dt025"


def linot_data_paths(L: float) -> dict[str, Path]:
    prefix = linot_data_prefix(L)
    return {
        "train": DATA_DIR / f"{prefix}_train.npy",
        "test": DATA_DIR / f"{prefix}_test.npy",
        "meta": DATA_DIR / f"{prefix}_meta.json",
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_cached_dataset(L: float) -> tuple[np.ndarray, np.ndarray, dict] | None:
    paths = linot_data_paths(L)
    if not all(path.exists() for path in paths.values()):
        return None
    x_train = np.load(paths["train"])
    x_test = np.load(paths["test"])
    with paths["meta"].open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return x_train, x_test, meta


def save_dataset(L: float, x_train: np.ndarray, x_test: np.ndarray, meta: dict) -> dict[str, Path]:
    paths = linot_data_paths(L)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(paths["train"], x_train)
    np.save(paths["test"], x_test)
    write_json(paths["meta"], meta)
    return paths


def ensure_dataset(
    L: float,
    *,
    N: int = 64,
    dt: float = 0.25,
    seed: int = 0,
    n_warmup: int = 2000,
    n_train: int = 20000,
    n_test: int = 8000,
    allow_generate: bool = False,
    progress_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cached = load_cached_dataset(L)
    if cached is not None:
        return cached

    if not allow_generate:
        raise FileNotFoundError(
            f"Missing cached Linot dataset for L={L:.0f}. "
            f"Run the dataset-prep script first or call with allow_generate=True."
        )

    if progress_path is not None:
        write_json(
            progress_path,
            {
                "status": "running",
                "stage": "generating_dataset",
                "L": L,
                "N": N,
                "dt": dt,
                "seed": seed,
                "n_warmup": n_warmup,
                "n_train": n_train,
                "n_test": n_test,
                "time": time.time(),
            },
        )

    solver = KSSolver(L=L, N=N, dt=dt)
    key = jax.random.PRNGKey(seed)
    data = generate_training_data(
        solver,
        key,
        n_warmup=n_warmup,
        n_train=n_train,
        n_test=n_test,
    )
    x_train = np.asarray(data["u_train"])
    x_test = np.asarray(data["u_test"])
    meta = {
        "domain_length": float(L),
        "state_dim": int(N),
        "dt": float(dt),
        "seed": int(seed),
        "n_warmup": int(n_warmup),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }
    save_dataset(L, x_train, x_test, meta)

    if progress_path is not None:
        write_json(
            progress_path,
            {
                "status": "completed",
                "stage": "generating_dataset",
                "L": L,
                "train_shape": list(x_train.shape),
                "test_shape": list(x_test.shape),
                "time": time.time(),
            },
        )

    return x_train, x_test, meta
