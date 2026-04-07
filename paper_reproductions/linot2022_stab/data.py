from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from ks_solver import KSSolver, generate_training_data


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = ROOT / "paper_reproductions" / "linot2022_stab"
DATA_DIR = THIS_DIR / "data"
LINOT2021_DATA_DIR = ROOT / "paper_reproductions" / "linot2021_ks" / "data"

LOCAL_TRAIN = DATA_DIR / "linot2022_ks_train.npy"
LOCAL_TEST = DATA_DIR / "linot2022_ks_test.npy"
LOCAL_META = DATA_DIR / "linot2022_ks_meta.json"


@dataclass(frozen=True)
class DatasetConfig:
    L: float = 22.0
    N: int = 64
    dt: float = 0.25
    seed: int = 0
    n_warmup: int = 2000
    n_train: int = 20000
    n_test: int = 8000
    prefer_linot2021_cache: bool = True
    allow_generate: bool = False


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def linot2021_cached_paths(L: float, N: int, dt: float) -> dict[str, Path]:
    prefix = f"ks_l{int(round(L))}_n{N}_dt{str(dt).replace('.', '')}"
    return {
        "train": LINOT2021_DATA_DIR / f"{prefix}_train.npy",
        "test": LINOT2021_DATA_DIR / f"{prefix}_test.npy",
        "meta": LINOT2021_DATA_DIR / f"{prefix}_meta.json",
    }


def load_local_dataset() -> tuple[np.ndarray, np.ndarray, dict] | None:
    if not (LOCAL_TRAIN.exists() and LOCAL_TEST.exists() and LOCAL_META.exists()):
        return None
    train = np.load(LOCAL_TRAIN)
    test = np.load(LOCAL_TEST)
    with LOCAL_META.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return train, test, meta


def save_local_dataset(train: np.ndarray, test: np.ndarray, meta: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(LOCAL_TRAIN, train)
    np.save(LOCAL_TEST, test)
    write_json(LOCAL_META, meta)


def ensure_dataset(config: DatasetConfig = DatasetConfig()) -> tuple[np.ndarray, np.ndarray, dict]:
    cached = load_local_dataset()
    if cached is not None:
        return cached

    source = None
    if config.prefer_linot2021_cache:
        paths = linot2021_cached_paths(config.L, config.N, config.dt)
        if all(path.exists() for path in paths.values()):
            train = np.load(paths["train"])
            test = np.load(paths["test"])
            with paths["meta"].open("r", encoding="utf-8") as f:
                upstream_meta = json.load(f)
            meta = {
                "source": "linot2021_cache",
                "linot2021_paths": {name: str(path) for name, path in paths.items()},
                "dataset_config": asdict(config),
                "upstream_meta": upstream_meta,
                "train_shape": list(train.shape),
                "test_shape": list(test.shape),
            }
            save_local_dataset(train, test, meta)
            return train, test, meta
        source = "generated"

    if not config.allow_generate:
        raise FileNotFoundError(
            "Missing local Linot 2022 dataset and no reusable Linot 2021 cache was found. "
            "Call with allow_generate=True to build a fresh KSE trajectory."
        )

    solver = KSSolver(L=config.L, N=config.N, dt=config.dt)
    key = jax.random.PRNGKey(config.seed)
    data = generate_training_data(
        solver,
        key,
        n_warmup=config.n_warmup,
        n_train=config.n_train,
        n_test=config.n_test,
    )
    train = np.asarray(data["u_train"])
    test = np.asarray(data["u_test"])
    meta = {
        "source": source or "generated",
        "dataset_config": asdict(config),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
    }
    save_local_dataset(train, test, meta)
    return train, test, meta


def make_onestep_pairs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("Expected a time-ordered state array of shape (T, N) with T >= 2.")
    return states[:-1], states[1:]


def sample_rollout_start_indices(
    states: np.ndarray,
    *,
    n_windows: int,
    window_steps: int,
) -> np.ndarray:
    max_start = len(states) - window_steps - 1
    if max_start < 0:
        raise ValueError("Not enough states for the requested rollout window.")
    if n_windows <= 1:
        return np.array([0], dtype=int)
    return np.linspace(0, max_start, n_windows, dtype=int)


def add_band_limited_noise(
    u: np.ndarray,
    *,
    epsilon: float,
    mode_lo: int = 20,
    mode_hi: int = 31,
    seed: int = 0,
) -> np.ndarray:
    noisy_hat = np.fft.rfft(np.asarray(u)).astype(np.complex128)
    rng = np.random.default_rng(seed)
    start = max(mode_lo, 0)
    stop = min(mode_hi + 1, noisy_hat.shape[0])
    if start < stop and epsilon > 0.0:
        noise_real = rng.normal(scale=epsilon, size=stop - start)
        noise_imag = rng.normal(scale=epsilon, size=stop - start)
        noisy_hat[start:stop] += noise_real + 1j * noise_imag
    return np.fft.irfft(noisy_hat, n=u.shape[-1])
