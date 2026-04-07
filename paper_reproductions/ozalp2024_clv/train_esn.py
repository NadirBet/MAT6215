from __future__ import annotations

import itertools
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from paper_reproductions.ozalp2024_clv.cae import CAEConfig, encode_batch
from paper_reproductions.ozalp2024_clv.esn import (
    ESNConfig,
    fit_readout,
    init_esn,
    validation_rollout_mse,
)


@dataclass(frozen=True)
class ESNSearchConfig:
    spectral_radii: tuple[float, ...] = (0.7, 0.9, 1.1)
    input_scales: tuple[float, ...] = (0.1, 0.25, 0.5)
    leak_rates: tuple[float, ...] = (0.3, 0.6, 1.0)
    ridge_betas: tuple[float, ...] = (1.0e-8, 1.0e-6, 1.0e-4)
    validation_horizon: int = 120
    seed: int = 0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def encode_splits(cae_artifact: dict, cae_config: CAEConfig, train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray):
    params = cae_artifact["params"]
    z_train = np.asarray(encode_batch(params, train_x, cae_config))
    z_val = np.asarray(encode_batch(params, val_x, cae_config))
    z_test = np.asarray(encode_batch(params, test_x, cae_config))
    return z_train, z_val, z_test


def fit_best_esn(
    z_train: np.ndarray,
    z_val: np.ndarray,
    *,
    base_config: ESNConfig,
    search_config: ESNSearchConfig,
    checkpoint_path: Path | None = None,
    history_path: Path | None = None,
) -> dict:
    trials = []
    best = None
    best_score = float("inf")

    for idx, (rho, sigma_in, alpha, beta) in enumerate(
        itertools.product(
            search_config.spectral_radii,
            search_config.input_scales,
            search_config.leak_rates,
            search_config.ridge_betas,
        )
    ):
        cfg = ESNConfig(
            n_input=base_config.n_input,
            n_reservoir=base_config.n_reservoir,
            spectral_radius=rho,
            input_scale=sigma_in,
            leak_rate=alpha,
            ridge_beta=beta,
            recurrent_sparsity=base_config.recurrent_sparsity,
            input_sparsity=base_config.input_sparsity,
            bias_scale=base_config.bias_scale,
            seed=search_config.seed + idx,
        )
        esn = init_esn(cfg)
        esn = fit_readout(esn, z_train)
        val_prefix = z_val[: max(search_config.validation_horizon, 2)]
        val_target = z_val[1:]
        score = validation_rollout_mse(
            esn,
            val_prefix[:-1],
            val_target,
            horizon=min(search_config.validation_horizon, len(val_target)),
        )
        trial = {
            "spectral_radius": rho,
            "input_scale": sigma_in,
            "leak_rate": alpha,
            "ridge_beta": beta,
            "validation_mse": score,
        }
        trials.append(trial)
        if score < best_score:
            best_score = score
            best = {
                "config": asdict(cfg),
                "params": esn,
                "validation_mse": score,
            }

    result = {
        "search_config": asdict(search_config),
        "best_validation_mse": best_score,
        "best": {
            "config": best["config"],
            "validation_mse": best["validation_mse"],
            "params": {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in best["params"].items()},
        },
        "trials": trials,
    }
    if history_path is not None:
        write_json(history_path, {"trials": trials, "best_validation_mse": best_score, "best_config": best["config"]})
    if checkpoint_path is not None:
        save_pickle(checkpoint_path, result)
    return result
