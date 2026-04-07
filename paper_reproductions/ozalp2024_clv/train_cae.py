from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from paper_reproductions.ozalp2024_clv.cae import CAEConfig, init_cae, reconstruct_batch

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class CAETrainingConfig:
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    eval_every: int = 5
    patience_epochs: int = 25
    seed: int = 0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def batched_reconstruction_loss(params, x_data: jnp.ndarray, cae_config: CAEConfig, batch_size: int) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(x_data), batch_size):
        xb = x_data[start:start + batch_size]
        recon = reconstruct_batch(params, xb, cae_config)
        loss = float(jnp.mean((recon - xb) ** 2))
        total += loss * len(xb)
        count += len(xb)
    return total / max(count, 1)


def train_cae(
    train_x: np.ndarray,
    val_x: np.ndarray,
    *,
    cae_config: CAEConfig,
    train_config: CAETrainingConfig,
    checkpoint_path: Path | None = None,
    history_path: Path | None = None,
) -> dict:
    params = init_cae(jax.random.PRNGKey(train_config.seed), cae_config)
    optimizer = optax.adam(train_config.learning_rate)
    opt_state = optimizer.init(params)

    x_train_j = jnp.asarray(train_x)
    x_val_j = jnp.asarray(val_x)
    rng = np.random.default_rng(train_config.seed)

    @jax.jit
    def batch_loss_fn(params, xb):
        recon = reconstruct_batch(params, xb, cae_config)
        return jnp.mean((recon - xb) ** 2)

    @jax.jit
    def train_step(params, opt_state, xb):
        loss, grads = jax.value_and_grad(batch_loss_fn)(params, xb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    best_val = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(train_config.epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(train_x), train_config.batch_size):
            idx = order[start:start + train_config.batch_size]
            params, opt_state, _ = train_step(params, opt_state, x_train_j[idx])

        should_eval = epoch == 0 or (epoch + 1) % train_config.eval_every == 0 or epoch + 1 == train_config.epochs
        if not should_eval:
            continue

        train_loss = batched_reconstruction_loss(params, x_train_j, cae_config, train_config.batch_size)
        val_loss = batched_reconstruction_loss(params, x_val_j, cae_config, train_config.batch_size)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
            stale = 0
        else:
            stale += train_config.eval_every

        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val
        if history_path is not None:
            write_json(history_path, history)
        if stale >= train_config.patience_epochs:
            break

    result = {
        "cae_config": asdict(cae_config),
        "train_config": asdict(train_config),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x), best_params),
        "history": history,
    }
    if checkpoint_path is not None:
        save_pickle(checkpoint_path, result)
    return result


def latent_dimension_sweep(
    train_x: np.ndarray,
    val_x: np.ndarray,
    *,
    n_grid: int,
    latent_dims: list[int],
    base_train_config: CAETrainingConfig,
) -> dict:
    curves = {"latent_dim": [], "val_mse": []}
    for idx, latent_dim in enumerate(latent_dims):
        cae_config = CAEConfig(n_grid=n_grid, latent_dim=latent_dim)
        result = train_cae(
            train_x,
            val_x,
            cae_config=cae_config,
            train_config=CAETrainingConfig(**{**asdict(base_train_config), "seed": base_train_config.seed + idx}),
        )
        curves["latent_dim"].append(latent_dim)
        curves["val_mse"].append(float(result["best_val_loss"]))
    return curves
