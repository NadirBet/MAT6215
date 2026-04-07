from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from paper_reproductions.linot2022_stab.models import ModelBundle

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3000
    batch_size: int = 256
    eval_batch_size: int = 256
    learning_rate: float = 1.0e-3
    decay_boundaries: tuple[float, ...] = (0.5,)
    decay_scales: tuple[float, ...] = (0.1,)
    eval_every: int = 10
    patience_epochs: Optional[int] = 10
    seed: int = 0


def _to_serializable_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def save_history_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def make_schedule(config: TrainingConfig, steps_per_epoch: int):
    boundaries = {}
    for frac, scale in zip(config.decay_boundaries, config.decay_scales):
        step = max(int(frac * config.epochs * steps_per_epoch), 1)
        boundaries[step] = scale
    return optax.piecewise_constant_schedule(
        init_value=config.learning_rate,
        boundaries_and_scales=boundaries,
    )


def batched_dataset_loss(
    loss_fn,
    params,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    *,
    batch_size: int,
) -> float:
    total_loss = 0.0
    total_count = 0
    n_samples = len(x_data)
    for start in range(0, n_samples, batch_size):
        xb = x_data[start:start + batch_size]
        yb = y_data[start:start + batch_size]
        batch_loss = float(loss_fn(params, xb, yb))
        batch_count = len(xb)
        total_loss += batch_loss * batch_count
        total_count += batch_count
    return total_loss / max(total_count, 1)


def train_model(
    bundle: ModelBundle,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    config: TrainingConfig,
    checkpoint_path: Path | None = None,
    history_json_path: Path | None = None,
    progress_json_path: Path | None = None,
) -> dict:
    n_train = len(train_x)
    steps_per_epoch = max(math.ceil(n_train / config.batch_size), 1)
    schedule = make_schedule(config, steps_per_epoch)
    optimizer = optax.adam(schedule)

    key = jax.random.PRNGKey(config.seed)
    params = bundle.init_params(key)
    opt_state = optimizer.init(params)

    x_train_j = jnp.asarray(train_x)
    y_train_j = jnp.asarray(train_y)
    x_test_j = jnp.asarray(test_x)
    y_test_j = jnp.asarray(test_y)
    rng = np.random.default_rng(config.seed)

    @jax.jit
    def loss_fn(params, xb, yb):
        pred = jax.vmap(lambda x: bundle.step(params, x))(xb)
        return jnp.mean(jnp.abs(pred - yb))

    @jax.jit
    def train_step(params, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    history = {
        "model_name": bundle.name,
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "config": asdict(config),
    }
    best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    best_test_loss = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(config.epochs):
        order = rng.permutation(n_train)
        for start in range(0, n_train, config.batch_size):
            idx = order[start:start + config.batch_size]
            params, opt_state, _ = train_step(params, opt_state, x_train_j[idx], y_train_j[idx])

        should_eval = (
            epoch == 0
            or (epoch + 1) % config.eval_every == 0
            or epoch + 1 == config.epochs
        )
        if not should_eval:
            continue

        train_loss = batched_dataset_loss(
            loss_fn,
            params,
            x_train_j,
            y_train_j,
            batch_size=config.eval_batch_size,
        )
        test_loss = batched_dataset_loss(
            loss_fn,
            params,
            x_test_j,
            y_test_j,
            batch_size=config.eval_batch_size,
        )
        improved = test_loss < best_test_loss
        if improved:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += config.eval_every

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["best_epoch"] = best_epoch
        history["best_test_loss"] = best_test_loss

        if history_json_path is not None:
            save_history_json(history_json_path, history)
        if progress_json_path is not None:
            save_history_json(
                progress_json_path,
                {
                    "status": "running",
                    "model": bundle.name,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "best_epoch": best_epoch,
                    "best_test_loss": best_test_loss,
                },
            )
        if config.patience_epochs is not None and epochs_since_improvement >= config.patience_epochs:
            history["stopped_early"] = True
            history["stop_epoch"] = epoch + 1
            break
    else:
        history["stopped_early"] = False
        history["stop_epoch"] = config.epochs

    checkpoint = {
        "model_name": bundle.name,
        "display_name": bundle.display_name,
        "config": asdict(config),
        "history": history,
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss,
        "params": _to_serializable_tree(best_params),
    }
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, checkpoint)
    if progress_json_path is not None:
        save_history_json(
            progress_json_path,
            {
                "status": "completed",
                "model": bundle.name,
                "best_epoch": best_epoch,
                "best_test_loss": best_test_loss,
                "stop_epoch": history["stop_epoch"],
            },
        )
    return checkpoint
