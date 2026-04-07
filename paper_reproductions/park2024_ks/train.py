"""One-step Neural-ODE training for the Park 2024 KS reproduction."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .model import flow_map_forward, flow_map_jacobian

jax.config.update("jax_enable_x64", True)


def mse_map_loss(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray, dt: float) -> jnp.ndarray:
    y_pred = jax.vmap(lambda x: flow_map_forward(params, x, dt))(x_batch)
    # stacNODE's KS MSE branch scales one-step MSE by 1 / dt.
    return jnp.mean((y_pred - y_batch) ** 2) / max(float(dt), 1e-12)


def jac_map_loss(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray, jac_batch: jnp.ndarray, lam: float, dt: float) -> jnp.ndarray:
    # Exact formula from stacNODE test_KS.py:
    #   total_loss = reg_param * ||True_J - cur_model_J||_F  +  (1/dt^2) * MSE_loss
    # The Frobenius norm is over the full batch tensor (not averaged per sample).
    y_pred = jax.vmap(lambda x: flow_map_forward(params, x, dt))(x_batch)
    jac_pred = jax.vmap(lambda x: flow_map_jacobian(params, x, dt))(x_batch)
    mse_term = jnp.mean((y_pred - y_batch) ** 2) / max(float(dt) ** 2, 1e-12)
    jac_diff = jac_batch - jac_pred          # (B, d, d)
    jac_term = jnp.linalg.norm(jac_diff)    # Frobenius over full batch tensor
    return lam * jac_term + mse_term


def mean_relative_error(params: dict, x_eval: np.ndarray, y_eval: np.ndarray, dt: float) -> float:
    x_eval_j = jnp.array(x_eval, dtype=jnp.float64)
    y_eval_j = jnp.array(y_eval, dtype=jnp.float64)
    pred = jax.vmap(lambda x: flow_map_forward(params, x, dt))(x_eval_j)
    numer = jnp.linalg.norm(pred - y_eval_j, axis=1)
    denom = jnp.linalg.norm(y_eval_j, axis=1) + 1e-12
    return float(jnp.mean(numer / denom))


def evaluate_in_chunks(
    params: dict,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    dt: float,
    loss_mode: str,
    lam: float,
    jac_eval: np.ndarray | None = None,
    chunk_size: int = 64,
) -> tuple[float, float]:
    """
    Evaluate loss and mean relative error without materializing the whole test
    Jacobian tensor on device at once.
    """
    n_eval = len(x_eval)
    relerr_weighted = 0.0

    if loss_mode == "mse":
        total_sqerr = 0.0
        total_count = 0
        for start in range(0, n_eval, chunk_size):
            stop = min(start + chunk_size, n_eval)
            x_chunk = jnp.array(x_eval[start:stop], dtype=jnp.float64)
            y_chunk = jnp.array(y_eval[start:stop], dtype=jnp.float64)
            pred_chunk = jax.vmap(lambda x: flow_map_forward(params, x, dt))(x_chunk)
            sqerr = np.array((pred_chunk - y_chunk) ** 2)
            total_sqerr += float(np.sum(sqerr))
            total_count += int(np.prod(sqerr.shape))

            numer = np.linalg.norm(np.array(pred_chunk - y_chunk), axis=1)
            denom = np.linalg.norm(np.array(y_chunk), axis=1) + 1e-12
            relerr_weighted += float(np.sum(numer / denom))

        mse_only = total_sqerr / max(total_count, 1)
        mse_only = mse_only / max(float(dt), 1e-12)
        return mse_only, relerr_weighted / max(n_eval, 1), mse_only

    if jac_eval is None:
        raise ValueError("jac_eval is required for chunked Jacobian evaluation")

    total_sqerr = 0.0
    total_y_count = 0
    total_jac_sq = 0.0
    for start in range(0, n_eval, chunk_size):
        stop = min(start + chunk_size, n_eval)
        x_chunk = jnp.array(x_eval[start:stop], dtype=jnp.float64)
        y_chunk = jnp.array(y_eval[start:stop], dtype=jnp.float64)
        jac_chunk = jnp.array(jac_eval[start:stop], dtype=jnp.float64)

        pred_chunk = jax.vmap(lambda x: flow_map_forward(params, x, dt))(x_chunk)
        jac_pred_chunk = jax.vmap(lambda x: flow_map_jacobian(params, x, dt))(x_chunk)

        sqerr = np.array((pred_chunk - y_chunk) ** 2)
        total_sqerr += float(np.sum(sqerr))
        total_y_count += int(np.prod(sqerr.shape))

        jac_diff = np.array(jac_chunk - jac_pred_chunk)
        total_jac_sq += float(np.sum(jac_diff ** 2))

        numer = np.linalg.norm(np.array(pred_chunk - y_chunk), axis=1)
        denom = np.linalg.norm(np.array(y_chunk), axis=1) + 1e-12
        relerr_weighted += float(np.sum(numer / denom))

    mse_term = total_sqerr / max(total_y_count, 1)
    jac_term = float(np.sqrt(total_jac_sq))
    # Keep JAC's composite loss faithful to stacNODE while also reporting a
    # comparable MSE-only metric for Figure-3-style diagnostics.
    mse_only = mse_term / max(float(dt), 1e-12)
    total_loss = mse_term / max(float(dt) ** 2, 1e-12) + lam * jac_term
    return total_loss, relerr_weighted / max(n_eval, 1), mse_only


def train_flow_map(
    init_params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    x_test: np.ndarray,
    y_test: np.ndarray,
    jac_train: np.ndarray | None = None,
    jac_test: np.ndarray | None = None,
    loss_mode: str = "mse",
    lam: float = 1.0,
    dt: float = 0.25,
    epochs: int = 3000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    lr_schedule: str = "exp_decay",
    select_best: bool = True,
    grad_clip: float | None = None,
    seed: int = 0,
    eval_every: int = 50,
    eval_chunk_size: int = 64,
    progress_callback=None,
):
    """
    Train a one-step flow-map model with MSE or JAC loss.

    Returns:
        selected_params, history
    """
    if loss_mode not in {"mse", "jac"}:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    if loss_mode == "jac" and jac_train is None:
        raise ValueError("jac_train is required for loss_mode='jac'")
    if lr_schedule not in {"exp_decay", "constant"}:
        raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")

    x_train_np = np.asarray(x_train, dtype=np.float64)
    y_train_np = np.asarray(y_train, dtype=np.float64)
    x_test_np = np.asarray(x_test, dtype=np.float64)
    y_test_np = np.asarray(y_test, dtype=np.float64)
    jac_train_np = None if jac_train is None else np.asarray(jac_train, dtype=np.float64)
    jac_test_np = None if jac_test is None else np.asarray(jac_test, dtype=np.float64)

    n_train = len(x_train_np)
    effective_batch_size = n_train if batch_size >= n_train else max(batch_size, 1)
    n_batches_per_epoch = max(int(np.ceil(n_train / max(effective_batch_size, 1))), 1)
    if lr_schedule == "constant":
        base_optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    else:
        base_optimizer = optax.adamw(
            optax.exponential_decay(
                init_value=learning_rate,
                transition_steps=n_batches_per_epoch,
                decay_rate=0.1 ** (1.0 / max(epochs, 1)),
                staircase=False,
            ),
            weight_decay=weight_decay,
        )
    if grad_clip is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(float(grad_clip)), base_optimizer)
    else:
        optimizer = base_optimizer
    opt_state = optimizer.init(init_params)

    @jax.jit
    def mse_step(params, opt_state, x_batch, y_batch):
        loss_fn = lambda p: mse_map_loss(p, x_batch, y_batch, dt)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def jac_step(params, opt_state, x_batch, y_batch, jac_batch):
        loss_fn = lambda p: jac_map_loss(p, x_batch, y_batch, jac_batch, lam, dt)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng = np.random.default_rng(seed)
    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "train_mse_only": [],
        "test_mse_only": [],
        "test_relative_error": [],
        "best_epoch": None,
        "best_test_loss": float("inf"),
        "selection_mode": "best_test_loss" if select_best else "final_epoch",
        "grad_clip": grad_clip,
    }

    params = init_params
    best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)

    for epoch in range(epochs):
        order = rng.permutation(n_train)
        batch_losses = []
        for start in range(0, n_train, effective_batch_size):
            batch_idx = order[start:start + effective_batch_size]
            x_batch = jnp.array(x_train_np[batch_idx], dtype=jnp.float64)
            y_batch = jnp.array(y_train_np[batch_idx], dtype=jnp.float64)
            if loss_mode == "mse":
                params, opt_state, loss = mse_step(params, opt_state, x_batch, y_batch)
            else:
                jac_batch = jnp.array(jac_train_np[batch_idx], dtype=jnp.float64)
                params, opt_state, loss = jac_step(params, opt_state, x_batch, y_batch, jac_batch)
            batch_losses.append(float(loss))

        should_eval = (epoch == 0) or ((epoch + 1) % eval_every == 0) or (epoch + 1 == epochs)
        if should_eval:
            test_loss, relerr, test_mse_only = evaluate_in_chunks(
                params,
                x_test_np,
                y_test_np,
                dt=dt,
                loss_mode=loss_mode,
                lam=lam,
                jac_eval=jac_test_np,
                chunk_size=eval_chunk_size,
            )
            train_loss_eval, _, train_mse_only = evaluate_in_chunks(
                params,
                x_train_np,
                y_train_np,
                dt=dt,
                loss_mode=loss_mode,
                lam=lam,
                jac_eval=jac_train_np,
                chunk_size=eval_chunk_size,
            )

            history["epochs"].append(epoch + 1)
            history["train_loss"].append(train_loss_eval)
            history["test_loss"].append(test_loss)
            history["train_mse_only"].append(train_mse_only)
            history["test_mse_only"].append(test_mse_only)
            history["test_relative_error"].append(relerr)

            improved = False
            if test_loss < history["best_test_loss"]:
                history["best_test_loss"] = test_loss
                history["best_epoch"] = epoch + 1
                best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
                improved = True

            if progress_callback is not None:
                progress_callback(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss_eval,
                        "test_loss": test_loss,
                        "train_mse_only": train_mse_only,
                        "test_mse_only": test_mse_only,
                        "test_relative_error": relerr,
                        "best_epoch": history["best_epoch"],
                        "best_test_loss": history["best_test_loss"],
                        "selected_epoch": history["best_epoch"] if select_best else epoch + 1,
                        "improved": improved,
                    }
                )

    history["selected_epoch"] = history["best_epoch"] if select_best else epochs
    selected_params = best_params if select_best else jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    return selected_params, history
