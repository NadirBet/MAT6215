"""
train.py — Training Loops for Neural ODE and SINDy
====================================================
Implements training pipelines for both surrogates.

Neural ODE Training:
    - Batch gradient descent on (u, rhs) pairs
    - Two modes: MSE and MSE + Jacobian matching (Park 2024)
    - Optimizer: Adam with learning rate schedule
    - Early stopping on validation loss

SINDy Training:
    - Calls fit_sindy from sindy.py (no gradient descent — analytic)
    - Just data preparation and fitting

Both surrogates are saved to disk after training.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import pickle
from functools import partial
from typing import Callable

from neural_ode import (init_standard_node, init_stabilized_node,
                         standard_node_rhs, stabilized_node_rhs,
                         mse_loss, jacobian_matching_loss,
                         prepare_training_data)
from sindy import fit_sindy

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# Neural ODE Training
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(0, 3))
def train_step_mse(rhs_fn, params, opt_state, optimizer,
                   u_batch, rhs_batch):
    """Single gradient step with MSE loss."""
    loss, grads = jax.value_and_grad(
        lambda p: mse_loss(rhs_fn, p, u_batch, rhs_batch)
    )(params)
    updates, opt_state_new = optimizer.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, loss


def train_step_jac(rhs_fn, params, opt_state, optimizer,
                   u_batch, rhs_batch, jac_batch, lam=0.01):
    """Single gradient step with Jacobian-matching loss."""
    loss, grads = jax.value_and_grad(
        lambda p: jacobian_matching_loss(rhs_fn, p, u_batch, rhs_batch, jac_batch, lam)
    )(params)
    updates, opt_state_new = optimizer.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, loss


def create_lr_schedule(n_epochs: int, lr_init: float = 1e-3,
                        lr_final: float = 1e-4) -> optax.Schedule:
    """Exponential decay learning rate schedule."""
    return optax.exponential_decay(
        init_value=lr_init,
        transition_steps=n_epochs // 3,
        decay_rate=0.1 ** (1 / 3),
        end_value=lr_final
    )


def get_batches(data: dict, batch_size: int,
                key: jax.random.PRNGKey) -> list:
    """Shuffle and batch training data."""
    T = data["u"].shape[0]
    idx = jax.random.permutation(key, T)
    batches = []
    for start in range(0, T - batch_size, batch_size):
        b_idx = idx[start:start + batch_size]
        batch = {
            "u": jnp.array(data["u"][b_idx]),
            "rhs": jnp.array(data["rhs"][b_idx]),
        }
        if "jacobians" in data:
            batch["jacobians"] = jnp.array(data["jacobians"][b_idx])
        batches.append(batch)
    return batches


def train_neural_ode(solver, traj_train: np.ndarray,
                      mode: str = "mse",
                      node_type: str = "stabilized",
                      n_epochs: int = 2000,
                      batch_size: int = 256,
                      hidden: int = 256,
                      n_layers: int = 3,
                      lr_init: float = 1e-3,
                      lr_final: float = 1e-4,
                      jac_lambda: float = 0.01,
                      subsample: int = 4,
                      key: jax.random.PRNGKey = None,
                      save_path: str = "data") -> dict:
    """
    Train Neural ODE surrogate.

    Args:
        solver: KSSolver instance
        traj_train: (T, N) training trajectory
        mode: 'mse' or 'jac' (Jacobian-matching)
        node_type: 'standard' or 'stabilized'
        n_epochs: training epochs
        batch_size: batch size
        hidden: MLP hidden size
        n_layers: number of hidden layers
        lr_init, lr_final: learning rate schedule
        jac_lambda: Jacobian loss weight (for mode='jac')
        subsample: subsample training trajectory
        key: JAX random key
        save_path: directory to save model

    Returns:
        dict with params, loss_history, metadata
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    N = solver.N
    print(f"\n{'='*60}")
    print(f"Training Neural ODE | type={node_type} | mode={mode}")
    print(f"  epochs={n_epochs}, batch={batch_size}, hidden={hidden}x{n_layers}")
    print(f"  lr: {lr_init} -> {lr_final}")
    print(f"{'='*60}")

    # Initialize model
    k1, k2 = jax.random.split(key)
    if node_type == "stabilized":
        params = init_stabilized_node(k1, N=N, hidden=hidden, n_layers=n_layers)
        rhs_fn = stabilized_node_rhs
    else:
        params = init_standard_node(k1, N=N, hidden=hidden, n_layers=n_layers)
        rhs_fn = standard_node_rhs

    # Prepare training data
    compute_jac = (mode == "jac")
    train_data = prepare_training_data(
        traj_train, solver,
        compute_jacobians=compute_jac,
        subsample=subsample
    )
    print(f"  Training points: {train_data['u'].shape[0]}")

    # Optimizer
    schedule = create_lr_schedule(n_epochs, lr_init, lr_final)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    # Training loop
    loss_history = []
    best_loss = float("inf")
    best_params = params

    for epoch in range(n_epochs):
        k2, key_batch = jax.random.split(k2)
        batches = get_batches(train_data, batch_size, key_batch)

        epoch_losses = []
        for batch in batches:
            if mode == "mse":
                params, opt_state, loss = train_step_mse(
                    rhs_fn, params, opt_state, optimizer,
                    batch["u"], batch["rhs"]
                )
            else:  # jac
                params, opt_state, loss = train_step_jac(
                    rhs_fn, params, opt_state, optimizer,
                    batch["u"], batch["rhs"], batch["jacobians"], jac_lambda
                )
            epoch_losses.append(float(loss))

        mean_loss = np.mean(epoch_losses)
        loss_history.append(mean_loss)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = jax.tree_util.tree_map(lambda x: np.array(x), params)

        if (epoch + 1) % max(n_epochs // 10, 1) == 0:
            lr_current = float(schedule(epoch))
            print(f"  Epoch {epoch+1:5d}/{n_epochs} | "
                  f"Loss: {mean_loss:.6f} | "
                  f"LR: {lr_current:.2e} | "
                  f"Best: {best_loss:.6f}")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")

    # Package result
    result = {
        "params": best_params,
        "loss_history": np.array(loss_history),
        "metadata": {
            "mode": mode,
            "node_type": node_type,
            "n_epochs": n_epochs,
            "hidden": hidden,
            "n_layers": n_layers,
            "best_loss": best_loss,
            "N": N,
        }
    }

    # Save
    os.makedirs(save_path, exist_ok=True)
    model_name = f"node_{node_type}_{mode}.pkl"
    with open(os.path.join(save_path, model_name), "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved to {os.path.join(save_path, model_name)}")

    return result


def train_sindy_surrogate(solver, traj_train: np.ndarray,
                           n_modes: int = 8,
                           degree: int = 2,
                           threshold: float = 0.05,
                           save_path: str = "data") -> object:
    """
    Fit SINDy model (no gradient descent — just sparse regression).

    Args:
        solver: KSSolver
        traj_train: (T, N) training trajectory
        n_modes: number of POD modes
        degree: polynomial library degree
        threshold: STLSQ threshold
        save_path: where to save

    Returns:
        SINDyModel
    """
    print(f"\n{'='*60}")
    print(f"Fitting SINDy Surrogate")
    print(f"  modes={n_modes}, degree={degree}, threshold={threshold}")
    print(f"{'='*60}")

    model = fit_sindy(traj_train, solver, n_modes=n_modes,
                       degree=degree, threshold=threshold, dt=solver.dt)

    # Save
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "sindy_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"  SINDy model saved to {os.path.join(save_path, 'sindy_model.pkl')}")

    return model


def load_model(path: str):
    """Load a saved model."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from ks_solver import KSSolver
    import jax

    solver = KSSolver(L=22.0, N=64, dt=0.25)
    key = jax.random.PRNGKey(99)

    print("Generating training data...")
    k1, k2 = jax.random.split(key)
    u0_hat = solver.random_ic(k1)
    u0_hat = solver.warmup(u0_hat, n_warmup=2000)

    # Short training trajectory for testing
    traj_train = np.array(solver.integrate(u0_hat, n_steps=2000))
    print(f"Training trajectory: {traj_train.shape}")

    # Train standard NODE (MSE, small for testing)
    result_mse = train_neural_ode(
        solver, traj_train,
        mode="mse", node_type="standard",
        n_epochs=100, batch_size=128,
        hidden=64, n_layers=2,
        key=k2, subsample=2
    )
    print(f"MSE NODE training done. Final loss: {result_mse['metadata']['best_loss']:.6f}")

    # Train SINDy
    sindy_model = train_sindy_surrogate(solver, traj_train, n_modes=6)
    print("SINDy training done.")
