"""Lyapunov spectrum utilities for differentiable discrete maps."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def compute_map_lyapunov(
    map_fn,
    x0: np.ndarray,
    *,
    n_steps: int = 30000,
    n_lyap: int = 15,
    n_warmup: int = 1000,
    time_per_step: float = 1.0,
    progress_callback=None,
    progress_block_size: int = 250,
) -> np.ndarray:
    """
    Benettin / QR algorithm for a discrete-time map x_{k+1} = F(x_k).
    """
    x = jnp.array(x0, dtype=jnp.float64)
    dim = int(x.shape[0])
    n_lyap = min(n_lyap, dim)
    q = jnp.eye(dim, n_lyap, dtype=jnp.float64)
    log_accum = jnp.zeros((n_lyap,), dtype=jnp.float64)

    step = jax.jit(map_fn)

    def qr_step(carry, _):
        state, basis, logs = carry
        mapped_basis = jax.vmap(lambda vec: jax.jvp(step, (state,), (vec,))[1], in_axes=1, out_axes=1)(basis)
        next_state = step(state)
        q_next, r = jnp.linalg.qr(mapped_basis)
        diag = jnp.diag(r)
        signs = jnp.where(diag < 0.0, -1.0, 1.0)
        q_next = q_next * signs[None, :]
        diag = jnp.abs(diag) + jnp.finfo(jnp.float64).tiny
        logs = logs + jnp.log(diag)
        return (next_state, q_next, logs), None

    qr_block = jax.jit(lambda carry, block_len: jax.lax.scan(qr_step, carry, None, length=block_len)[0], static_argnums=1)

    if n_warmup > 0:
        def warmup_step(state, _):
            return step(state), None
        x, _ = jax.lax.scan(warmup_step, x, None, length=n_warmup)

    carry = (x, q, log_accum)
    completed = 0
    block_size = max(int(progress_block_size), 1)
    while completed < n_steps:
        block_len = min(block_size, n_steps - completed)
        carry = qr_block(carry, block_len)
        completed += block_len
        if progress_callback is not None:
            progress_callback(
                {
                    "completed_steps": completed,
                    "total_steps": n_steps,
                    "fraction": completed / max(n_steps, 1),
                }
            )

    x, q, log_accum = carry
    total_time = max(n_steps * float(time_per_step), np.finfo(np.float64).tiny)
    exponents = np.array(log_accum / total_time)
    return np.sort(exponents)[::-1]


def compute_map_lyapunov_along_trajectory(
    map_fn,
    trajectory: np.ndarray,
    *,
    n_steps: int | None = None,
    n_lyap: int = 15,
    time_per_step: float = 1.0,
    progress_callback=None,
    progress_block_size: int = 250,
) -> np.ndarray:
    """
    Benettin / QR algorithm for a discrete-time map, but evaluate Jacobians
    along a supplied true trajectory x_k instead of the model's autonomous orbit.

    This mirrors the evaluation path in the released stacNODE KS script.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"trajectory must have shape (T, d), got {traj.shape}")

    total_states, dim = traj.shape
    if n_steps is None:
        n_steps = total_states
    n_steps = min(int(n_steps), total_states)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    n_lyap = min(int(n_lyap), dim)
    q = jnp.eye(dim, n_lyap, dtype=jnp.float64)
    log_accum = jnp.zeros((n_lyap,), dtype=jnp.float64)

    step = jax.jit(map_fn)

    @jax.jit
    def qr_step_at_state(state, basis, logs):
        mapped_basis = jax.vmap(
            lambda vec: jax.jvp(step, (state,), (vec,))[1],
            in_axes=1,
            out_axes=1,
        )(basis)
        q_next, r = jnp.linalg.qr(mapped_basis)
        diag = jnp.diag(r)
        signs = jnp.where(diag < 0.0, -1.0, 1.0)
        q_next = q_next * signs[None, :]
        diag = jnp.abs(diag) + jnp.finfo(jnp.float64).tiny
        logs = logs + jnp.log(diag)
        return q_next, logs

    block_size = max(int(progress_block_size), 1)
    for idx in range(n_steps):
        state = jnp.array(traj[idx], dtype=jnp.float64)
        q, log_accum = qr_step_at_state(state, q, log_accum)
        completed = idx + 1
        if progress_callback is not None and ((completed % block_size == 0) or (completed == n_steps)):
            progress_callback(
                {
                    "completed_steps": completed,
                    "total_steps": n_steps,
                    "fraction": completed / max(n_steps, 1),
                }
            )

    total_time = max(n_steps * float(time_per_step), np.finfo(np.float64).tiny)
    exponents = np.array(log_accum / total_time)
    return np.sort(exponents)[::-1]
