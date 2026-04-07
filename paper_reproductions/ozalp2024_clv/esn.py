from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ESNConfig:
    n_input: int
    n_reservoir: int
    spectral_radius: float = 0.9
    input_scale: float = 0.25
    leak_rate: float = 1.0
    ridge_beta: float = 1.0e-6
    recurrent_sparsity: float = 0.05
    input_sparsity: float = 1.0
    bias_scale: float = 0.1
    seed: int = 0


def _sparse_random_matrix(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    sparsity: float,
    scale: float = 1.0,
) -> np.ndarray:
    mat = rng.normal(size=shape) * scale
    mask = rng.random(size=shape) < sparsity
    return mat * mask


def init_esn(config: ESNConfig) -> dict:
    rng = np.random.default_rng(config.seed)
    w = _sparse_random_matrix(
        rng,
        (config.n_reservoir, config.n_reservoir),
        sparsity=config.recurrent_sparsity,
    )
    eigvals = np.linalg.eigvals(w)
    radius = np.max(np.abs(eigvals)) if eigvals.size else 1.0
    if radius > 0:
        w = w * (config.spectral_radius / radius)

    w_in = _sparse_random_matrix(
        rng,
        (config.n_reservoir, config.n_input),
        sparsity=config.input_sparsity,
        scale=config.input_scale,
    )
    bias = rng.normal(scale=config.bias_scale, size=(config.n_reservoir,))
    return {
        "config": asdict(config),
        "W": jnp.asarray(w),
        "W_in": jnp.asarray(w_in),
        "bias": jnp.asarray(bias),
    }


def reservoir_update(esn: dict, r: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    alpha = esn["config"]["leak_rate"]
    pre = esn["W"] @ r + esn["W_in"] @ z + esn["bias"]
    return (1.0 - alpha) * r + alpha * jnp.tanh(pre)


def open_loop_states(esn: dict, z_seq: jnp.ndarray, *, r0: jnp.ndarray | None = None) -> jnp.ndarray:
    n_res = esn["W"].shape[0]
    r0 = jnp.zeros((n_res,), dtype=jnp.float64) if r0 is None else jnp.asarray(r0)

    def step(r, z):
        r_next = reservoir_update(esn, r, z)
        return r_next, r_next

    _, states = jax.lax.scan(step, r0, z_seq)
    return states


def fit_readout(
    esn: dict,
    z_seq: np.ndarray,
    *,
    ridge_beta: float | None = None,
) -> dict:
    z_seq_j = jnp.asarray(z_seq)
    states = np.asarray(open_loop_states(esn, z_seq_j[:-1]))
    augmented = np.concatenate([states, np.ones((states.shape[0], 1))], axis=1)
    targets = np.asarray(z_seq[1:])
    beta = esn["config"]["ridge_beta"] if ridge_beta is None else ridge_beta
    gram = augmented.T @ augmented
    rhs = augmented.T @ targets
    w_out = np.linalg.solve(gram + beta * np.eye(gram.shape[0]), rhs).T
    fitted = dict(esn)
    fitted["W_out"] = jnp.asarray(w_out)
    return fitted


def latent_readout(esn: dict, r: jnp.ndarray) -> jnp.ndarray:
    augmented = jnp.concatenate([r, jnp.ones((1,), dtype=r.dtype)])
    return esn["W_out"] @ augmented


def closed_loop_step(esn: dict, r: jnp.ndarray) -> jnp.ndarray:
    z = latent_readout(esn, r)
    return reservoir_update(esn, r, z)


def closed_loop_rollout(esn: dict, r0: jnp.ndarray, n_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    def step(r, _):
        r_next = closed_loop_step(esn, r)
        z_next = latent_readout(esn, r_next)
        return r_next, (r_next, z_next)

    _, (r_hist, z_hist) = jax.lax.scan(step, jnp.asarray(r0), None, length=n_steps)
    return r_hist, z_hist


def warm_start_state(esn: dict, z_prefix: np.ndarray) -> jnp.ndarray:
    states = open_loop_states(esn, jnp.asarray(z_prefix))
    return states[-1]


def esn_closed_loop_jacobian(esn: dict, r: jnp.ndarray) -> jnp.ndarray:
    alpha = esn["config"]["leak_rate"]
    w_out_nb = esn["W_out"][:, :-1]
    pre = esn["W"] @ r + esn["W_in"] @ latent_readout(esn, r) + esn["bias"]
    gain = 1.0 - jnp.tanh(pre) ** 2
    linear_part = esn["W"] + esn["W_in"] @ w_out_nb
    return (1.0 - alpha) * jnp.eye(esn["W"].shape[0], dtype=jnp.float64) + alpha * (gain[:, None] * linear_part)


def esn_tangent_step(esn: dict, r: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    alpha = esn["config"]["leak_rate"]
    w_out_nb = esn["W_out"][:, :-1]
    pre = esn["W"] @ r + esn["W_in"] @ latent_readout(esn, r) + esn["bias"]
    gain = 1.0 - jnp.tanh(pre) ** 2
    linear_q = esn["W"] @ q + esn["W_in"] @ (w_out_nb @ q)
    return (1.0 - alpha) * q + alpha * (gain[:, None] * linear_q)


def esn_output_tangent(esn: dict, q: jnp.ndarray) -> jnp.ndarray:
    w_out_nb = esn["W_out"][:, :-1]
    return w_out_nb @ q


def validation_rollout_mse(esn: dict, z_prefix: np.ndarray, z_target: np.ndarray, horizon: int) -> float:
    r0 = warm_start_state(esn, z_prefix)
    _, z_pred = closed_loop_rollout(esn, r0, horizon)
    return float(np.mean((np.asarray(z_pred) - np.asarray(z_target[:horizon])) ** 2))
