"""Neural-ODE vector-field model for the Park 2024 KS reproduction."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def init_mlp(
    key: jax.Array,
    layer_sizes: list[int],
    scale: float = 0.05,
    init_style: str = "scaled_normal",
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        if init_style == "scaled_normal":
            weight = jax.random.normal(k, (n_in, n_out), dtype=jnp.float64) * scale / jnp.sqrt(max(n_in, 1))
            bias = jnp.zeros((n_out,), dtype=jnp.float64)
        elif init_style == "pytorch_linear":
            kw, kb = jax.random.split(k)
            bound = 1.0 / jnp.sqrt(max(n_in, 1))
            weight = jax.random.uniform(
                kw,
                (n_in, n_out),
                minval=-bound,
                maxval=bound,
                dtype=jnp.float64,
            )
            bias = jax.random.uniform(
                kb,
                (n_out,),
                minval=-bound,
                maxval=bound,
                dtype=jnp.float64,
            )
        else:
            raise ValueError(f"Unsupported init_style: {init_style}")
        params.append((weight, bias))
    return params


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.gelu(x, approximate=False)


def mlp_forward(params: list[tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    for idx, (weight, bias) in enumerate(params):
        x = x @ weight + bias
        if idx < len(params) - 1:
            x = gelu(x)
    return x


def init_vector_field_mlp(
    key: jax.Array,
    input_dim: int,
    hidden_widths: tuple[int, ...],
    output_dim: int | None = None,
    *,
    init_style: str = "scaled_normal",
    architecture: str = "plain_mlp",
) -> dict:
    output_dim = input_dim if output_dim is None else output_dim
    if architecture == "plain_mlp":
        sizes = [input_dim, *hidden_widths, output_dim]
        return {"vf": init_mlp(key, sizes, init_style=init_style)}
    if architecture == "linear_plus_mlp":
        _, key_res = jax.random.split(key)
        sizes = [input_dim, *hidden_widths, output_dim]
        zero_linear = [
            (
                jnp.zeros((input_dim, output_dim), dtype=jnp.float64),
                jnp.zeros((output_dim,), dtype=jnp.float64),
            )
        ]
        return {
            "vf_linear": zero_linear,
            "vf_res": init_mlp(key_res, sizes, init_style=init_style),
        }
    raise ValueError(f"Unsupported architecture: {architecture}")


def vector_field_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    if "vf" in params:
        return mlp_forward(params["vf"], x)
    if "vf_linear" in params and "vf_res" in params:
        return mlp_forward(params["vf_linear"], x) + mlp_forward(params["vf_res"], x)
    raise ValueError("Unsupported parameter structure for vector field")


def one_step_rk4(params: dict, x: jnp.ndarray, dt: float) -> jnp.ndarray:
    dt = jnp.asarray(dt, dtype=jnp.float64)
    k1 = vector_field_forward(params, x)
    k2 = vector_field_forward(params, x + 0.5 * dt * k1)
    k3 = vector_field_forward(params, x + 0.5 * dt * k2)
    k4 = vector_field_forward(params, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def flow_map_forward(params: dict, x: jnp.ndarray, dt: float) -> jnp.ndarray:
    return one_step_rk4(params, x, dt)


def flow_map_jacobian(params: dict, x: jnp.ndarray, dt: float) -> jnp.ndarray:
    return jax.jacfwd(lambda y: flow_map_forward(params, y, dt))(x)
