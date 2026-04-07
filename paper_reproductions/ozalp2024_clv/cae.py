from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class CAEConfig:
    n_grid: int
    latent_dim: int
    channels1: int = 8
    channels2: int = 16
    kernel_size: int = 5
    init_std: float = 1.0e-2


def init_linear(key, n_in: int, n_out: int, std: float = 1.0e-2):
    w = jax.random.normal(key, (n_in, n_out)) * std
    b = jnp.zeros((n_out,), dtype=jnp.float64)
    return w, b


def apply_linear(params, x):
    w, b = params
    return x @ w + b


def init_conv(key, kernel_size: int, c_in: int, c_out: int, std: float = 1.0e-2):
    w = jax.random.normal(key, (kernel_size, c_in, c_out)) * std
    b = jnp.zeros((c_out,), dtype=jnp.float64)
    return w, b


def periodic_conv1d(x: jnp.ndarray, params, *, stride: int = 1) -> jnp.ndarray:
    w, b = params
    pad = w.shape[0] // 2
    x_pad = jnp.pad(x, ((pad, pad), (0, 0)), mode="wrap")
    y = jax.lax.conv_general_dilated(
        lhs=x_pad[None, ...],
        rhs=w,
        window_strides=(stride,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )[0]
    return y + b


def upsample_repeat(x: jnp.ndarray, factor: int = 2) -> jnp.ndarray:
    return jnp.repeat(x, factor, axis=0)


def init_cae(key: jax.Array, config: CAEConfig) -> dict:
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    reduced_len = config.n_grid // 4
    hidden_flat = reduced_len * config.channels2
    return {
        "enc_conv1": init_conv(k1, config.kernel_size, 1, config.channels1, config.init_std),
        "enc_conv2": init_conv(k2, config.kernel_size, config.channels1, config.channels2, config.init_std),
        "enc_dense": init_linear(k3, hidden_flat, config.latent_dim, config.init_std),
        "dec_dense": init_linear(k4, config.latent_dim, hidden_flat, config.init_std),
        "dec_conv1": init_conv(k5, config.kernel_size, config.channels2, config.channels1, config.init_std),
        "dec_conv2": init_conv(k6, config.kernel_size, config.channels1, 1, config.init_std),
    }


def encode(params: dict, u: jnp.ndarray, config: CAEConfig) -> jnp.ndarray:
    x = u[:, None]
    x = jax.nn.relu(periodic_conv1d(x, params["enc_conv1"], stride=2))
    x = jax.nn.relu(periodic_conv1d(x, params["enc_conv2"], stride=2))
    x = x.reshape((-1,))
    return apply_linear(params["enc_dense"], x)


def decode(params: dict, z: jnp.ndarray, config: CAEConfig) -> jnp.ndarray:
    reduced_len = config.n_grid // 4
    x = jax.nn.relu(apply_linear(params["dec_dense"], z))
    x = x.reshape((reduced_len, config.channels2))
    x = upsample_repeat(x, 2)
    x = jax.nn.relu(periodic_conv1d(x, params["dec_conv1"], stride=1))
    x = upsample_repeat(x, 2)
    x = periodic_conv1d(x, params["dec_conv2"], stride=1)
    return x[:, 0]


def reconstruct(params: dict, u: jnp.ndarray, config: CAEConfig) -> jnp.ndarray:
    return decode(params, encode(params, u, config), config)


def encode_batch(params: dict, u_batch: jnp.ndarray, config: CAEConfig) -> jnp.ndarray:
    return jax.vmap(lambda u: encode(params, u, config))(u_batch)


def decode_batch(params: dict, z_batch: jnp.ndarray, config: CAEConfig) -> jnp.ndarray:
    return jax.vmap(lambda z: decode(params, z, config))(z_batch)


def reconstruct_batch(params: dict, u_batch: jnp.ndarray, config: CAEConfig) -> jnp.ndarray:
    return jax.vmap(lambda u: reconstruct(params, u, config))(u_batch)
