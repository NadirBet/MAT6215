from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


DEFAULT_HIDDEN_SIZES = (200, 200, 200)
DEFAULT_MLP_STD = 1.0e-2
DEFAULT_CNN_LIMIT = float(np.sqrt(1.0 / 3.0))


@dataclass(frozen=True)
class KSEContext:
    L: float = 22.0
    N: int = 64
    tau: float = 0.25
    rtol: float = 1.0e-4
    atol: float = 1.0e-6

    @property
    def q(self) -> jnp.ndarray:
        k = jnp.fft.fftfreq(self.N, d=1.0 / self.N)
        return 2.0 * jnp.pi * k / self.L

    @property
    def linear_symbol(self) -> jnp.ndarray:
        q = self.q
        return q ** 2 - q ** 4


@dataclass(frozen=True)
class ModelBundle:
    name: str
    display_name: str
    context: KSEContext
    init_params: Callable[[jax.Array], dict]
    rhs: Callable[[dict, jnp.ndarray], jnp.ndarray]
    step: Callable[[dict, jnp.ndarray], jnp.ndarray]
    rollout: Callable[[dict, jnp.ndarray, int], jnp.ndarray]


def init_mlp(
    key: jax.Array,
    layer_sizes: Sequence[int],
    *,
    std: float = DEFAULT_MLP_STD,
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w = jax.random.normal(k, (n_in, n_out)) * std
        b = jnp.zeros((n_out,), dtype=jnp.float64)
        params.append((w, b))
    return params


def mlp_forward(params: list[tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    for idx, (w, b) in enumerate(params):
        x = x @ w + b
        if idx < len(params) - 1:
            x = jax.nn.sigmoid(x)
    return x


def apply_true_linear_operator(u: jnp.ndarray, context: KSEContext) -> jnp.ndarray:
    u_hat = jnp.fft.fft(u)
    out_hat = context.linear_symbol * u_hat
    return jnp.fft.ifft(out_hat).real


def apply_periodic_conv1d(u: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    radius = kernel.shape[0] // 2
    out = jnp.zeros_like(u)
    for idx in range(kernel.shape[0]):
        shift = radius - idx
        out = out + kernel[idx] * jnp.roll(u, shift)
    return out


def init_model_params(
    model_name: str,
    key: jax.Array,
    *,
    state_dim: int = 64,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
) -> dict:
    key_mlp, key_aux = jax.random.split(key)
    params = {"mlp": init_mlp(key_mlp, [state_dim, *hidden_sizes, state_dim])}
    if model_name == "cnn":
        params["filter"] = jax.random.uniform(
            key_aux,
            (5,),
            minval=-DEFAULT_CNN_LIMIT,
            maxval=DEFAULT_CNN_LIMIT,
        )
    return params


def build_rhs(model_name: str, context: KSEContext) -> Callable[[dict, jnp.ndarray], jnp.ndarray]:
    if model_name == "nonlinear":
        return lambda params, u: mlp_forward(params["mlp"], u)
    if model_name == "fixed_linear":
        return lambda params, u: apply_true_linear_operator(u, context) + mlp_forward(params["mlp"], u)
    if model_name == "cnn":
        return lambda params, u: apply_periodic_conv1d(u, params["filter"]) + mlp_forward(params["mlp"], u)
    raise ValueError(f"Unknown model name: {model_name}")


def integrate_onestep(
    rhs_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    u0: jnp.ndarray,
    *,
    tau: float,
    rtol: float,
    atol: float,
    dt0: float | None = None,
) -> jnp.ndarray:
    dt0 = tau / 4.0 if dt0 is None else dt0

    def vector_field(t, y, args):
        return rhs_fn(args, y)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Dopri5(),
        t0=0.0,
        t1=tau,
        dt0=dt0,
        y0=u0,
        args=params,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=10000,
    )
    return sol.ys[-1]


def build_model_bundle(
    model_name: str,
    *,
    context: KSEContext,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
) -> ModelBundle:
    rhs_fn = build_rhs(model_name, context)

    def init_params(key: jax.Array) -> dict:
        return init_model_params(
            model_name,
            key,
            state_dim=context.N,
            hidden_sizes=hidden_sizes,
        )

    @jax.jit
    def step_fn(params: dict, u: jnp.ndarray) -> jnp.ndarray:
        return integrate_onestep(
            rhs_fn,
            params,
            u,
            tau=context.tau,
            rtol=context.rtol,
            atol=context.atol,
        )

    def rollout_fn(params: dict, u0: jnp.ndarray, n_steps: int) -> jnp.ndarray:
        def scan_step(u, _):
            u_next = step_fn(params, u)
            return u_next, u_next

        _, traj = jax.lax.scan(scan_step, jnp.asarray(u0), None, length=n_steps)
        return traj

    labels = {
        "nonlinear": "Standard NODE",
        "fixed_linear": "Fixed-linear NODE",
        "cnn": "CNN NODE",
    }
    return ModelBundle(
        name=model_name,
        display_name=labels[model_name],
        context=context,
        init_params=init_params,
        rhs=rhs_fn,
        step=step_fn,
        rollout=rollout_fn,
    )


def bundle_dict(
    *,
    context: KSEContext,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
) -> dict[str, ModelBundle]:
    return {
        name: build_model_bundle(name, context=context, hidden_sizes=hidden_sizes)
        for name in ("nonlinear", "fixed_linear", "cnn")
    }
