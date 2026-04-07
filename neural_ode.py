"""
neural_ode.py — Neural ODE Surrogate for KSE
=============================================
Implements the Neural ODE surrogate following Chen et al. 2018 and
Linot et al. 2022 (stabilized variant).

Architecture:
    MLP: N → 256 → 256 → 256 → N
    Activation: tanh (smooth, differentiable — needed for Lyapunov)
    Integration: diffrax Dopri5 (adaptive step size)

Two training modes (Park 2024):
    1. MSE only:   L = ||f_θ(u) - true_rhs||²
    2. JAC:        L = ||f_θ(u) - true_rhs||² + λ||Jf_θ(u) - J_true(u)||²

The Jacobian loss (JAC mode) forces the surrogate to learn not just the
vector field but also its first derivative — this is what makes the learned
model reproduce physical measures and correct Lyapunov exponents.

Key insight from Park 2024:
    MSE model: small vector field error, WRONG Lyapunov spectrum
    JAC model: similar vector field error, CORRECT Lyapunov spectrum
    Reason: shadowing requires C¹ closeness, not just C⁰ closeness

Parameters are stored as a pytree (nested dicts of arrays) — pure JAX,
no frameworks required.
"""

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optax
from functools import partial
from typing import Callable

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# MLP in Pure JAX
# ─────────────────────────────────────────────────────────────────────────────

def init_mlp(key: jax.random.PRNGKey,
             layer_sizes: list,
             scale: float = 0.1) -> list:
    """
    Initialize MLP parameters.

    Args:
        layer_sizes: e.g. [64, 256, 256, 256, 64]
        scale: weight initialization scale

    Returns:
        params: list of (W, b) pairs, one per layer
    """
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = jax.random.normal(k, (n_in, n_out)) * scale / jnp.sqrt(n_in)
        b = jnp.zeros(n_out)
        params.append((W, b))
    return params


def mlp_forward(params: list, x: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass through MLP.
    tanh activations on all but last layer (linear output).
    """
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def mlp_jacobian(params: list, x: jnp.ndarray) -> jnp.ndarray:
    """
    Full Jacobian of MLP: shape (N_out, N_in).
    Uses JAX reverse-mode differentiation (jacrev, efficient for N_out << N_in,
    but here N_out = N_in = 64 so either mode is fine).
    """
    return jax.jacobian(mlp_forward, argnums=1)(params, x)


# ─────────────────────────────────────────────────────────────────────────────
# Stabilized Neural ODE (Linot 2022)
# ─────────────────────────────────────────────────────────────────────────────

def init_stabilized_node(key: jax.random.PRNGKey,
                          N: int = 64,
                          hidden: int = 256,
                          n_layers: int = 3) -> dict:
    """
    Stabilized Neural ODE: f_θ(u) = A_θ * u + F_θ(u)
    Linear term A_θ handles dissipation (stabilizes high wavenumbers).
    Nonlinear term F_θ captures the nonlinear KSE dynamics.

    Args:
        N: state dimension (number of Fourier modes)
        hidden: hidden layer size
        n_layers: number of hidden layers

    Returns:
        params dict with 'linear' (N, N) and 'nonlinear' (MLP params)
    """
    k1, k2 = jax.random.split(key)

    # Linear term: initialized as small random matrix
    # Will learn to approximate the KSE linear operator L_k = q²-q⁴
    A = jax.random.normal(k1, (N, N)) * 0.01

    # Nonlinear MLP
    sizes = [N] + [hidden] * n_layers + [N]
    nonlinear_params = init_mlp(k2, sizes, scale=0.01)

    return {"linear": A, "nonlinear": nonlinear_params}


def stabilized_node_rhs(params: dict, u: jnp.ndarray) -> jnp.ndarray:
    """Stabilized Neural ODE RHS: A*u + F(u)."""
    linear_out = u @ params["linear"].T
    nonlinear_out = mlp_forward(params["nonlinear"], u)
    return linear_out + nonlinear_out


def init_standard_node(key: jax.random.PRNGKey,
                        N: int = 64,
                        hidden: int = 256,
                        n_layers: int = 3) -> dict:
    """Standard Neural ODE (no explicit linear term). Baseline from Linot 2022."""
    sizes = [N] + [hidden] * n_layers + [N]
    return {"nonlinear": init_mlp(key, sizes, scale=0.01)}


def standard_node_rhs(params: dict, u: jnp.ndarray) -> jnp.ndarray:
    """Standard Neural ODE RHS: just the MLP."""
    return mlp_forward(params["nonlinear"], u)


# ─────────────────────────────────────────────────────────────────────────────
# Neural ODE Integration via diffrax
# ─────────────────────────────────────────────────────────────────────────────

def integrate_node(rhs_fn: Callable, params: dict,
                   u0: jnp.ndarray, t0: float, t1: float,
                   dt0: float = 0.1) -> jnp.ndarray:
    """
    Integrate Neural ODE from t0 to t1 using diffrax Dopri5.

    Args:
        rhs_fn: callable(params, u) -> du/dt
        params: model parameters
        u0: initial condition, shape (N,)
        t0, t1: time interval
        dt0: initial step size hint

    Returns:
        u1: state at t1, shape (N,)
    """
    def vector_field(t, y, args):
        return rhs_fn(args, y)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Dopri5(),
        t0=t0, t1=t1, dt0=dt0,
        y0=u0,
        args=params,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        max_steps=10000,
    )
    return sol.ys[-1]


def rollout_node(rhs_fn: Callable, params: dict,
                 u0: jnp.ndarray, n_steps: int,
                 dt: float = 0.25) -> jnp.ndarray:
    """
    Autoregressive rollout of Neural ODE for n_steps.
    Each step integrates from t to t+dt.

    Returns:
        trajectory: shape (n_steps, N)
    """
    def step_fn(u, _):
        u_next = integrate_node(rhs_fn, params, u, 0.0, dt, dt0=dt / 4)
        return u_next, u_next

    _, traj = jax.lax.scan(step_fn, u0, None, length=n_steps)
    return traj


def node_jacobian(rhs_fn: Callable, params: dict,
                  u: jnp.ndarray) -> jnp.ndarray:
    """
    Jacobian of Neural ODE RHS: ∂f_θ/∂u, shape (N, N).
    Used for Jacobian-matching loss and Lyapunov computation.
    """
    return jax.jacobian(lambda x: rhs_fn(params, x))(u)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(rhs_fn: Callable, params: dict,
             u_batch: jnp.ndarray, rhs_true: jnp.ndarray) -> jnp.ndarray:
    """
    MSE loss: L = mean ||f_θ(u_i) - rhs_true_i||²

    Args:
        u_batch: (B, N) batch of states
        rhs_true: (B, N) true RHS values
    """
    rhs_pred = jax.vmap(lambda u: rhs_fn(params, u))(u_batch)
    return jnp.mean((rhs_pred - rhs_true) ** 2)


def jacobian_matching_loss(rhs_fn: Callable, params: dict,
                           u_batch: jnp.ndarray,
                           rhs_true: jnp.ndarray,
                           jac_true: jnp.ndarray,
                           lam: float = 0.01) -> jnp.ndarray:
    """
    Jacobian-matching loss (Park 2024):
    L = ||f_θ(u) - f_true(u)||² + λ * ||Jf_θ(u) - Jf_true(u)||²

    The Jacobian term forces C¹ closeness, not just C⁰.
    This is what enables correct Lyapunov spectrum recovery.

    Args:
        u_batch: (B, N)
        rhs_true: (B, N) true vector field values
        jac_true: (B, N, N) true Jacobians at each point
        lam: weighting for Jacobian term
    """
    def single_loss(u, f_true, J_true):
        f_pred = rhs_fn(params, u)
        # jacfwd: forward-mode AD — same result as jacrev, ~20-30% faster
        # for square (N_in == N_out) Jacobians
        J_pred = jax.jacfwd(lambda x: rhs_fn(params, x))(u)
        mse = jnp.mean((f_pred - f_true) ** 2)
        jac_mse = jnp.mean((J_pred - J_true) ** 2)
        return mse + lam * jac_mse

    losses = jax.vmap(single_loss)(u_batch, rhs_true, jac_true)
    return jnp.mean(losses)


# ─────────────────────────────────────────────────────────────────────────────
# Training Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_training_data(traj: np.ndarray, solver,
                           compute_jacobians: bool = False,
                           subsample: int = 1,
                           cache_path: str = None) -> dict:
    """
    Prepare (u, rhs, jacobian) training pairs from trajectory.

    Caching: if cache_path is given and the file exists, load from disk.
    Otherwise compute and save to cache_path. This avoids recomputing
    expensive Jacobians across multiple training runs (T3 sweep etc.).

    Jacobians use jacfwd (forward-mode AD) — same result as jacrev,
    slightly faster for N_out == N_in == 64.

    Args:
        traj: (T, N) physical space trajectory
        solver: KSSolver instance (provides rhs_physical)
        compute_jacobians: whether to compute Jacobians
        subsample: use every subsample-th point
        cache_path: optional .npz path to cache/load from

    Returns:
        dict with u, rhs, [jacobians]
    """
    # --- Try cache first ---
    if cache_path is not None:
        import os
        if os.path.exists(cache_path):
            print(f"  Loading training data from cache: {cache_path}")
            cached = np.load(cache_path)
            result = {"u": cached["u"], "rhs": cached["rhs"]}
            if "jacobians" in cached and compute_jacobians:
                jac_cached = cached["jacobians"]
                expected_shape = (len(result["u"]), result["u"].shape[1], result["u"].shape[1])
                if jac_cached.shape == expected_shape:
                    result["jacobians"] = jac_cached
                    print(f"  Loaded {len(result['u'])} points + Jacobians from cache.")
                    return result
                print(f"  Cached Jacobians have shape {jac_cached.shape}, expected {expected_shape}. Recomputing cache.")
            elif not compute_jacobians:
                print(f"  Loaded {len(result['u'])} points from cache.")
                return result

    traj_sub = traj[::subsample]
    T, N = traj_sub.shape
    traj_jax = jnp.array(traj_sub, dtype=jnp.float64)

    print(f"  Computing RHS for {T} points (vmapped)...")
    rhs_vals = np.array(jax.jit(jax.vmap(solver.rhs_physical))(traj_jax))
    result = {"u": traj_sub, "rhs": rhs_vals}

    if compute_jacobians:
        print(f"  Computing Jacobians for {T} points (vmap over solver.jacobian_physical)...")
        # solver.jacobian_physical already returns the full N x N Jacobian.
        # We vmap that function directly across the batch.
        jac_fn = jax.jit(jax.vmap(solver.jacobian_physical))
        chunk = 256
        jac_list = []
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            jac_list.append(np.array(jac_fn(traj_jax[start:end])))
            print(f"    Jacobians {end}/{T}")
        result["jacobians"] = np.concatenate(jac_list, axis=0)

    # --- Save to cache ---
    if cache_path is not None:
        save_dict = {"u": result["u"], "rhs": result["rhs"]}
        if "jacobians" in result:
            save_dict["jacobians"] = result["jacobians"]
        np.savez(cache_path, **save_dict)
        print(f"  Cached training data to: {cache_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Neural ODE Lyapunov Analysis
# ─────────────────────────────────────────────────────────────────────────────

def node_lyapunov_rhs(rhs_fn: Callable, params: dict) -> Callable:
    """
    Wrap Neural ODE RHS for use in Lyapunov computation.
    Returns a function u -> du/dt in physical space.
    """
    def f(u):
        return rhs_fn(params, jnp.array(u))
    return f


if __name__ == "__main__":
    from ks_solver import KSSolver

    N = 64
    solver = KSSolver(L=22.0, N=N, dt=0.25)
    key = jax.random.PRNGKey(42)

    # Test standard Neural ODE
    k1, k2 = jax.random.split(key)
    params_std = init_standard_node(k1, N=N, hidden=128, n_layers=2)
    params_stab = init_stabilized_node(k2, N=N, hidden=128, n_layers=2)

    # Test forward pass
    u_test = jnp.ones(N) * 0.1
    out_std = standard_node_rhs(params_std, u_test)
    out_stab = stabilized_node_rhs(params_stab, u_test)
    print(f"Standard NODE output shape: {out_std.shape}, norm: {jnp.linalg.norm(out_std):.4f}")
    print(f"Stabilized NODE output shape: {out_stab.shape}, norm: {jnp.linalg.norm(out_stab):.4f}")

    # Test Jacobian
    J = node_jacobian(standard_node_rhs, params_std, u_test)
    print(f"Jacobian shape: {J.shape}")

    # Test integration
    u_next = integrate_node(standard_node_rhs, params_std, u_test, 0.0, 0.25)
    print(f"Integration output shape: {u_next.shape}")

    print("neural_ode.py: All checks passed.")
