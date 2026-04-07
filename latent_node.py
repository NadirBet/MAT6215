"""
latent_node.py - Linot-style Reduced-Manifold Neural ODE for KSE
=================================================================
Implements the full encoder/decoder + latent ODE pipeline from
Linot & Graham 2022 (arXiv:2203.15706).

Architecture:
    Encoder chi:     R^N -> R^d   (N=64 physical -> d-dim latent)
    Latent ODE g:    R^d -> R^d   (dynamics in latent space)
    Decoder chi_inv: R^d -> R^N   (latent -> physical)

Training Protocol:
    Phase 1 — Autoencoder pretraining:
        L_AE = mean ||chi_inv(chi(u)) - u||^2
    Phase 2 — Latent ODE training:
        Compute h(t) = chi(u(t))
        Compute dh/dt = J_chi(u) * (du/dt)   [chain rule]
        L_ODE = mean ||g_theta(h) - dh/dt||^2
    Optional Phase 3 — End-to-end trajectory supervision:
        L_traj = mean ||chi_inv(Phi_g(chi(u0), T)) - u_T||^2

Comparison modes:
    'latent_node'    - this module (nonlinear AE + latent ODE)
    'linear_latent'  - POD + latent ODE (linear AE, as in linear ROM)
    'discrete_map'   - learned discrete map h_{n+1} = G(h_n)

Key insight: Operating in latent space separates representation from dynamics.
The autoencoder captures the geometry of the attractor; the ODE captures the
flow restricted to it. Both can be tested independently.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from typing import Callable, Optional, Tuple

jax.config.update("jax_enable_x64", True)


# =============================================================================
# MLP utilities (shared with neural_ode.py, duplicated for independence)
# =============================================================================

def init_mlp(key, layer_sizes, scale=0.1):
    """Initialize MLP with He-like scaling."""
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = jax.random.normal(k, (n_in, n_out)) * scale / jnp.sqrt(n_in)
        b = jnp.zeros(n_out)
        params.append((W, b))
    return params


def mlp_forward(params, x):
    """MLP forward: tanh on all layers except last (linear output)."""
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


# =============================================================================
# Autoencoder: Encoder + Decoder
# =============================================================================

def init_autoencoder(key, N=64, d=8, hidden_ae=128, n_layers_ae=2):
    """
    Initialize encoder/decoder pair.

    Encoder: N -> hidden -> hidden -> d   (tanh activations)
    Decoder: d -> hidden -> hidden -> N   (tanh hidden, linear output)

    Args:
        N: physical state dimension (64 for KSE)
        d: latent dimension
        hidden_ae: hidden layer width
        n_layers_ae: number of hidden layers in each network

    Returns:
        ae_params dict with 'encoder' and 'decoder' MLP params
    """
    k1, k2 = jax.random.split(key)

    enc_sizes = [N] + [hidden_ae] * n_layers_ae + [d]
    dec_sizes = [d] + [hidden_ae] * n_layers_ae + [N]

    return {
        "encoder": init_mlp(k1, enc_sizes, scale=0.1),
        "decoder": init_mlp(k2, dec_sizes, scale=0.1),
    }


def encode(ae_params, u):
    """Encode physical state u (N,) -> latent h (d,)."""
    return mlp_forward(ae_params["encoder"], u)


def decode(ae_params, h):
    """Decode latent h (d,) -> physical u (N,)."""
    return mlp_forward(ae_params["decoder"], h)


def ae_loss(ae_params, u_batch):
    """
    Autoencoder reconstruction loss.
    L_AE = (1/B) sum ||decode(encode(u_i)) - u_i||^2
    """
    h_batch = jax.vmap(lambda u: encode(ae_params, u))(u_batch)
    u_rec = jax.vmap(lambda h: decode(ae_params, h))(h_batch)
    return jnp.mean((u_rec - u_batch) ** 2)


def ae_reconstruction_error(ae_params, u_batch):
    """Per-sample reconstruction MSE (for diagnostics)."""
    h_batch = jax.vmap(lambda u: encode(ae_params, u))(u_batch)
    u_rec = jax.vmap(lambda h: decode(ae_params, h))(h_batch)
    return jnp.mean((u_rec - u_batch) ** 2, axis=1)  # (B,)


# =============================================================================
# Linear Autoencoder (POD) for comparison
# =============================================================================

def fit_pod_autoencoder(traj, d):
    """
    Fit a linear (POD) autoencoder via SVD.

    Encoder: u -> h = (u - u_mean) @ Phi       [Phi: (N, d)]
    Decoder: h -> u = u_mean + h @ Phi.T

    Returns:
        pod_params dict with 'Phi', 'u_mean', 'd'
    """
    u_mean = np.mean(traj, axis=0)
    traj_c = traj - u_mean
    _, _, Vt = np.linalg.svd(traj_c, full_matrices=False)
    Phi = Vt[:d].T  # (N, d)
    return {"Phi": jnp.array(Phi), "u_mean": jnp.array(u_mean), "d": d}


def pod_encode(pod_params, u):
    """POD encode: u (N,) -> h (d,)."""
    return (u - pod_params["u_mean"]) @ pod_params["Phi"]


def pod_decode(pod_params, h):
    """POD decode: h (d,) -> u (N,)."""
    return pod_params["u_mean"] + h @ pod_params["Phi"].T


# =============================================================================
# Latent ODE
# =============================================================================

def init_latent_ode(key, d=8, hidden=128, n_layers=3):
    """
    Latent ODE vector field: g: R^d -> R^d.

    Architecture: d -> hidden -> ... -> hidden -> d
    Activation: tanh throughout (smooth, needed for Lyapunov JVP).

    Args:
        d: latent dimension
        hidden: hidden layer width
        n_layers: number of hidden layers

    Returns:
        ode_params dict with 'mlp' params and 'd'
    """
    sizes = [d] + [hidden] * n_layers + [d]
    return {
        "mlp": init_mlp(key, sizes, scale=0.01),
    }


def latent_ode_rhs(ode_params, h):
    """Latent ODE RHS: dh/dt = g_theta(h)."""
    return mlp_forward(ode_params["mlp"], h)


# =============================================================================
# Discrete-time Latent Map (T7 comparison)
# =============================================================================

def init_discrete_map(key, d=8, hidden=128, n_layers=3):
    """
    Discrete-time map: G: R^d -> R^d, h_{n+1} = G(h_n).

    Same architecture as latent ODE but trained on (h_n, h_{n+1}) pairs
    instead of (h, dh/dt) pairs.
    """
    sizes = [d] + [hidden] * n_layers + [d]
    return {
        "mlp": init_mlp(key, sizes, scale=0.01),
    }


def discrete_map_step(map_params, h):
    """One step of discrete map: h_n -> h_{n+1}."""
    return mlp_forward(map_params["mlp"], h)


# =============================================================================
# Training Data Preparation for Latent Space
# =============================================================================

def prepare_latent_data_nonlinear(traj, ae_params, solver, subsample=1):
    """
    Compute latent training data from physical trajectory using nonlinear AE.

    Returns h(t) and dh/dt via chain rule:
        dh/dt = J_chi(u) * (du/dt)

    where J_chi is the Jacobian of the encoder w.r.t. physical state.

    Args:
        traj: (T, N) physical trajectory (real space)
        ae_params: trained autoencoder params
        solver: KSSolver (provides rhs_physical)
        subsample: use every subsample-th point

    Returns:
        dict with 'h' (T', d) and 'dhdt' (T', d)
    """
    traj_sub = traj[::subsample]
    T, N = traj_sub.shape

    # infer d from encoder output
    u_tmp = jnp.array(traj_sub[0], dtype=jnp.float64)
    d = int(encode(ae_params, u_tmp).shape[0])

    print(f"  Computing latent training data for {T} points (nonlinear AE, d={d})...")

    # Encoder Jacobian function
    jac_encoder = jax.jit(jax.jacobian(lambda u: encode(ae_params, u)))
    encode_jit = jax.jit(lambda u: encode(ae_params, u))

    h_all = np.zeros((T, d))
    dhdt_all = np.zeros((T, d))

    for i in range(T):
        if i % max(T // 5, 1) == 0:
            print(f"    {i}/{T}")
        u = jnp.array(traj_sub[i], dtype=jnp.float64)

        # RHS: support both KSSolver (has .rhs Fourier method) and
        # NormSolver (has .rhs_physical_norm method)
        if hasattr(solver, 'rhs_physical_norm'):
            dudt = jnp.array(solver.rhs_physical_norm(np.array(u)), dtype=jnp.float64)
        else:
            u_hat = jnp.fft.fft(u)
            rhs_hat = solver.rhs(u_hat)
            dudt = jnp.fft.ifft(rhs_hat).real

        # Latent state
        h_all[i] = np.array(encode_jit(u))

        # dh/dt via chain rule: J_chi(u) @ dudt
        J = np.array(jac_encoder(u))  # (d, N)
        dhdt_all[i] = J @ np.array(dudt)

    return {"h": h_all, "dhdt": dhdt_all}


def prepare_latent_data_pod(traj, pod_params, solver, subsample=1):
    """
    Compute latent training data using linear POD encoder.

    dh/dt = Phi^T * du/dt  (Phi is orthonormal, so chain rule is just projection)

    Args:
        traj: (T, N) physical trajectory
        pod_params: dict with 'Phi' (N, d) and 'u_mean' (N,)
        solver: KSSolver
        subsample: subsampling factor

    Returns:
        dict with 'h' (T', d) and 'dhdt' (T', d)
    """
    traj_sub = traj[::subsample]
    T, N = traj_sub.shape
    Phi = np.array(pod_params["Phi"])    # (N, d)
    u_mean = np.array(pod_params["u_mean"])  # (N,)

    print(f"  Computing latent training data for {T} points (POD)...")

    # Project all states
    h_all = (traj_sub - u_mean) @ Phi  # (T, d)
    dhdt_all = np.zeros_like(h_all)

    for i in range(T):
        if hasattr(solver, 'rhs_physical_norm'):
            dudt = np.array(solver.rhs_physical_norm(traj_sub[i]))
        else:
            u_hat = jnp.fft.fft(jnp.array(traj_sub[i]))
            rhs_hat = solver.rhs(u_hat)
            dudt = np.array(jnp.fft.ifft(rhs_hat).real)
        dhdt_all[i] = dudt @ Phi  # project RHS onto POD modes

    return {"h": h_all, "dhdt": dhdt_all}


def prepare_discrete_latent_data(traj, encode_fn, subsample=1):
    """
    Prepare (h_n, h_{n+1}) pairs for discrete-map training.

    Args:
        traj: (T, N) physical trajectory
        encode_fn: callable u -> h
        subsample: subsampling factor

    Returns:
        dict with 'h_n' (T-1, d) and 'h_n1' (T-1, d)
    """
    traj_sub = traj[::subsample]
    T = len(traj_sub)

    encode_jit = jax.jit(encode_fn)
    h_all = np.array(jax.vmap(encode_jit)(jnp.array(traj_sub)))

    return {"h_n": h_all[:-1], "h_n1": h_all[1:]}


# =============================================================================
# Loss Functions for Latent ODE and Discrete Map
# =============================================================================

def latent_ode_loss(ode_params, h_batch, dhdt_batch):
    """
    Latent ODE MSE loss: L = mean ||g_theta(h_i) - dhdt_i||^2

    Args:
        h_batch: (B, d) latent states
        dhdt_batch: (B, d) true latent time derivatives
    """
    dhdt_pred = jax.vmap(lambda h: latent_ode_rhs(ode_params, h))(h_batch)
    return jnp.mean((dhdt_pred - dhdt_batch) ** 2)


def discrete_map_loss(map_params, h_n_batch, h_n1_batch):
    """
    Discrete map MSE loss: L = mean ||G(h_n) - h_{n+1}||^2

    Args:
        h_n_batch: (B, d) current latent states
        h_n1_batch: (B, d) next latent states
    """
    h_pred = jax.vmap(lambda h: discrete_map_step(map_params, h))(h_n_batch)
    return jnp.mean((h_pred - h_n1_batch) ** 2)


# =============================================================================
# Training Loops
# =============================================================================

def train_autoencoder(ae_params, u_data, n_epochs=500, batch_size=256,
                      lr_init=1e-3, lr_final=1e-5, key=None):
    """
    Train autoencoder via Adam + exponential LR decay.

    Args:
        ae_params: initial AE params
        u_data: (T, N) training states (physical space)
        n_epochs: number of training epochs
        batch_size: batch size
        lr_init, lr_final: learning rate schedule endpoints
        key: JAX random key

    Returns:
        trained ae_params, loss_history
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n_data = len(u_data)
    lr_schedule = optax.exponential_decay(
        lr_init, transition_steps=n_epochs, decay_rate=lr_final / lr_init
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(ae_params)

    @jax.jit
    def train_step(params, opt_state, u_batch):
        loss, grads = jax.value_and_grad(ae_loss)(params, u_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    loss_history = []
    u_data_jax = jnp.array(u_data, dtype=jnp.float64)

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, n_data)

        epoch_losses = []
        for start in range(0, n_data - batch_size + 1, batch_size):
            batch_idx = idx[start:start + batch_size]
            u_batch = u_data_jax[batch_idx]
            ae_params, opt_state, loss = train_step(ae_params, opt_state, u_batch)
            epoch_losses.append(float(loss))

        mean_loss = np.mean(epoch_losses)
        loss_history.append(mean_loss)

        if epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1:
            print(f"  AE Epoch {epoch:4d}/{n_epochs}  loss={mean_loss:.6f}")

    return ae_params, loss_history


def train_latent_ode(ode_params, h_data, dhdt_data, n_epochs=500,
                     batch_size=256, lr_init=1e-3, lr_final=1e-5, key=None):
    """
    Train latent ODE via Adam + exponential LR decay.

    Args:
        ode_params: initial latent ODE params
        h_data: (T, d) latent states
        dhdt_data: (T, d) latent time derivatives
        n_epochs, batch_size, lr_init, lr_final: training hyperparams
        key: JAX random key

    Returns:
        trained ode_params, loss_history
    """
    if key is None:
        key = jax.random.PRNGKey(1)

    n_data = len(h_data)
    lr_schedule = optax.exponential_decay(
        lr_init, transition_steps=n_epochs, decay_rate=lr_final / lr_init
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(ode_params)

    @jax.jit
    def train_step(params, opt_state, h_batch, dhdt_batch):
        loss, grads = jax.value_and_grad(latent_ode_loss)(params, h_batch, dhdt_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    h_jax = jnp.array(h_data, dtype=jnp.float64)
    dhdt_jax = jnp.array(dhdt_data, dtype=jnp.float64)
    loss_history = []

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, n_data)

        epoch_losses = []
        for start in range(0, n_data - batch_size + 1, batch_size):
            batch_idx = idx[start:start + batch_size]
            h_batch = h_jax[batch_idx]
            dhdt_batch = dhdt_jax[batch_idx]
            ode_params, opt_state, loss = train_step(
                ode_params, opt_state, h_batch, dhdt_batch
            )
            epoch_losses.append(float(loss))

        mean_loss = np.mean(epoch_losses)
        loss_history.append(mean_loss)

        if epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1:
            print(f"  ODE Epoch {epoch:4d}/{n_epochs}  loss={mean_loss:.6f}")

    return ode_params, loss_history


def train_discrete_map(map_params, h_n_data, h_n1_data, n_epochs=500,
                       batch_size=256, lr_init=1e-3, lr_final=1e-5, key=None):
    """
    Train discrete map G: h_n -> h_{n+1}.

    Args:
        map_params: initial discrete map params
        h_n_data: (T-1, d) current latent states
        h_n1_data: (T-1, d) next latent states
        other args: same as train_latent_ode

    Returns:
        trained map_params, loss_history
    """
    if key is None:
        key = jax.random.PRNGKey(2)

    n_data = len(h_n_data)
    lr_schedule = optax.exponential_decay(
        lr_init, transition_steps=n_epochs, decay_rate=lr_final / lr_init
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(map_params)

    @jax.jit
    def train_step(params, opt_state, h_n_batch, h_n1_batch):
        loss, grads = jax.value_and_grad(discrete_map_loss)(
            params, h_n_batch, h_n1_batch
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    h_n_jax = jnp.array(h_n_data, dtype=jnp.float64)
    h_n1_jax = jnp.array(h_n1_data, dtype=jnp.float64)
    loss_history = []

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, n_data)

        epoch_losses = []
        for start in range(0, n_data - batch_size + 1, batch_size):
            batch_idx = idx[start:start + batch_size]
            h_n_b = h_n_jax[batch_idx]
            h_n1_b = h_n1_jax[batch_idx]
            map_params, opt_state, loss = train_step(
                map_params, opt_state, h_n_b, h_n1_b
            )
            epoch_losses.append(float(loss))

        mean_loss = np.mean(epoch_losses)
        loss_history.append(mean_loss)

        if epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1:
            print(f"  Map Epoch {epoch:4d}/{n_epochs}  loss={mean_loss:.6f}")

    return map_params, loss_history


# =============================================================================
# Rollout Functions
# =============================================================================

def rk4_step_latent(ode_params, h, dt):
    """Fixed-step RK4 for latent ODE (enables JVP for Lyapunov)."""
    k1 = latent_ode_rhs(ode_params, h)
    k2 = latent_ode_rhs(ode_params, h + dt / 2 * k1)
    k3 = latent_ode_rhs(ode_params, h + dt / 2 * k2)
    k4 = latent_ode_rhs(ode_params, h + dt * k3)
    return h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout_latent_node(ae_params, ode_params, u0, n_steps, dt=0.25,
                        encode_fn=None, decode_fn=None):
    """
    Rollout latent NODE:
        1. Encode u0 -> h0 = chi(u0)
        2. Integrate h(t) via RK4 in latent space
        3. Decode h(t) -> u(t) = chi_inv(h(t))

    Args:
        ae_params: autoencoder params (or pod_params for linear AE)
        ode_params: latent ODE params
        u0: (N,) initial physical state
        n_steps: number of rollout steps
        dt: time step
        encode_fn: override for encoder (e.g., pod_encode). Uses encode() if None.
        decode_fn: override for decoder (e.g., pod_decode). Uses decode() if None.

    Returns:
        traj_phys: (n_steps, N) physical space trajectory
        traj_lat: (n_steps, d) latent space trajectory
    """
    if encode_fn is None:
        encode_fn = lambda u: encode(ae_params, u)
    if decode_fn is None:
        decode_fn = lambda h: decode(ae_params, h)

    h0 = encode_fn(jnp.array(u0))

    def step_fn(h, _):
        h_next = rk4_step_latent(ode_params, h, dt)
        u_next = decode_fn(h_next)
        return h_next, (h_next, u_next)

    h_final, (traj_lat, traj_phys) = jax.lax.scan(
        step_fn, h0, None, length=n_steps
    )
    return np.array(traj_phys), np.array(traj_lat)


def rollout_discrete_map(ae_params, map_params, u0, n_steps,
                         encode_fn=None, decode_fn=None):
    """
    Rollout discrete latent map:
        1. h0 = chi(u0)
        2. h_{n+1} = G(h_n)
        3. u_n = chi_inv(h_n)

    Args:
        Same as rollout_latent_node but uses map_params.

    Returns:
        traj_phys: (n_steps, N) physical trajectory
        traj_lat: (n_steps, d) latent trajectory
    """
    if encode_fn is None:
        encode_fn = lambda u: encode(ae_params, u)
    if decode_fn is None:
        decode_fn = lambda h: decode(ae_params, h)

    h0 = encode_fn(jnp.array(u0))

    def step_fn(h, _):
        h_next = discrete_map_step(map_params, h)
        u_next = decode_fn(h_next)
        return h_next, (h_next, u_next)

    h_final, (traj_lat, traj_phys) = jax.lax.scan(
        step_fn, h0, None, length=n_steps
    )
    return np.array(traj_phys), np.array(traj_lat)


# =============================================================================
# Lyapunov Computation
# =============================================================================

def _benettin_qr(step_fn: Callable,
                 x0,
                 n_steps=2000,
                 n_lyap=None,
                 time_per_step=1.0,
                 n_warmup=0):
    """
    Generic Benettin QR Lyapunov spectrum for a differentiable discrete map.

    Args:
        step_fn: differentiable one-step map x_n -> x_{n+1}
        x0: initial state
        n_steps: number of QR accumulation steps
        n_lyap: number of exponents to compute (default: full dimension)
        time_per_step: physical time represented by one map step
        n_warmup: number of map steps to discard before accumulation

    Returns:
        exponents: (n_lyap,) Lyapunov exponents per unit time
    """
    x0_jax = jnp.array(x0, dtype=jnp.float64)
    d = int(x0_jax.shape[0])
    if n_lyap is None:
        n_lyap = d

    step_fn_jit = jax.jit(step_fn)

    if n_warmup > 0:
        def warmup_step(x, _):
            return step_fn_jit(x), None

        x0_jax, _ = jax.lax.scan(warmup_step, x0_jax, None, length=n_warmup)

    Q0 = jnp.eye(d, n_lyap, dtype=jnp.float64)
    log0 = jnp.zeros(n_lyap, dtype=jnp.float64)

    def benettin_step(carry, _):
        x, Q, log_sum = carry

        Q_raw = jax.vmap(
            lambda q: jax.jvp(step_fn_jit, (x,), (q,))[1],
            in_axes=1, out_axes=1
        )(Q)

        x_next = step_fn_jit(x)
        Q_next, R = jnp.linalg.qr(Q_raw)

        # Keep the QR basis orientation stable. If a diagonal entry lands exactly
        # on zero, leave that column unchanged instead of zeroing it out.
        diag_R = jnp.diag(R)
        signs = jnp.where(diag_R < 0, -1.0, 1.0)
        Q_next = Q_next * signs[None, :]
        R = R * signs[:, None]

        safe_diag = jnp.maximum(jnp.abs(jnp.diag(R)), jnp.finfo(jnp.float64).tiny)
        log_next = log_sum + jnp.log(safe_diag)
        return (x_next, Q_next, log_next), None

    (_, _, log_total), _ = jax.lax.scan(
        benettin_step, (x0_jax, Q0, log0), None, length=n_steps
    )
    exponents = np.array(log_total / (n_steps * time_per_step))
    return np.sort(exponents)[::-1]


def compute_latent_lyapunov(ode_params, h0, n_steps=2000, n_lyap=None, dt=0.25,
                            n_warmup=200):
    """
    Benettin QR Lyapunov spectrum for the latent ODE.

    Uses JVP through a fixed-step RK4 latent step, with an optional warmup
    phase so the finite-time estimate is not dominated by the chosen start.
    """
    return _benettin_qr(
        lambda h: rk4_step_latent(ode_params, h, dt),
        h0,
        n_steps=n_steps,
        n_lyap=n_lyap,
        time_per_step=dt,
        n_warmup=n_warmup,
    )


def compute_discrete_map_lyapunov(map_params, h0, n_steps=2000, n_lyap=None,
                                  tau=1.0, n_warmup=200):
    """
    Benettin QR Lyapunov spectrum for the learned discrete latent map.

    Args:
        map_params: discrete latent-map parameters
        h0: initial latent state
        n_steps: number of map steps in the accumulation window
        n_lyap: number of exponents to compute
        tau: physical time represented by one learned map step
        n_warmup: number of map steps to discard before accumulation
    """
    return _benettin_qr(
        lambda h: discrete_map_step(map_params, h),
        h0,
        n_steps=n_steps,
        n_lyap=n_lyap,
        time_per_step=tau,
        n_warmup=n_warmup,
    )


# =============================================================================
# Diagnostics
# =============================================================================

def reconstruction_diagnostics(ae_params, u_test, encode_fn=None, decode_fn=None):
    """
    Compute autoencoder reconstruction diagnostics on test data.

    Returns:
        dict with 'mse', 'rel_error', 'energy_ratio'
    """
    if encode_fn is None:
        encode_fn = lambda u: encode(ae_params, u)
    if decode_fn is None:
        decode_fn = lambda h: decode(ae_params, h)

    u_jax = jnp.array(u_test)
    h_all = jax.vmap(encode_fn)(u_jax)
    u_rec = jax.vmap(decode_fn)(h_all)

    mse = float(jnp.mean((u_rec - u_jax) ** 2))
    rel_err = float(jnp.mean(
        jnp.linalg.norm(u_rec - u_jax, axis=1) / jnp.linalg.norm(u_jax, axis=1)
    ))
    energy_true = float(jnp.mean(jnp.sum(u_jax ** 2, axis=1)))
    energy_rec = float(jnp.mean(jnp.sum(u_rec ** 2, axis=1)))
    energy_ratio = energy_rec / energy_true

    return {
        "mse": mse,
        "rel_error": rel_err,
        "energy_true": energy_true,
        "energy_rec": energy_rec,
        "energy_ratio": energy_ratio,
        "h_variance": np.array(jnp.var(h_all, axis=0)),  # (d,) latent variance
    }


def kaplan_yorke(exponents):
    """Kaplan-Yorke dimension from Lyapunov spectrum."""
    exponents = np.sort(np.asarray(exponents))[::-1]
    cs = np.cumsum(exponents)
    k_arr = np.where(cs < 0)[0]
    if len(k_arr) == 0:
        return float(len(exponents))
    k = k_arr[0]
    if k == 0:
        return 0.0
    return float(k) + cs[k - 1] / abs(exponents[k])


# =============================================================================
# Full Latent NODE Pipeline
# =============================================================================

def run_latent_node_pipeline(traj_train, traj_test, solver, d=8,
                              ae_epochs=500, ode_epochs=500,
                              ae_hidden=128, ae_layers=2,
                              ode_hidden=128, ode_layers=3,
                              batch_size=256, subsample=2,
                              mode='nonlinear', key=None):
    """
    Full pipeline: train AE + latent ODE, run diagnostics.

    Args:
        traj_train: (T_train, N) training trajectory
        traj_test: (T_test, N) test trajectory
        solver: KSSolver instance
        d: latent dimension
        ae_epochs, ode_epochs: training epochs
        ae_hidden, ae_layers: AE architecture
        ode_hidden, ode_layers: latent ODE architecture
        batch_size: training batch size
        subsample: subsampling factor for latent data
        mode: 'nonlinear' (MLP AE) or 'pod' (linear POD AE)
        key: JAX random key

    Returns:
        results dict with all trained params and diagnostics
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    N = traj_train.shape[1]
    k1, k2, k3 = jax.random.split(key, 3)

    results = {"d": d, "mode": mode, "N": N}

    # --- Phase 1: Train or fit AE ---
    if mode == 'nonlinear':
        print(f"\n[Latent NODE d={d}] Phase 1: Training nonlinear AE ({ae_epochs} epochs)...")
        ae_params = init_autoencoder(k1, N=N, d=d,
                                     hidden_ae=ae_hidden, n_layers_ae=ae_layers)
        ae_params, ae_loss_hist = train_autoencoder(
            ae_params, traj_train, n_epochs=ae_epochs, batch_size=batch_size,
            key=k2
        )
        encode_fn = lambda u: encode(ae_params, u)
        decode_fn = lambda h: decode(ae_params, h)
        results["ae_params"] = ae_params
        results["ae_loss_history"] = ae_loss_hist

    elif mode == 'pod':
        print(f"\n[Latent NODE d={d}] Phase 1: Fitting POD AE (d={d} modes)...")
        ae_params = fit_pod_autoencoder(traj_train, d)
        encode_fn = lambda u: pod_encode(ae_params, u)
        decode_fn = lambda h: pod_decode(ae_params, h)
        results["ae_params"] = ae_params
        results["ae_loss_history"] = []

    # AE reconstruction diagnostics
    ae_diag = reconstruction_diagnostics(
        ae_params, traj_test[:1000], encode_fn, decode_fn
    )
    results["ae_diagnostics"] = ae_diag
    print(f"  AE reconstruction: MSE={ae_diag['mse']:.6f}, "
          f"rel_err={ae_diag['rel_error']:.4f}, "
          f"energy_ratio={ae_diag['energy_ratio']:.4f}")

    # --- Phase 2: Prepare latent training data ---
    print(f"\n[Latent NODE d={d}] Phase 2a: Preparing latent training data...")
    if mode == 'nonlinear':
        lat_data = prepare_latent_data_nonlinear(
            traj_train, ae_params, solver, subsample=subsample
        )
    else:
        lat_data = prepare_latent_data_pod(
            traj_train, ae_params, solver, subsample=subsample
        )

    # --- Phase 2b: Train latent ODE ---
    print(f"\n[Latent NODE d={d}] Phase 2b: Training latent ODE ({ode_epochs} epochs)...")
    ode_params = init_latent_ode(k3, d=d, hidden=ode_hidden, n_layers=ode_layers)
    ode_params, ode_loss_hist = train_latent_ode(
        ode_params, lat_data["h"], lat_data["dhdt"],
        n_epochs=ode_epochs, batch_size=batch_size
    )
    results["ode_params"] = ode_params
    results["ode_loss_history"] = ode_loss_hist

    # --- Phase 3: Train discrete map ---
    print(f"\n[Latent NODE d={d}] Phase 3: Training discrete map ({ode_epochs} epochs)...")
    k4 = jax.random.PRNGKey(99)
    map_params = init_discrete_map(k4, d=d, hidden=ode_hidden, n_layers=ode_layers)
    disc_data = prepare_discrete_latent_data(traj_train, encode_fn, subsample=subsample)
    map_params, map_loss_hist = train_discrete_map(
        map_params, disc_data["h_n"], disc_data["h_n1"],
        n_epochs=ode_epochs, batch_size=batch_size
    )
    results["map_params"] = map_params
    results["map_loss_history"] = map_loss_hist

    # --- Rollout diagnostics ---
    print(f"\n[Latent NODE d={d}] Rollout diagnostics...")
    u0_test = traj_test[0]

    # NODE rollout
    try:
        traj_node, traj_lat = rollout_latent_node(
            ae_params, ode_params, u0_test, n_steps=2000, dt=0.25,
            encode_fn=encode_fn, decode_fn=decode_fn
        )
        energy_node = float(np.mean(np.sum(traj_node ** 2, axis=1)))
        results["traj_latent_node"] = traj_node
        results["rollout_node_energy"] = energy_node
        print(f"  NODE rollout: energy={energy_node:.3f}")
    except Exception as e:
        print(f"  NODE rollout failed: {e}")
        results["traj_latent_node"] = None

    # Discrete map rollout
    try:
        traj_map, traj_lat_map = rollout_discrete_map(
            ae_params, map_params, u0_test, n_steps=2000,
            encode_fn=encode_fn, decode_fn=decode_fn
        )
        energy_map = float(np.mean(np.sum(traj_map ** 2, axis=1)))
        results["traj_discrete_map"] = traj_map
        results["rollout_map_energy"] = energy_map
        print(f"  Map rollout: energy={energy_map:.3f}")
    except Exception as e:
        print(f"  Map rollout failed: {e}")
        results["traj_discrete_map"] = None

    # --- Lyapunov spectrum ---
    print(f"\n[Latent NODE d={d}] Computing Lyapunov spectrum in latent space...")
    try:
        h0 = np.array(encode_fn(jnp.array(traj_test[100])))
        le_latent = compute_latent_lyapunov(ode_params, h0, n_steps=2000, dt=0.25)
        dky = kaplan_yorke(le_latent)
        h_ks = float(np.sum(le_latent[le_latent > 0]))
        n_pos = int(np.sum(le_latent > 0))
        results["lyapunov_latent"] = le_latent
        results["dky_latent"] = dky
        results["h_ks_latent"] = h_ks
        results["n_pos_latent"] = n_pos
        print(f"  Latent Lyapunov: L1={le_latent[0]:+.4f}, n_pos={n_pos}, "
              f"D_KY={dky:.2f}, h_KS={h_ks:.4f}")
    except Exception as e:
        print(f"  Lyapunov failed: {e}")
        results["lyapunov_latent"] = None

    return results


if __name__ == "__main__":
    """Quick smoke test of all components."""
    import sys
    sys.path.insert(0, '.')
    from ks_solver import KSSolver

    print("=== latent_node.py smoke test ===")
    N, d = 64, 8
    key = jax.random.PRNGKey(0)

    # Test AE
    k1, k2, k3 = jax.random.split(key, 3)
    ae_params = init_autoencoder(k1, N=N, d=d, hidden_ae=64, n_layers_ae=2)
    u_test = jnp.ones(N) * 0.1
    h = encode(ae_params, u_test)
    u_rec = decode(ae_params, h)
    print(f"AE: u -> h shape {h.shape}, rec shape {u_rec.shape}, "
          f"rec error {float(jnp.mean((u_rec - u_test)**2)):.4f}")

    # Test latent ODE
    ode_params = init_latent_ode(k2, d=d, hidden=64, n_layers=2)
    dhdt = latent_ode_rhs(ode_params, h)
    print(f"Latent ODE: dhdt shape {dhdt.shape}, norm {float(jnp.linalg.norm(dhdt)):.4f}")

    # Test discrete map
    map_params = init_discrete_map(k3, d=d, hidden=64, n_layers=2)
    h_next = discrete_map_step(map_params, h)
    print(f"Discrete map: h_next shape {h_next.shape}")

    # Test POD AE
    traj_dummy = np.random.randn(200, N).astype(np.float64)
    pod = fit_pod_autoencoder(traj_dummy, d)
    h_pod = pod_encode(pod, jnp.array(traj_dummy[0]))
    u_rec_pod = pod_decode(pod, h_pod)
    print(f"POD AE: h shape {h_pod.shape}, rec shape {u_rec_pod.shape}")

    # Test rollout
    solver = KSSolver(L=22.0, N=N, dt=0.25)
    traj_node, traj_lat = rollout_latent_node(
        ae_params, ode_params, traj_dummy[0], n_steps=10, dt=0.25
    )
    print(f"NODE rollout: shape {traj_node.shape}")

    # Test Lyapunov
    h0 = np.zeros(d)
    le = compute_latent_lyapunov(ode_params, h0, n_steps=50, dt=0.25)
    print(f"Lyapunov: {np.round(le, 3)}")

    print("\nAll smoke tests passed.")
