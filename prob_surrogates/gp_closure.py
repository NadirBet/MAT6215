"""
gp_closure.py — Sparse Variational Gaussian Process Closure
=============================================================
Self-contained manual sparse GP implementation (no gpjax dependency).
Operates in POD-reduced latent space to avoid the curse of dimensionality.

Model
-----
Given a deterministic latent mean prediction μ_det(h_t),
the residual is r_t = h_{t+1} - μ_det(h_t).
We fit a GP: r ~ GP(0, k(h, h'))  independently per output dimension.

For d output dimensions we fit d independent GPs sharing the same
inducing points and kernel hyperparameters (tied by default).

Sparse GP (inducing-point / SVGP) formulation
----------------------------------------------
  Inducing inputs  : Z ∈ R^{M × d_in}  (M << N)
  Variational mean : m ∈ R^M
  Variational cov  : L (lower Cholesky of S = LL^T), S ∈ R^{M × M}

  Predictive:
    q(f_*) = N( K_{*u} K_{uu}^{-1} m,
                K_{**} + K_{*u} K_{uu}^{-1}(S - K_{uu}) K_{uu}^{-1} K_{u*} )

  ELBO objective:
    ELBO = Σ_n E_q[log p(r_n | f_n)] - KL[q(u) || p(u)]

  Gaussian likelihood → E_q[log p(r | f)] has analytic form.

Kernel
------
  ARD RBF:  k(h, h') = σ_f² exp( -0.5 Σ_i (h_i - h'_i)² / l_i² )
  + diagonal noise: σ_n²

All JAX — JIT-compiled ELBO for fast training.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial

jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────────────────────────
# Kernel
# ──────────────────────────────────────────────────────────────────────────────

def rbf_kernel(X, Z, log_sf, log_l):
    """
    ARD RBF kernel matrix K(X, Z).

    Args:
        X      : (N, d) inputs
        Z      : (M, d) inducing points
        log_sf : scalar log signal std
        log_l  : (d,) log length scales (ARD)

    Returns:
        K : (N, M)
    """
    l   = jnp.exp(log_l)                           # (d,)
    sf2 = jnp.exp(2 * log_sf)
    X_s = X / l                                    # (N, d)
    Z_s = Z / l                                    # (M, d)
    # squared distances: (N, M)
    diff = X_s[:, None, :] - Z_s[None, :, :]      # (N, M, d)
    sqdist = jnp.sum(diff ** 2, axis=-1)           # (N, M)
    return sf2 * jnp.exp(-0.5 * sqdist)


def rbf_diag(X, log_sf):
    """k(x, x) = sf² for all x."""
    return jnp.exp(2 * log_sf) * jnp.ones(X.shape[0])


# ──────────────────────────────────────────────────────────────────────────────
# SVGP parameters
# ──────────────────────────────────────────────────────────────────────────────

def init_svgp(key, X_inducing, d_out, log_sf_init=0.0, log_l_init=0.0,
              log_noise_init=-1.0):
    """
    Initialise sparse GP variational parameters for d_out independent GPs.

    Args:
        key         : JAX PRNGKey
        X_inducing  : (M, d_in) inducing point locations
        d_out       : number of independent output dimensions
        log_sf_init : initial log signal std (shared)
        log_l_init  : initial log length scale (shared per dimension → broadcast)
        log_noise_init : initial log noise std

    Returns:
        params dict
    """
    M, d_in = X_inducing.shape
    params = {
        "Z"         : jnp.array(X_inducing),      # (M, d_in)
        "log_sf"    : jnp.full((), log_sf_init),
        "log_l"     : jnp.full((d_in,), log_l_init),
        "log_noise" : jnp.full((), log_noise_init),
        # Variational parameters: one set per output dimension
        "m"         : jnp.zeros((d_out, M)),       # variational means
        "L_raw"     : jnp.stack([jnp.eye(M) * 0.1
                                 for _ in range(d_out)]),  # (d_out, M, M)
    }
    return params


def tril(L_raw):
    """Extract lower triangular from raw (zeros strictly upper)."""
    return jnp.tril(L_raw)


# ──────────────────────────────────────────────────────────────────────────────
# ELBO
# ──────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=())
def svgp_elbo(params, X_batch, R_batch):
    """
    Gaussian SVGP ELBO (lower bound on log marginal likelihood).
    Scaled to full dataset size N / batch_size (caller must pass scale).

    Args:
        params  : dict from init_svgp
        X_batch : (B, d_in) batch of input latent states
        R_batch : (B, d_out) batch of target residuals

    Returns:
        scalar ELBO (negated for minimization → returns -ELBO)
    """
    Z       = params["Z"]
    log_sf  = params["log_sf"]
    log_l   = params["log_l"]
    log_sn  = params["log_noise"]
    m       = params["m"]          # (d_out, M)
    L_raw   = params["L_raw"]      # (d_out, M, M)
    L       = jax.vmap(tril)(L_raw)   # (d_out, M, M)

    sn2 = jnp.exp(2 * log_sn)
    M   = Z.shape[0]
    B   = X_batch.shape[0]

    # Kernel matrices
    K_uu = rbf_kernel(Z, Z, log_sf, log_l) + 1e-6 * jnp.eye(M)  # (M, M)
    K_fu = rbf_kernel(X_batch, Z, log_sf, log_l)                  # (B, M)
    k_ff_diag = rbf_diag(X_batch, log_sf)                         # (B,)

    # Cholesky of K_uu
    L_uu = jnp.linalg.cholesky(K_uu)   # (M, M)

    # A = L_uu^{-1} K_{uf}  →  K_{fu} K_{uu}^{-1} K_{uf} via A^T A
    A = jax.scipy.linalg.solve_triangular(L_uu, K_fu.T, lower=True)  # (M, B)

    # q(f_*) mean: K_{fu} K_{uu}^{-1} m  — per output dim
    # m shape: (d_out, M)
    alpha_m = jax.scipy.linalg.solve_triangular(
        L_uu, m.T, lower=True)              # (M, d_out)
    q_mean = (A.T @ alpha_m).T              # (d_out, B)

    # q(f_*) variance (diagonal): k_ff - A^T A + A^T S A  per output dim
    # S = L L^T  →  A^T S A = A^T L L^T A = ||L^T A||^2
    base_var = k_ff_diag - jnp.sum(A ** 2, axis=0)  # (B,) prior correction

    # Per output dim: trace term and KL
    def per_output(m_d, L_d):
        # L_d : (M, M) lower cholesky of S_d
        S_d = L_d @ L_d.T                   # (M, M)
        # q variance: base_var + diag(A^T S_d A)
        LA  = L_d.T @ A                     # (M, B)
        q_var_d = base_var + jnp.sum(LA**2, axis=0)   # (B,)
        q_var_d = jnp.clip(q_var_d, 1e-12, None)

        # Expected log-likelihood (Gaussian)
        alpha_d = jax.scipy.linalg.solve_triangular(L_uu, m_d, lower=True)
        q_mean_d = A.T @ alpha_d            # (B,)
        ell = -0.5 * (jnp.log(2*jnp.pi*sn2)
                      + (R_batch[:, 0] - q_mean_d)**2 / sn2
                      + q_var_d / sn2)      # (B,)

        # KL[q(u) || p(u)] = 0.5 * [ tr(K_{uu}^{-1} S) + m^T K_{uu}^{-1} m
        #                             - M + log |K_{uu}| - log |S| ]
        L_uu_inv_L_d = jax.scipy.linalg.solve_triangular(L_uu, L_d, lower=True)
        tr_term  = jnp.sum(L_uu_inv_L_d**2)
        m_term   = jnp.sum(alpha_d**2)
        logdet_K = 2 * jnp.sum(jnp.log(jnp.diag(L_uu)))
        logdet_S = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(L_d)) + 1e-12))
        kl = 0.5 * (tr_term + m_term - M + logdet_K - logdet_S)

        return jnp.sum(ell), kl

    # vmap over output dims, passing per-output residuals
    def per_output_wrapper(args):
        m_d, L_d, r_d = args
        # rebuild R_batch column for this output: shape (B, 1)
        return per_output(m_d, L_d)

    ells = []
    kls  = []
    for o in range(m.shape[0]):
        # Temporarily swap R_batch column
        R_col = R_batch[:, o:o+1]
        # Redefine per_output inline to capture R_col
        alpha_o = jax.scipy.linalg.solve_triangular(L_uu, m[o], lower=True)
        q_mean_o = A.T @ alpha_o                       # (B,)
        LA_o     = L[o].T @ A                          # (M, B)
        q_var_o  = jnp.clip(base_var + jnp.sum(LA_o**2, axis=0), 1e-12, None)

        ell_o = jnp.sum(-0.5 * (jnp.log(2*jnp.pi*sn2)
                                 + (R_batch[:, o] - q_mean_o)**2 / sn2
                                 + q_var_o / sn2))

        L_uu_inv_L_o = jax.scipy.linalg.solve_triangular(L_uu, L[o], lower=True)
        tr_o    = jnp.sum(L_uu_inv_L_o**2)
        m_term_o = jnp.sum(alpha_o**2)
        logdet_K = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(L_uu)) + 1e-12))
        logdet_S_o = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(L[o])) + 1e-12))
        kl_o = 0.5 * (tr_o + m_term_o - M + logdet_K - logdet_S_o)

        ells.append(ell_o)
        kls.append(kl_o)

    total_ell = sum(ells)
    total_kl  = sum(kls)
    return -(total_ell - total_kl)   # negative ELBO for minimization


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def svgp_predict(params, X_new):
    """
    Predictive mean and variance at new inputs.

    Returns:
        pred_mean : (N_new, d_out)
        pred_var  : (N_new, d_out)  (diagonal predictive variance)
    """
    Z      = params["Z"]
    log_sf = params["log_sf"]
    log_l  = params["log_l"]
    log_sn = params["log_noise"]
    m      = params["m"]           # (d_out, M)
    L_raw  = params["L_raw"]
    L      = jax.vmap(tril)(L_raw)

    sn2 = jnp.exp(2 * log_sn)
    M   = Z.shape[0]

    K_uu     = rbf_kernel(Z, Z, log_sf, log_l) + 1e-6 * jnp.eye(M)
    K_su     = rbf_kernel(X_new, Z, log_sf, log_l)
    k_ss_diag = rbf_diag(X_new, log_sf)

    L_uu = jnp.linalg.cholesky(K_uu)
    A    = jax.scipy.linalg.solve_triangular(L_uu, K_su.T, lower=True)  # (M, N_new)
    base_var = k_ss_diag - jnp.sum(A**2, axis=0)                        # (N_new,)

    pred_means = []
    pred_vars  = []
    for o in range(m.shape[0]):
        alpha_o  = jax.scipy.linalg.solve_triangular(L_uu, m[o], lower=True)
        mu_o     = A.T @ alpha_o                    # (N_new,)
        LA_o     = L[o].T @ A                       # (M, N_new)
        var_o    = base_var + jnp.sum(LA_o**2, axis=0) + sn2  # (N_new,)
        pred_means.append(mu_o)
        pred_vars.append(var_o)

    return (jnp.stack(pred_means, axis=1),      # (N_new, d_out)
            jnp.stack(pred_vars,  axis=1))       # (N_new, d_out)


def svgp_sample(params, X_new, key, n_samples=1):
    """
    Draw posterior samples at X_new.

    Returns:
        samples : (n_samples, N_new, d_out)
    """
    mean, var = svgp_predict(params, X_new)
    std  = jnp.sqrt(jnp.clip(var, 1e-12, None))
    eps  = jax.random.normal(key, shape=(n_samples,) + mean.shape)
    return mean[None, :, :] + std[None, :, :] * eps


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_svgp(params, H_t, R_t, n_epochs=300, batch_size=256, lr=1e-3,
               key=None, verbose=True):
    """
    Train SVGP by maximizing ELBO via Adam.

    Args:
        params     : init_svgp output
        H_t        : (N, d_in) input latent states
        R_t        : (N, d_out) target residuals
        n_epochs   : training epochs
        batch_size : mini-batch size
        lr         : Adam learning rate
        key        : JAX PRNGKey

    Returns:
        trained params, elbo_history (list of floats)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    N = H_t.shape[0]
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    elbo_history = []

    @jax.jit
    def step(params, opt_state, X_b, R_b):
        loss, grads = jax.value_and_grad(svgp_elbo)(params, X_b, R_b)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, N - batch_size, batch_size):
            idx = perm[i:i + batch_size]
            params, opt_state, loss = step(
                params, opt_state,
                H_t[idx], R_t[idx])
            epoch_loss += float(loss)
            n_batches  += 1
        avg = epoch_loss / max(n_batches, 1)
        elbo_history.append(avg)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    GP epoch {epoch+1}/{n_epochs}: -ELBO={avg:.4f}")

    return params, elbo_history
