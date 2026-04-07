"""
lyapunov.py -- Lyapunov Spectrum and Covariant Lyapunov Vectors
==============================================================
Implements two algorithms:

1. Benettin et al. (1980) -- Forward QR algorithm
   Computes the full Lyapunov spectrum via repeated QR decomposition
   of the tangent map product. Gives L1 >= L2 >= ... >= LN.

2. Ginelli et al. (2007, 2013) -- CLV algorithm
   Forward pass: same as Benettin, stores Q matrices.
   Backward pass: evolves upper triangular coefficients backward.
   CLV_i(t) = Q(t) * c_i(t) where c_i is the i-th column of C(t).
   These are coordinate-independent (covariant) unlike Gram-Schmidt vectors.

CRITICAL: Tangent evolution uses JVP through the ETD-RK4 discrete step,
NOT a naive Euler/RK4 on the continuous RHS. The KSE is stiff (largest
negative eigenvalue ~ -(2pi*N/2/L)^4), so any explicit method on the
continuous-time tangent equation will blow up. JVP through the stable
discrete map automatically inherits the ETD stability.

Reference:
- Ginelli 2007: arXiv:0706.0510 -- original CLV algorithm
- Ginelli 2013: arXiv:1212.3961 -- full review with convergence proofs
- Ozalp 2024: arXiv:2410.00480 -- CLV angles as surrogate diagnostic
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)


def _physical_discrete_step(solver, u_phys: jnp.ndarray) -> jnp.ndarray:
    """
    One ETD-RK4 step in physical space.
    Used as the differentiable map for JVP-based tangent evolution.
    """
    u_hat = jnp.fft.fft(u_phys)
    u_hat_next = solver.step(u_hat)
    return jnp.fft.ifft(u_hat_next).real


def compute_lyapunov_spectrum(solver, u0: jnp.ndarray,
                               n_steps: int = 4000,
                               reortho_every: int = 1,
                               n_modes: int = 20) -> tuple:
    """
    Compute Lyapunov spectrum via Benettin's QR algorithm.
    Tangent evolution uses JVP through the ETD-RK4 discrete map.

    Args:
        solver: KSSolver instance (provides step() method)
        u0: Initial condition in physical space, shape (N,)
        n_steps: number of integration steps (each step = solver.dt)
        reortho_every: QR reorthogonalization every this many steps
        n_modes: number of Lyapunov exponents to compute

    Returns:
        (lyapunov_exponents, lyapunov_history)
        lyapunov_exponents: shape (n_modes,), sorted descending
        lyapunov_history: shape (n_checkpoints, n_modes), running average
    """
    N = u0.shape[0]
    discrete_step = partial(_physical_discrete_step, solver)

    # Initialize orthonormal frame in physical space
    Q = jnp.eye(N, n_modes)
    u = jnp.array(u0, dtype=jnp.float64)

    log_stretches = np.zeros(n_modes)
    history = []
    orth_count = 0

    for i in range(n_steps):
        # Advance state
        u = discrete_step(u)

        # Advance each tangent vector via JVP of discrete map at PREVIOUS u
        # Note: u was updated above, so we recompute with the old u
        # We need to advance tangents BEFORE updating state:
        # Fix: use carry pattern below
        pass

    # Redo with correct ordering: advance tangent at current u, then advance u
    Q = jnp.eye(N, n_modes)
    u = jnp.array(u0, dtype=jnp.float64)
    log_stretches = np.zeros(n_modes)
    history = []
    orth_count = 0

    for i in range(n_steps):
        # Advance each tangent via JVP of discrete map at current u
        def tangent_jvp(q):
            _, dq = jax.jvp(discrete_step, (u,), (q,))
            return dq

        Q_raw = jax.vmap(tangent_jvp, in_axes=1, out_axes=1)(Q)

        # Advance state
        u = discrete_step(u)

        if (i + 1) % reortho_every == 0:
            Q, R = jnp.linalg.qr(Q_raw)
            diag_R = jnp.diag(R)
            # Flip sign of columns where R diagonal is negative
            signs = jnp.sign(diag_R)
            Q = Q * signs[None, :]
            log_stretches += np.array(jnp.log(jnp.abs(diag_R)))
            orth_count += 1

            if (i + 1) % max(n_steps // 20, 1) == 0:
                current = log_stretches / (orth_count * reortho_every * solver.dt)
                history.append(current.copy())
                print(f"  Step {i+1}/{n_steps}: L1={current[0]:.4f}, "
                      f"L2={current[1]:.4f}, Llast={current[-1]:.4f}")
        else:
            Q = Q_raw

    exponents = log_stretches / (orth_count * reortho_every * solver.dt)
    return np.array(exponents), np.array(history)


def compute_lyapunov_spectrum_jit(solver, u0: jnp.ndarray,
                                   n_steps: int = 2000,
                                   n_modes: int = 20) -> tuple:
    """
    JIT-compiled Lyapunov computation using lax.scan.
    Uses JVP through discrete ETD-RK4 step.
    More memory efficient for large n_steps.

    Returns:
        (exponents, u_final, Q_final)
    """
    N = u0.shape[0]
    discrete_step = partial(_physical_discrete_step, solver)

    Q0 = jnp.eye(N, n_modes)
    u0_jax = jnp.array(u0, dtype=jnp.float64)
    log0 = jnp.zeros(n_modes)

    def step_fn(carry, _):
        u, Q, log_sum = carry

        # Advance tangent frame via JVP of discrete map
        Q_raw = jax.vmap(
            lambda q: jax.jvp(discrete_step, (u,), (q,))[1],
            in_axes=1, out_axes=1
        )(Q)

        # Advance state
        u_next = discrete_step(u)

        # QR orthogonalization
        Q_next, R = jnp.linalg.qr(Q_raw)
        # Fix sign convention
        signs = jnp.sign(jnp.diag(R))
        Q_next = Q_next * signs[None, :]
        R = R * signs[:, None]

        log_next = log_sum + jnp.log(jnp.abs(jnp.diag(R)))
        return (u_next, Q_next, log_next), jnp.diag(R)

    (u_final, Q_final, log_total), r_diags = jax.lax.scan(
        step_fn, (u0_jax, Q0, log0), None, length=n_steps
    )

    exponents = log_total / (n_steps * solver.dt)
    return np.array(exponents), np.array(u_final), np.array(Q_final)


class CLVComputer:
    """
    Ginelli algorithm for Covariant Lyapunov Vectors.

    Algorithm (Ginelli 2007, 2013):
    Phase 1 (Forward): Run Benettin algorithm, store all Q and R matrices.
    Phase 2 (Backward): Initialize C = random upper triangular.
                        Iterate backward: C_{n-1} = R_n^{-1} @ C_n
                        Normalize columns of C.
    Result: CLV_i(t) = Q(t) @ c_i(t)

    CLV angles: phi_{i,j}(t) = arccos(|CLV_i(t) . CLV_j(t)|)
    Near-zero angles indicate tangencies between stable/unstable manifolds.
    """

    def __init__(self, solver, n_clv: int = 16):
        self.solver = solver
        self.n_clv = n_clv
        self._discrete_step = partial(_physical_discrete_step, solver)

    def forward_pass(self, u0: jnp.ndarray, n_steps: int) -> tuple:
        """
        Forward Benettin pass. Stores Q and R matrices for CLV computation.

        Returns:
            u_final: final state (physical space)
            Q_history: (n_steps, N, n_clv) orthonormal frames
            R_history: (n_steps, n_clv, n_clv) upper triangular matrices
            exponents: Lyapunov exponents from this pass
        """
        N = u0.shape[0]
        n_clv = self.n_clv
        discrete_step = self._discrete_step

        Q = np.eye(N, n_clv)
        u = np.array(u0, dtype=np.float64)

        Q_history = np.zeros((n_steps, N, n_clv))
        R_history = np.zeros((n_steps, n_clv, n_clv))
        log_stretches = np.zeros(n_clv)

        print(f"  CLV forward pass: {n_steps} steps, {n_clv} vectors...")
        for i in range(n_steps):
            u_jax = jnp.array(u)

            # Advance tangent frame via JVP of discrete map
            Q_jax = jnp.array(Q)
            Q_raw = np.array(jax.vmap(
                lambda q: jax.jvp(discrete_step, (u_jax,), (q,))[1],
                in_axes=1, out_axes=1
            )(Q_jax))

            # Advance state with ETD-RK4
            u = np.array(discrete_step(u_jax))

            # QR decompose
            Q, R = np.linalg.qr(Q_raw)
            # Fix sign convention
            signs = np.sign(np.diag(R))
            Q = Q * signs[None, :]
            R = R * signs[:, None]

            Q_history[i] = Q
            R_history[i] = R
            log_stretches += np.log(np.abs(np.diag(R)))

            if (i + 1) % max(n_steps // 5, 1) == 0:
                current = log_stretches / ((i + 1) * self.solver.dt)
                print(f"    Forward {i+1}/{n_steps}: L1={current[0]:.4f}")

        exponents = log_stretches / (n_steps * self.solver.dt)
        return u, Q_history, R_history, exponents

    def backward_pass(self, Q_history: np.ndarray,
                      R_history: np.ndarray,
                      key: jax.random.PRNGKey) -> np.ndarray:
        """
        Backward Ginelli pass to extract CLVs.

        Args:
            Q_history: (n_steps, N, n_clv) from forward pass
            R_history: (n_steps, n_clv, n_clv) upper triangular matrices
            key: JAX random key for initialization

        Returns:
            CLVs: (n_steps, N, n_clv)
        """
        n_steps, N, n_clv = Q_history.shape
        np.random.seed(int(key[0]))

        # Initialize with random upper triangular matrix
        C = np.random.randn(n_clv, n_clv)
        C = np.triu(C)
        C = C / np.linalg.norm(C, axis=0, keepdims=True)

        CLVs = np.zeros((n_steps, N, n_clv))

        print(f"  CLV backward pass: {n_steps} steps...")
        for i in range(n_steps - 1, -1, -1):
            R = R_history[i]
            C = np.linalg.solve(R, C)
            norms = np.linalg.norm(C, axis=0, keepdims=True)
            C = C / (norms + 1e-14)
            CLVs[i] = Q_history[i] @ C

            if i % max(n_steps // 5, 1) == 0:
                print(f"    Backward {n_steps-i}/{n_steps}")

        return CLVs

    def compute_clv_angles(self, CLVs: np.ndarray,
                           pairs: list = None) -> np.ndarray:
        """
        Compute angles between pairs of CLVs at each time step.

        Args:
            CLVs: (n_steps, N, n_clv)
            pairs: list of (i, j) index pairs. Default: adjacent pairs.

        Returns:
            angles: (n_steps, n_pairs) in degrees
        """
        n_steps, N, n_clv = CLVs.shape

        if pairs is None:
            pairs = [(i, i + 1) for i in range(n_clv - 1)]

        angles = np.zeros((n_steps, len(pairs)))
        for t in range(n_steps):
            for p_idx, (i, j) in enumerate(pairs):
                vi = CLVs[t, :, i]
                vj = CLVs[t, :, j]
                cos_a = np.abs(np.dot(vi, vj)) / (
                    np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-14
                )
                cos_a = np.clip(cos_a, 0, 1)
                angles[t, p_idx] = np.degrees(np.arccos(cos_a))

        return angles

    def run(self, u0: jnp.ndarray, n_steps: int,
            key: jax.random.PRNGKey) -> dict:
        """Full CLV computation pipeline."""
        u_final, Q_hist, R_hist, exponents = self.forward_pass(u0, n_steps)
        CLVs = self.backward_pass(Q_hist, R_hist, key)

        n_pos = int(np.sum(exponents > 0))
        key_pair = [(n_pos - 1, n_pos)] if n_pos > 0 else [(0, 1)]
        all_pairs = [(i, i + 1) for i in range(min(self.n_clv - 1, 8))]

        angles = self.compute_clv_angles(CLVs, all_pairs)
        key_angles = self.compute_clv_angles(CLVs, key_pair)

        return {
            "exponents": exponents,
            "CLVs": CLVs,
            "angles": angles,
            "key_angles": key_angles,
            "n_positive": n_pos,
            "pairs": all_pairs,
            "key_pair": key_pair,
        }


def kaplan_yorke_dimension(exponents: np.ndarray) -> float:
    """
    Kaplan-Yorke (Lyapunov) dimension.
    D_KY = k + sum(L_1,...,L_k) / |L_{k+1}|
    where k is the largest index such that sum(L_1,...,L_k) >= 0.
    """
    cumsum = np.cumsum(exponents)
    # Find last index where cumulative sum is still non-negative
    pos_mask = cumsum >= 0
    if not np.any(pos_mask):
        return 0.0
    k = int(np.where(pos_mask)[0][-1]) + 1  # 1-based count
    if k >= len(exponents):
        return float(len(exponents))
    return k + cumsum[k - 1] / np.abs(exponents[k])


def ks_entropy(exponents: np.ndarray) -> float:
    """
    Kolmogorov-Sinai entropy via Pesin's formula.
    h_KS = sum of all positive Lyapunov exponents.
    """
    return float(np.sum(exponents[exponents > 0]))


def lyapunov_summary(exponents: np.ndarray, label: str = "System") -> dict:
    """Print and return summary of Lyapunov analysis."""
    n_pos = int(np.sum(exponents > 0))
    n_zero = int(np.sum(np.abs(exponents) < 0.01))
    n_neg = int(np.sum(exponents < -0.01))
    dky = kaplan_yorke_dimension(exponents)
    hks = ks_entropy(exponents)
    lyap_time = 1.0 / exponents[0] if exponents[0] > 0 else float('inf')

    summary = {
        "label": label,
        "n_positive": n_pos,
        "n_zero": n_zero,
        "n_negative": n_neg,
        "lambda_1": float(exponents[0]),
        "lyapunov_time": float(lyap_time),
        "kaplan_yorke_dim": float(dky),
        "ks_entropy": float(hks),
        "exponents": exponents,
    }

    print(f"\n{'='*50}")
    print(f"Lyapunov Analysis: {label}")
    print(f"{'='*50}")
    print(f"  Leading exponent L1 = {exponents[0]:.4f}")
    print(f"  Lyapunov time TL    = {lyap_time:.2f}")
    print(f"  Positive exponents  = {n_pos}")
    print(f"  Zero exponents      = {n_zero}")
    print(f"  KY dimension        = {dky:.2f}")
    print(f"  KS entropy          = {hks:.4f}")
    print(f"  First 8 exponents   = {np.array2string(exponents[:8], precision=4)}")
    return summary


if __name__ == "__main__":
    from ks_solver import KSSolver

    solver = KSSolver(L=22.0, N=64, dt=0.25)
    key = jax.random.PRNGKey(0)

    print("Generating initial condition on attractor...")
    u0_hat = solver.random_ic(key)
    u0_hat = solver.warmup(u0_hat, n_warmup=2000)
    u0 = jnp.fft.ifft(u0_hat).real

    print("Computing Lyapunov spectrum (JIT version)...")
    exponents, u_final, Q_final = compute_lyapunov_spectrum_jit(
        solver, u0, n_steps=500, n_modes=20
    )
    summary = lyapunov_summary(exponents, label="True KSE (N=20 modes)")
    import os; os.makedirs("data", exist_ok=True)
    np.save("data/lyapunov_exponents_true.npy", exponents)
    print("Saved to data/lyapunov_exponents_true.npy")
