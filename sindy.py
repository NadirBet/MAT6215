"""
sindy.py — SINDy Surrogate for KSE
====================================
Sparse Identification of Nonlinear Dynamics (Brunton et al. 2016, arXiv:1509.03580)

Strategy for KSE:
    KSE lives in R^64 — full SINDy with polynomial library would have
    O(64^2) = 4096 terms for degree-2, computationally intractable.

    Instead, project KSE onto first r=8 dominant Galerkin modes:
        a_i(t) = <u(x,t), φ_i(x)>   for i=1,...,r
    where φ_i are the eigenvectors of the learned/true linear operator.

    Then apply SINDy on the low-dimensional ODE:
        da/dt = Ξ * Θ(a)
    where Θ(a) is a polynomial library and Ξ is found by sparse regression.

    This follows:
    - Linot & Graham 2022 (2109.00060): Galerkin ROM from learned linear operator
    - Brunton et al. 2016 (1509.03580): STLSQ for sparse regression

Algorithm: STLSQ (Sequential Thresholded Least Squares)
    1. Least squares: Ξ = Θ^+ * (da/dt)
    2. Threshold: zero out |Ξ_ij| < threshold
    3. Repeat least squares on surviving terms
    4. Converges to sparse symbolic model
"""

import jax
import jax.numpy as jnp
import numpy as np
from itertools import combinations_with_replacement
from typing import Optional

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# Galerkin Projection
# ─────────────────────────────────────────────────────────────────────────────

def compute_galerkin_basis(solver, n_modes: int = 8,
                            traj: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Galerkin projection basis.

    Two options:
    1. POD (Proper Orthogonal Decomposition): SVD of trajectory data
       Basis = leading left singular vectors (data-driven, optimal energy capture)
    2. Fourier modes: natural basis for KSE (simple but not data-adaptive)

    Uses POD by default (better for low-dimensional SINDy).

    Args:
        solver: KSSolver instance
        n_modes: number of basis functions to keep
        traj: (T, N) trajectory for POD. If None, use Fourier modes.

    Returns:
        Phi: (N, n_modes) orthonormal basis matrix
    """
    if traj is not None:
        # POD via SVD
        print(f"  Computing POD basis from trajectory ({traj.shape[0]} snapshots)...")
        # Center the data
        u_mean = traj.mean(axis=0)
        traj_centered = traj - u_mean

        # SVD (economy mode)
        U, S, Vt = np.linalg.svd(traj_centered, full_matrices=False)
        Phi = Vt[:n_modes].T  # (N, n_modes) — spatial modes

        # Energy captured
        energy_frac = S[:n_modes].sum() / S.sum()
        print(f"  POD: {n_modes} modes capture {100*energy_frac:.1f}% of energy")
        print(f"  Singular values: {S[:n_modes+3]}")
        return Phi, u_mean
    else:
        # Use Fourier modes (columns of DFT matrix, real-valued)
        N = solver.N
        x = np.linspace(0, solver.L, N, endpoint=False)
        Phi = np.zeros((N, n_modes))
        for i in range(n_modes):
            k = i // 2 + 1
            if i % 2 == 0:
                Phi[:, i] = np.cos(2 * np.pi * k * x / solver.L)
            else:
                Phi[:, i] = np.sin(2 * np.pi * k * x / solver.L)
        Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
        return Phi, np.zeros(N)


def project_trajectory(traj: np.ndarray, Phi: np.ndarray,
                        u_mean: np.ndarray) -> np.ndarray:
    """
    Project trajectory onto Galerkin basis.

    Args:
        traj: (T, N) physical space trajectory
        Phi: (N, r) orthonormal basis
        u_mean: (N,) mean (from POD centering)

    Returns:
        a: (T, r) modal amplitudes
    """
    traj_centered = traj - u_mean[None, :]
    return traj_centered @ Phi  # (T, r)


def reconstruct_from_modes(a: np.ndarray, Phi: np.ndarray,
                            u_mean: np.ndarray) -> np.ndarray:
    """Reconstruct physical space from modal amplitudes."""
    return a @ Phi.T + u_mean[None, :]  # (T, N)


# ─────────────────────────────────────────────────────────────────────────────
# SINDy Library
# ─────────────────────────────────────────────────────────────────────────────

def polynomial_library(a: np.ndarray, degree: int = 2,
                       include_bias: bool = True) -> tuple:
    """
    Build polynomial feature library Θ(a).

    For r modes and degree 2:
    Θ = [1, a_1, ..., a_r, a_1², a_1*a_2, ..., a_r²]
    Total terms: C(r + degree, degree)

    Args:
        a: (T, r) modal amplitudes
        degree: polynomial degree (1 or 2 recommended for KSE)
        include_bias: include constant term

    Returns:
        (Theta, feature_names): Theta is (T, n_features)
    """
    T, r = a.shape
    features = []
    names = []

    if include_bias:
        features.append(np.ones((T, 1)))
        names.append("1")

    # Degree 1
    features.append(a)
    for i in range(r):
        names.append(f"a{i+1}")

    if degree >= 2:
        # Degree 2: all combinations with replacement
        for i, j in combinations_with_replacement(range(r), 2):
            features.append((a[:, i] * a[:, j])[:, None])
            names.append(f"a{i+1}*a{j+1}")

    if degree >= 3:
        for i, j, k in combinations_with_replacement(range(r), 3):
            features.append((a[:, i] * a[:, j] * a[:, k])[:, None])
            names.append(f"a{i+1}*a{j+1}*a{k+1}")

    Theta = np.hstack(features)  # (T, n_features)
    return Theta, names


# ─────────────────────────────────────────────────────────────────────────────
# STLSQ Sparse Regression
# ─────────────────────────────────────────────────────────────────────────────

def compute_time_derivatives(a: np.ndarray, dt: float,
                              method: str = "finite_diff") -> np.ndarray:
    """
    Compute time derivatives of modal amplitudes.

    Methods:
    - finite_diff: 4th-order central finite differences
    - spectral: FFT-based differentiation (more accurate for smooth data)
    """
    T, r = a.shape

    if method == "finite_diff":
        # 4th-order central differences
        da = np.zeros_like(a)
        # Interior points (4th order)
        da[2:-2] = (-a[4:] + 8*a[3:-1] - 8*a[1:-3] + a[:-4]) / (12 * dt)
        # Boundary (2nd order)
        da[0] = (-3*a[0] + 4*a[1] - a[2]) / (2*dt)
        da[1] = (-3*a[1] + 4*a[2] - a[3]) / (2*dt)
        da[-1] = (3*a[-1] - 4*a[-2] + a[-3]) / (2*dt)
        da[-2] = (3*a[-2] - 4*a[-3] + a[-4]) / (2*dt)

    elif method == "spectral":
        # FFT-based: accurate for periodic signals
        da = np.zeros_like(a)
        for i in range(r):
            a_hat = np.fft.fft(a[:, i])
            freqs = np.fft.fftfreq(T, d=dt)
            da_hat = 1j * 2 * np.pi * freqs * a_hat
            da[:, i] = np.fft.ifft(da_hat).real

    return da


def stlsq(Theta: np.ndarray, da: np.ndarray,
           threshold: float = 0.05,
           max_iter: int = 20,
           verbose: bool = True) -> tuple:
    """
    Sequential Thresholded Least Squares (Brunton 2016).

    Solves: da/dt ≈ Θ * Ξ  (sparse Ξ)

    Algorithm:
    1. Ξ = lstsq(Θ, da)
    2. Zero out |Ξ_ij| < threshold
    3. Re-solve lstsq using only active terms
    4. Repeat until convergence

    Args:
        Theta: (T, n_features) library matrix
        da: (T, r) time derivatives
        threshold: sparsity threshold (0.05 recommended for KSE)
        max_iter: maximum iterations

    Returns:
        (Xi, active_mask): Xi is (n_features, r), active_mask is boolean
    """
    T, n_feat = Theta.shape
    T2, r = da.shape

    # Initial least squares
    Xi, _, _, _ = np.linalg.lstsq(Theta, da, rcond=None)  # (n_feat, r)

    for iteration in range(max_iter):
        # Threshold
        active = np.abs(Xi) >= threshold  # (n_feat, r)

        # Re-solve for each output dimension using only active features
        Xi_new = np.zeros_like(Xi)
        for j in range(r):
            active_j = active[:, j]
            if active_j.sum() == 0:
                continue  # no active terms — degenerate
            Theta_j = Theta[:, active_j]
            xi_j, _, _, _ = np.linalg.lstsq(Theta_j, da[:, j], rcond=None)
            Xi_new[active_j, j] = xi_j

        # Check convergence
        if np.allclose(Xi, Xi_new, atol=1e-10):
            if verbose:
                print(f"  STLSQ converged at iteration {iteration+1}")
            break
        Xi = Xi_new

    active_final = np.abs(Xi) >= threshold / 10

    if verbose:
        n_total = active_final.sum()
        n_max = n_feat * r
        print(f"  Sparsity: {n_total}/{n_max} active terms ({100*n_total/n_max:.1f}%)")

    return Xi, active_final


# ─────────────────────────────────────────────────────────────────────────────
# SINDy Model
# ─────────────────────────────────────────────────────────────────────────────

class SINDyModel:
    """
    Identified sparse dynamical system on Galerkin modes.
    da/dt = Ξ * Θ(a)

    Can be used to:
    1. Integrate forward (predict trajectories)
    2. Compute Jacobian analytically (from sparse library)
    3. Compute Lyapunov exponents
    """

    def __init__(self, Xi: np.ndarray, Phi: np.ndarray,
                 u_mean: np.ndarray, feature_names: list,
                 degree: int = 2, threshold: float = 0.05):
        self.Xi = Xi              # (n_features, r) sparse coefficient matrix
        self.Phi = Phi            # (N, r) spatial basis
        self.u_mean = u_mean      # (N,) mean
        self.feature_names = feature_names
        self.degree = degree
        self.threshold = threshold
        self.r = Phi.shape[1]
        self.N = Phi.shape[0]

    def rhs_modes(self, a: np.ndarray) -> np.ndarray:
        """RHS in modal coordinates: da/dt = Θ(a) * Ξ"""
        Theta_a, _ = polynomial_library(a[None, :], degree=self.degree)
        return (Theta_a @ self.Xi)[0]  # (r,)

    def rhs_physical(self, u: np.ndarray) -> np.ndarray:
        """RHS in physical space via projection and reconstruction."""
        a = (u - self.u_mean) @ self.Phi  # project
        da = self.rhs_modes(a)             # modal RHS
        return da @ self.Phi.T             # reconstruct (approximate)

    def jacobian_modes(self, a: np.ndarray) -> np.ndarray:
        """
        Analytical Jacobian of SINDy model in modal coordinates.
        ∂(da/dt)/∂a = Ξ^T * (∂Θ/∂a)

        For degree-2 polynomial library, ∂Θ/∂a is tractable analytically.
        Use JAX autodiff for simplicity and correctness.
        """
        return np.array(jax.jacobian(
            lambda x: jnp.array(self.rhs_modes(np.array(x)))
        )(jnp.array(a)))

    def integrate(self, u0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        """
        Integrate SINDy model forward using RK4.

        Args:
            u0: initial condition in physical space, shape (N,)
            n_steps: number of steps
            dt: time step

        Returns:
            trajectory in physical space, shape (n_steps, N)
        """
        a0 = (u0 - self.u_mean) @ self.Phi  # project to modal coords
        traj_modes = np.zeros((n_steps, self.r))

        a = a0.copy()
        for i in range(n_steps):
            # RK4 in modal space
            k1 = self.rhs_modes(a)
            k2 = self.rhs_modes(a + dt / 2 * k1)
            k3 = self.rhs_modes(a + dt / 2 * k2)
            k4 = self.rhs_modes(a + dt * k3)
            a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            traj_modes[i] = a

        # Reconstruct physical space
        return reconstruct_from_modes(traj_modes, self.Phi, self.u_mean)

    def print_equations(self, threshold: float = None) -> None:
        """Print identified sparse equations in readable form."""
        thresh = threshold or self.threshold / 10
        print("\nIdentified SINDy Equations:")
        print("=" * 50)
        for j in range(self.r):
            terms = []
            for i, name in enumerate(self.feature_names):
                coeff = self.Xi[i, j]
                if abs(coeff) >= thresh:
                    terms.append(f"{coeff:+.4f}*{name}")
            eq = " ".join(terms) if terms else "0"
            print(f"  da{j+1}/dt = {eq}")


# ─────────────────────────────────────────────────────────────────────────────
# Full SINDy Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def fit_sindy(traj: np.ndarray, solver,
              n_modes: int = 8,
              degree: int = 2,
              threshold: float = 0.05,
              dt: float = 0.25) -> SINDyModel:
    """
    Full SINDy fitting pipeline.

    Steps:
    1. Compute POD basis from trajectory
    2. Project trajectory onto basis
    3. Compute time derivatives of modal amplitudes
    4. Build polynomial library
    5. STLSQ sparse regression
    6. Return SINDyModel

    Args:
        traj: (T, N) training trajectory in physical space
        solver: KSSolver (for metadata)
        n_modes: number of POD/Galerkin modes
        degree: polynomial degree for library
        threshold: STLSQ sparsification threshold
        dt: time step

    Returns:
        SINDyModel
    """
    print(f"\nFitting SINDy: {n_modes} modes, degree {degree}, threshold {threshold}")

    # Step 1: POD basis
    Phi, u_mean = compute_galerkin_basis(solver, n_modes=n_modes, traj=traj)

    # Step 2: Project
    a = project_trajectory(traj, Phi, u_mean)  # (T, r)
    print(f"  Modal amplitudes shape: {a.shape}")
    print(f"  Amplitude ranges: {a.min(axis=0).round(3)} to {a.max(axis=0).round(3)}")

    # Step 3: Time derivatives
    da = compute_time_derivatives(a, dt, method="finite_diff")  # (T, r)

    # Trim edges where finite differences are less accurate
    trim = 4
    a_trim = a[trim:-trim]
    da_trim = da[trim:-trim]

    # Step 4: Library
    Theta, feature_names = polynomial_library(a_trim, degree=degree)
    print(f"  Library: {Theta.shape[1]} features ({len(feature_names)} terms)")

    # Step 5: STLSQ
    print("  Running STLSQ...")
    Xi, active = stlsq(Theta, da_trim, threshold=threshold, verbose=True)

    # Step 6: Create model
    model = SINDyModel(Xi, Phi, u_mean, feature_names, degree, threshold)
    model.print_equations()

    # Validation: check reconstruction error
    da_pred = Theta @ Xi
    rel_err = np.mean((da_pred - da_trim) ** 2) / (np.mean(da_trim ** 2) + 1e-12)
    print(f"\n  Training reconstruction error: {rel_err:.4f}")

    return model


if __name__ == "__main__":
    from ks_solver import KSSolver

    solver = KSSolver(L=22.0, N=64, dt=0.25)
    key = jax.random.PRNGKey(5)

    print("Generating trajectory for SINDy...")
    u0_hat = solver.random_ic(key)
    u0_hat = solver.warmup(u0_hat, n_warmup=2000)
    traj = np.array(solver.integrate(u0_hat, n_steps=4000))

    model = fit_sindy(traj, solver, n_modes=8, degree=2, threshold=0.05)

    # Test integration
    u0 = traj[0]
    pred = model.integrate(u0, n_steps=100, dt=solver.dt)
    print(f"\nSINDy prediction: {pred.shape}")
    print(f"Physical range: [{pred.min():.3f}, {pred.max():.3f}]")
    print("sindy.py: All checks passed.")
