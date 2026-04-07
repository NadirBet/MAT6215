"""
ks_solver.py — Kuramoto-Sivashinsky Equation Solver
====================================================
PDE:  u_t = -u*u_x - u_xx - u_xxxx
Domain: [0, L], L=22, periodic boundary conditions
Method: Pseudospectral ETD-RK4 in Fourier space

Data Simulation Strategy
------------------------
The KSE is solved in Fourier space where the linear operator is diagonal:
    L_k = q_k^2 - q_k^4,   q_k = 2*pi*k/L

ETD-RK4 integrates the linear part *exactly* (no stiffness), and treats
the nonlinear part N_hat_k = -i*q_k/2 * FFT(u^2) pseudospectrally.
2/3 dealiasing prevents aliasing errors in the quadratic nonlinearity.

API
---
    solver = KSSolver(L=22, N=64, dt=0.25)
    u0 = solver.random_ic(key)
    traj = solver.integrate(u0, n_steps=10000)       # shape (n_steps, N)
    traj, tangents = solver.integrate_with_tangents(u0, Q0, n_steps=1000)
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)


class KSSolver:
    """Pseudospectral ETD-RK4 solver for the Kuramoto-Sivashinsky equation."""

    def __init__(self, L: float = 22.0, N: int = 64, dt: float = 0.25):
        self.L = L
        self.N = N
        self.dt = dt

        # Wavenumbers
        k = jnp.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers
        self.q = 2.0 * jnp.pi * k / L       # physical wavenumbers

        # Linear operator (diagonal in Fourier space)
        # L_k = q^2 - q^4  (linearly unstable for small q, stable for large q)
        self.Lin = self.q ** 2 - self.q ** 4

        # Dealiasing mask: zero out top 1/3 of modes (2/3 rule)
        self.dealias = jnp.abs(k) < N / 3

        # Precompute ETD-RK4 coefficients
        self._precompute_etdrk4()

    def _phi1(self, z: jnp.ndarray) -> jnp.ndarray:
        """phi_1(z) = (exp(z) - 1) / z, stable near z=0 via Taylor series."""
        small = jnp.abs(z) < 1e-8
        safe_z = jnp.where(small, jnp.ones_like(z), z)
        exact = (jnp.exp(safe_z) - 1.0) / safe_z
        taylor = 1.0 + z / 2.0 + z ** 2 / 6.0 + z ** 3 / 24.0
        return jnp.where(small, taylor, exact)

    def _phi2(self, z: jnp.ndarray) -> jnp.ndarray:
        """phi_2(z) = (exp(z) - 1 - z) / z^2, stable near z=0."""
        small = jnp.abs(z) < 1e-8
        safe_z = jnp.where(small, jnp.ones_like(z), z)
        exact = (jnp.exp(safe_z) - 1.0 - safe_z) / safe_z ** 2
        taylor = 0.5 + z / 6.0 + z ** 2 / 24.0
        return jnp.where(small, taylor, exact)

    def _phi3(self, z: jnp.ndarray) -> jnp.ndarray:
        """phi_3(z) = (exp(z) - 1 - z - z^2/2) / z^3, stable near z=0."""
        small = jnp.abs(z) < 1e-8
        safe_z = jnp.where(small, jnp.ones_like(z), z)
        exact = (jnp.exp(safe_z) - 1.0 - safe_z - safe_z ** 2 / 2.0) / safe_z ** 3
        taylor = 1.0 / 6.0 + z / 24.0 + z ** 2 / 120.0
        return jnp.where(small, taylor, exact)

    def _precompute_etdrk4(self):
        """Precompute all ETD-RK4 coefficients (Kassam & Trefethen 2005)."""
        h = self.dt
        c = h * self.Lin        # full step argument
        c2 = h / 2 * self.Lin  # half step argument

        self.E = jnp.exp(c)
        self.E2 = jnp.exp(c2)

        # Phi functions at half and full step
        p1h = self._phi1(c2)
        p1f = self._phi1(c)
        p2f = self._phi2(c)
        p3f = self._phi3(c)

        # Intermediate stage coefficients
        self.a21 = h / 2 * p1h  # coefficient for stage 2 and 3

        # Final update coefficients (Cox-Matthews ETDRK4)
        # u_{n+1} = E*u_n + h*(f1*k1 + f2*(k2+k3) + f3*k4)
        self.f1 = h * (p1f - 3 * p2f + 4 * p3f)
        self.f2 = h * (2 * p2f - 4 * p3f)
        self.f3 = h * (-p2f + 4 * p3f)

    def nonlinear(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        """
        Compute nonlinear term N_hat = -i*q/2 * FFT(u^2).
        Pseudospectral: IFFT → square in physical space → FFT.
        Applies 2/3 dealiasing.
        """
        u_hat_dealiased = u_hat * self.dealias
        u = jnp.fft.ifft(u_hat_dealiased).real
        u2 = u ** 2
        return -1j * self.q / 2.0 * jnp.fft.fft(u2)

    def rhs(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        """Full RHS in Fourier space: L*u_hat + N(u_hat)."""
        return self.Lin * u_hat + self.nonlinear(u_hat)

    def rhs_physical(self, u: jnp.ndarray) -> jnp.ndarray:
        """RHS in physical space (for Jacobian computation)."""
        u_hat = jnp.fft.fft(u)
        rhs_hat = self.rhs(u_hat)
        return jnp.fft.ifft(rhs_hat).real

    def enforce_hermitian(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        """Enforce Hermitian symmetry so IFFT gives real output."""
        N = self.N
        # Zero mean, real k=0 and Nyquist
        u_hat = u_hat.at[0].set(u_hat[0].real)
        u_hat = u_hat.at[N // 2].set(u_hat[N // 2].real)
        # Mirror: u_hat[N-k] = conj(u_hat[k])
        u_hat = u_hat.at[N // 2 + 1:].set(jnp.conj(u_hat[1:N // 2][::-1]))
        return u_hat

    @partial(jax.jit, static_argnums=(0,))
    def step(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        """Single ETD-RK4 step in Fourier space."""
        k1 = self.nonlinear(u_hat)
        a = self.E2 * u_hat + self.a21 * k1

        k2 = self.nonlinear(a)
        b = self.E2 * u_hat + self.a21 * k2

        k3 = self.nonlinear(b)
        c = self.E2 * a + self.a21 * (2 * k3 - k1)

        k4 = self.nonlinear(c)

        u_next = self.E * u_hat + self.f1 * k1 + self.f2 * (k2 + k3) + self.f3 * k4
        return self.enforce_hermitian(u_next)

    def linearized_rhs(self, u_hat: jnp.ndarray, delta_hat: jnp.ndarray) -> jnp.ndarray:
        """
        Linearized RHS (tangent equation): J(u_hat) * delta_hat.
        Uses JAX JVP for exact linearization without forming full Jacobian.
        """
        _, tangent_out = jax.jvp(self.rhs, (u_hat,), (delta_hat,))
        return tangent_out

    @partial(jax.jit, static_argnums=(0,))
    def step_with_tangent(self, u_hat: jnp.ndarray, delta_hat: jnp.ndarray):
        """
        Simultaneous ETD-RK4 step for state + one tangent vector.
        Used in Benettin algorithm for Lyapunov exponents.
        """
        # State step
        u_next = self.step(u_hat)

        # Tangent step using linearized flow (ETD-RK4 on linearized equation)
        # For tangent we use standard RK4 since linearized system changes at each step
        Lin_d = self.Lin * delta_hat

        def rhs_tangent(d):
            return Lin_d + self.linearized_rhs(u_hat, d) - self.Lin * d

        # Simple RK4 for tangent (linearized system, no stiffness issue at tangent level)
        k1 = self.Lin * delta_hat + self.nonlinear_jacobian_vec(u_hat, delta_hat)
        k2 = self.Lin * (delta_hat + self.dt / 2 * k1) + self.nonlinear_jacobian_vec(u_hat, delta_hat + self.dt / 2 * k1)
        k3 = self.Lin * (delta_hat + self.dt / 2 * k2) + self.nonlinear_jacobian_vec(u_hat, delta_hat + self.dt / 2 * k2)
        k4 = self.Lin * (delta_hat + self.dt * k3) + self.nonlinear_jacobian_vec(u_hat, delta_hat + self.dt * k3)

        delta_next = delta_hat + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return u_next, delta_next

    def nonlinear_jacobian_vec(self, u_hat: jnp.ndarray, v_hat: jnp.ndarray) -> jnp.ndarray:
        """JVP of nonlinear term only."""
        _, out = jax.jvp(self.nonlinear, (u_hat,), (v_hat,))
        return out

    def integrate(self, u0_hat: jnp.ndarray, n_steps: int,
                  save_every: int = 1) -> jnp.ndarray:
        """
        Integrate KSE forward in time.

        Args:
            u0_hat: Initial condition in Fourier space, shape (N,)
            n_steps: Number of ETD-RK4 steps
            save_every: Save state every this many steps

        Returns:
            trajectory in physical space, shape (n_saves, N)
        """
        def scan_fn(u_hat, _):
            u_next = self.step(u_hat)
            return u_next, jnp.fft.ifft(u_next).real

        # Run with lax.scan for JIT efficiency
        _, traj = jax.lax.scan(scan_fn, u0_hat, None, length=n_steps)
        return traj[::save_every]  # shape (n_saves, N)

    def integrate_fourier(self, u0_hat: jnp.ndarray, n_steps: int) -> jnp.ndarray:
        """Integrate and return trajectory in Fourier space."""
        def scan_fn(u_hat, _):
            u_next = self.step(u_hat)
            return u_next, u_next

        _, traj_hat = jax.lax.scan(scan_fn, u0_hat, None, length=n_steps)
        return traj_hat  # shape (n_steps, N) complex

    def random_ic(self, key: jax.random.PRNGKey, amplitude: float = 0.1) -> jnp.ndarray:
        """
        Random initial condition in Fourier space with proper Hermitian symmetry.
        u(x) is real iff u_hat[-k] = conj(u_hat[k]).
        We construct this directly: set positive modes, mirror for negative modes.
        Modes decay as exp(-k) to ensure smoothness.
        """
        N = self.N
        n_pos = N // 2 - 1  # number of independent positive modes (k=1,...,N/2-1)

        key1, key2 = jax.random.split(key)
        k_pos = np.arange(1, N // 2)
        envelope = np.exp(-k_pos.astype(float))

        real_pos = np.array(jax.random.normal(key1, (n_pos,))) * envelope * amplitude
        imag_pos = np.array(jax.random.normal(key2, (n_pos,))) * envelope * amplitude

        u0_hat = np.zeros(N, dtype=complex)
        u0_hat[0] = 0.0                          # zero mean (KSE conserves mean)
        u0_hat[1:N // 2] = real_pos + 1j * imag_pos
        u0_hat[N // 2] = 0.0                     # Nyquist mode = 0
        u0_hat[N // 2 + 1:] = np.conj(u0_hat[1:N // 2][::-1])  # Hermitian symmetry

        return jnp.array(u0_hat)

    def warmup(self, u0_hat: jnp.ndarray, n_warmup: int = 2000) -> jnp.ndarray:
        """Run transient to reach attractor. Returns final state on attractor."""
        def scan_fn(u_hat, _):
            return self.step(u_hat), None
        u_final, _ = jax.lax.scan(scan_fn, u0_hat, None, length=n_warmup)
        return u_final

    def jacobian_physical(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Full N x N Jacobian of RHS in physical space.
        Used for Lyapunov exponent computation.
        """
        return jax.jacobian(self.rhs_physical)(u)

    def x_grid(self) -> np.ndarray:
        """Physical space grid points."""
        return np.linspace(0, self.L, self.N, endpoint=False)


def generate_training_data(solver: KSSolver, key: jax.random.PRNGKey,
                            n_warmup: int = 2000,
                            n_train: int = 80000,
                            n_test: int = 20000) -> dict:
    """
    Generate training and test trajectory data.

    Strategy:
    - Warmup: discard first n_warmup steps (transient to attractor)
    - Training: next n_train steps saved every step
    - Test: next n_test steps

    Returns dict with keys: u_train, u_test, u_hat_train, u_hat_test
    """
    print(f"Generating KSE data: warmup={n_warmup}, train={n_train}, test={n_test}")

    u0_hat = solver.random_ic(key)

    print("  Running warmup...")
    u_attractor = solver.warmup(u0_hat, n_warmup)

    print("  Running training trajectory...")
    def scan_full(u_hat, _):
        u_next = solver.step(u_hat)
        return u_next, (jnp.fft.ifft(u_next).real, u_next)

    u_final_train, (u_train, u_hat_train) = jax.lax.scan(
        scan_full, u_attractor, None, length=n_train + n_test
    )

    u_train_phys = u_train[:n_train]
    u_test_phys = u_train[n_train:]
    u_hat_train_f = u_hat_train[:n_train]
    u_hat_test_f = u_hat_train[n_train:]

    print(f"  Train shape: {u_train_phys.shape}, Test shape: {u_test_phys.shape}")

    return {
        "u_train": np.array(u_train_phys),
        "u_test": np.array(u_test_phys),
        "u_hat_train": np.array(u_hat_train_f),
        "u_hat_test": np.array(u_hat_test_f),
        "dt": solver.dt,
        "L": solver.L,
        "N": solver.N,
    }


def generate_ensemble(solver: KSSolver, key: jax.random.PRNGKey,
                      n_ensemble: int = 100,
                      n_steps: int = 400) -> jnp.ndarray:
    """
    Generate ensemble of short trajectories from random ICs on attractor.
    Used for ensemble-averaged error curves.
    Returns: shape (n_ensemble, n_steps, N)
    """
    keys = jax.random.split(key, n_ensemble)

    def single_traj(k):
        u0 = solver.random_ic(k)
        u_att = solver.warmup(u0, 2000)
        return solver.integrate(u_att, n_steps)

    trajs = jax.vmap(single_traj)(keys)
    return trajs  # (n_ensemble, n_steps, N)


if __name__ == "__main__":
    import os
    solver = KSSolver(L=22.0, N=64, dt=0.25)
    key = jax.random.PRNGKey(42)

    # Quick validation: check energy spectrum shape
    u0 = solver.random_ic(key)
    u_att = solver.warmup(u0, n_warmup=2000)
    traj = solver.integrate(u_att, n_steps=1000)
    print(f"Trajectory shape: {traj.shape}")
    print(f"State range: [{traj.min():.3f}, {traj.max():.3f}]")
    print(f"Mean energy: {jnp.mean(traj**2):.4f}")

    # Save data
    data = generate_training_data(solver, key, n_warmup=2000, n_train=40000, n_test=10000)
    save_path = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(save_path, exist_ok=True)
    for k_name, v in data.items():
        if isinstance(v, np.ndarray):
            np.save(os.path.join(save_path, f"{k_name}.npy"), v)
    print("Data saved to data/")
