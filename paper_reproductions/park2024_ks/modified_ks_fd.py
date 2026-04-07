"""
Repo-aligned modified Kuramoto-Sivashinsky solver for the Park 2024 KS path.

This follows the downloaded `stacNODE` implementation more closely than the
first scaffold:

- full state includes both boundary nodes
- L = 128, n_inner = 127, dx = 1 by default
- centered second-order finite differences
- explicit treatment of nonlinear and c*u_x terms
- implicit treatment of linear dissipative terms via the same staged update

The training path still operates on the 127 interior states because that is the
state used by the Park KS model.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _build_repo_operators(n_inner: int, domain_length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the full-state first-derivative and implicit linear operators."""
    n_full = n_inner + 2
    dx = domain_length / (n_inner + 1)
    dx2 = dx * dx
    dx4 = dx2 * dx2

    # Centered first derivative with zero boundary rows.
    d1 = np.zeros((n_full, n_full), dtype=np.float64)
    for i in range(1, n_full - 1):
        d1[i, i + 1] = 1.0 / (2.0 * dx)
        d1[i, i - 1] = -1.0 / (2.0 * dx)

    # Centered second derivative with zeroed boundary rows/cols.
    d2 = np.zeros((n_full, n_full), dtype=np.float64)
    for i in range(1, n_full - 1):
        d2[i, i - 1] = 1.0 / dx2
        d2[i, i] = -2.0 / dx2
        d2[i, i + 1] = 1.0 / dx2

    d2[0, :] = 0.0
    d2[-1, :] = 0.0
    d2[:, 0] = 0.0
    d2[:, -1] = 0.0

    # Fourth derivative with the repo's boundary closure.
    d4 = np.zeros((n_full, n_full), dtype=np.float64)
    for i in range(2, n_full - 2):
        d4[i, i - 2] = 1.0 / dx4
        d4[i, i - 1] = -4.0 / dx4
        d4[i, i] = 6.0 / dx4
        d4[i, i + 1] = -4.0 / dx4
        d4[i, i + 2] = 1.0 / dx4

    # Boundary closure copied from stacNODE's KS.py.
    d4[1, 1] = 7.0 / dx4
    d4[1, 2] = -4.0 / dx4
    d4[1, 3] = 1.0 / dx4
    d4[-2, -2] = 7.0 / dx4
    d4[-2, -3] = -4.0 / dx4
    d4[-2, -4] = 1.0 / dx4

    implicit_linear = -(d2 + d4)
    return d1, d2, implicit_linear


class ModifiedKSFD:
    """Finite-difference modified KS solver with repo-aligned IMEX stepping."""

    def __init__(self, n_inner: int = 127, domain_length: float = 128.0, c_param: float = 0.4, dt: float = 0.25):
        self.n_inner = int(n_inner)
        self.n_full = self.n_inner + 2
        self.domain_length = float(domain_length)
        self.c_param = float(c_param)
        self.dt = float(dt)
        self.dx = self.domain_length / (self.n_inner + 1)

        d1_np, _, implicit_np = _build_repo_operators(self.n_inner, self.domain_length)
        self.d1_full = jnp.array(d1_np)
        self.implicit_linear_full = jnp.array(implicit_np)
        self.identity_full = jnp.eye(self.n_full, dtype=jnp.float64)

        x_full = np.arange(0.0, self.domain_length + self.dx, self.dx, dtype=np.float64)
        self.x_full = jnp.array(x_full)
        self.x_inner = self.x_full[1:-1]

    def lift_interior(self, u_inner: jnp.ndarray) -> jnp.ndarray:
        u_inner = jnp.asarray(u_inner, dtype=jnp.float64)
        if u_inner.shape[0] != self.n_inner:
            raise ValueError(f"Expected interior state of length {self.n_inner}, got {u_inner.shape[0]}")
        return jnp.concatenate((jnp.array([0.0], dtype=jnp.float64), u_inner, jnp.array([0.0], dtype=jnp.float64)))

    def project_interior(self, u_full: jnp.ndarray) -> jnp.ndarray:
        u_full = self.enforce_boundaries(u_full)
        return u_full[1:-1]

    def enforce_boundaries(self, u_full: jnp.ndarray) -> jnp.ndarray:
        u_full = jnp.asarray(u_full, dtype=jnp.float64)
        return u_full.at[0].set(0.0).at[-1].set(0.0)

    def repo_initial_condition(self) -> jnp.ndarray:
        """Gaussian initial condition used in the downloaded stacNODE KS script."""
        x = self.x_full
        u0 = jnp.exp(-((x - self.domain_length / 2.0) ** 2) / 512.0)
        return self.enforce_boundaries(u0)

    def random_initial_condition(self, key: jax.Array, amplitude: float = 0.5) -> jnp.ndarray:
        """
        Smooth full-state random initial condition.

        This is kept as an optional helper, but the repo-style initial condition
        is the default path for the Park reproduction.
        """
        raw = jax.random.normal(key, (self.n_full,))
        envelope = jnp.sin(jnp.pi * self.x_full / self.domain_length) ** 2
        return self.enforce_boundaries(amplitude * envelope * raw)

    def explicit_nonlinear_full(self, u_full: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * (self.d1_full @ (u_full * u_full))

    def explicit_linear_full(self, u_full: jnp.ndarray) -> jnp.ndarray:
        return -self.c_param * (self.d1_full @ u_full)

    def explicit_increment_full(self, u_full: jnp.ndarray) -> jnp.ndarray:
        dt = self.dt

        def explicit_rhs(state: jnp.ndarray) -> jnp.ndarray:
            return self.explicit_nonlinear_full(state) + self.explicit_linear_full(state)

        k1 = explicit_rhs(u_full)
        k2 = explicit_rhs(u_full + dt * k1 / 3.0)
        k3 = explicit_rhs(u_full + dt * k2)
        k4 = explicit_rhs(u_full + dt * (0.75 * k2 + 0.25 * k3))
        return dt * (0.75 * k2 - 0.25 * k3 + 0.5 * k4)

    def implicit_increment_full(self, u_full: jnp.ndarray) -> jnp.ndarray:
        dt = self.dt
        a = self.implicit_linear_full
        au = a @ u_full
        eye = self.identity_full
        k2 = jnp.linalg.solve(eye - dt * a / 3.0, au)
        k3 = jnp.linalg.solve(eye - dt * a / 2.0, au + dt * (a @ k2) / 2.0)
        k4 = jnp.linalg.solve(eye - dt * a / 2.0, au + dt * (a @ (3.0 * k2 - k3)) / 4.0)
        return dt * (0.75 * k2 - 0.25 * k3 + 0.5 * k4)

    def step_full(self, u_full: jnp.ndarray) -> jnp.ndarray:
        u_full = self.enforce_boundaries(u_full)
        u_next = u_full + self.explicit_increment_full(u_full) + self.implicit_increment_full(u_full)
        return self.enforce_boundaries(u_next)

    def step(self, u_inner: jnp.ndarray) -> jnp.ndarray:
        u_full = self.lift_interior(u_inner)
        return self.project_interior(self.step_full(u_full))

    def map_jacobian(self, u_inner: jnp.ndarray) -> jnp.ndarray:
        return jax.jacfwd(self.step)(u_inner)

    def integrate_full(self, u0_full: jnp.ndarray, n_steps: int) -> jnp.ndarray:
        step_fn = jax.jit(self.step_full)

        def scan_fn(u, _):
            u_next = step_fn(u)
            return u_next, u_next

        _, traj = jax.lax.scan(scan_fn, self.enforce_boundaries(u0_full), None, length=int(n_steps))
        return traj

    def integrate(self, u0: jnp.ndarray, n_steps: int, *, state_kind: str = "interior") -> jnp.ndarray:
        if state_kind == "full":
            return self.integrate_full(u0, n_steps)
        if state_kind == "interior":
            traj_full = self.integrate_full(self.lift_interior(u0), n_steps)
            return traj_full[:, 1:-1]
        raise ValueError(f"Unsupported state_kind: {state_kind}")

    def warmup(self, u0: jnp.ndarray, n_steps: int, *, state_kind: str = "full") -> jnp.ndarray:
        step_fn = jax.jit(self.step_full if state_kind == "full" else self.step)

        def scan_fn(u, _):
            return step_fn(u), None

        u_final, _ = jax.lax.scan(scan_fn, u0, None, length=int(n_steps))
        return u_final

    def time_to_steps(self, total_time: float) -> int:
        return int(math.ceil(float(total_time) / self.dt))

    def to_full_state(self, u_inner: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.asarray(u_inner, dtype=np.float64), [0.0]))

    def trajectory_to_full(self, traj_inner: np.ndarray) -> np.ndarray:
        return np.stack([self.to_full_state(u) for u in np.asarray(traj_inner)], axis=0)
