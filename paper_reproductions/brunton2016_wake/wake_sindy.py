from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from paper_reproductions.brunton2016_wake.zenodo_re100 import load_snapshots


@dataclass(frozen=True)
class PODResult:
    mean_state: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    energy_fraction: np.ndarray
    coefficients: np.ndarray
    dt: float
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SINDyFit:
    coefficients: np.ndarray
    feature_names: list[str]
    threshold: float
    degree: int
    n_active: int
    derivative_rmse: float
    rollout_rmse: float
    stable: bool


def load_velocity_state_matrix(raw_path: str | Path, dtype: np.dtype | str = np.float32) -> dict[str, np.ndarray]:
    data = load_snapshots(raw_path, dtype=dtype, include_pressure=False, validate_mesh=True)
    u = np.asarray(data["u"], dtype=dtype)
    v = np.asarray(data["v"], dtype=dtype)
    state = np.concatenate([u, v], axis=1)
    return {
        "times": np.asarray(data["times"], dtype=dtype),
        "x": np.asarray(data["x"], dtype=dtype),
        "y": np.asarray(data["y"], dtype=dtype),
        "u": u,
        "v": v,
        "state": state,
    }


def snapshot_pod(state: np.ndarray, times: np.ndarray, x: np.ndarray, y: np.ndarray, n_modes: int = 6) -> PODResult:
    mean_state = state.mean(axis=0)
    centered = state - mean_state[None, :]
    temporal_cov = centered @ centered.T
    eigvals, eigvecs = np.linalg.eigh(temporal_cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    singular_values = np.sqrt(eigvals)
    safe = np.where(singular_values > 1e-12, singular_values, 1.0)
    modes = centered.T @ eigvecs[:, :n_modes]
    modes = modes / safe[:n_modes][None, :]

    coefficients = centered @ modes
    energy_fraction = eigvals / np.maximum(eigvals.sum(), 1e-12)
    dt = float(np.mean(np.diff(times)))
    return PODResult(
        mean_state=mean_state,
        modes=modes,
        singular_values=singular_values[:n_modes],
        energy_fraction=energy_fraction[:n_modes],
        coefficients=coefficients[:, :n_modes],
        dt=dt,
        times=times,
        x=x,
        y=y,
    )


def polynomial_library(a: np.ndarray, degree: int = 5, include_bias: bool = True, var_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    t, r = a.shape
    if var_names is None:
        default = ["x", "y", "z", "w", "v", "u"]
        var_names = default[:r]

    features: list[np.ndarray] = []
    names: list[str] = []

    if include_bias:
        features.append(np.ones((t, 1), dtype=a.dtype))
        names.append("1")

    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(r), d):
            term = np.prod(a[:, comb], axis=1, dtype=a.dtype)
            features.append(term[:, None])
            pieces: list[str] = []
            last = None
            count = 0
            for idx in comb + (-1,):
                if idx == last:
                    count += 1
                    continue
                if last is not None:
                    pieces.append(var_names[last] if count == 1 else f"{var_names[last]}^{count}")
                last = idx
                count = 1
            names.append("*".join(pieces))

    return np.hstack(features), names


def finite_difference(a: np.ndarray, dt: float) -> np.ndarray:
    da = np.zeros_like(a)
    da[2:-2] = (-a[4:] + 8 * a[3:-1] - 8 * a[1:-3] + a[:-4]) / (12 * dt)
    da[0] = (-3 * a[0] + 4 * a[1] - a[2]) / (2 * dt)
    da[1] = (-3 * a[1] + 4 * a[2] - a[3]) / (2 * dt)
    da[-2] = (3 * a[-2] - 4 * a[-3] + a[-4]) / (2 * dt)
    da[-1] = (3 * a[-1] - 4 * a[-2] + a[-3]) / (2 * dt)
    return da


def stlsq(theta: np.ndarray, da: np.ndarray, threshold: float, max_iter: int = 30) -> np.ndarray:
    xi, *_ = np.linalg.lstsq(theta, da, rcond=None)
    for _ in range(max_iter):
        active = np.abs(xi) >= threshold
        xi_new = np.zeros_like(xi)
        for j in range(da.shape[1]):
            active_j = active[:, j]
            if not np.any(active_j):
                continue
            xi_j, *_ = np.linalg.lstsq(theta[:, active_j], da[:, j], rcond=None)
            xi_new[active_j, j] = xi_j
        if np.allclose(xi, xi_new, atol=1e-10, rtol=1e-8):
            xi = xi_new
            break
        xi = xi_new
    return xi


def sindy_rhs(a: np.ndarray, xi: np.ndarray, degree: int, feature_names: list[str] | None = None) -> np.ndarray:
    theta, _ = polynomial_library(a[None, :], degree=degree, include_bias=True, var_names=feature_names)
    return (theta @ xi)[0]


def rk4_step(a: np.ndarray, dt: float, xi: np.ndarray, degree: int, feature_names: list[str] | None = None) -> np.ndarray:
    k1 = sindy_rhs(a, xi, degree, feature_names)
    k2 = sindy_rhs(a + 0.5 * dt * k1, xi, degree, feature_names)
    k3 = sindy_rhs(a + 0.5 * dt * k2, xi, degree, feature_names)
    k4 = sindy_rhs(a + dt * k3, xi, degree, feature_names)
    return a + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_sindy(a0: np.ndarray, times: np.ndarray, xi: np.ndarray, degree: int, feature_names: list[str] | None = None) -> np.ndarray:
    traj = np.zeros((len(times), len(a0)), dtype=np.float64)
    traj[0] = a0
    dt = float(np.mean(np.diff(times)))
    for i in range(1, len(times)):
        traj[i] = rk4_step(traj[i - 1], dt, xi, degree, feature_names)
    return traj


def fit_sindy(a: np.ndarray, times: np.ndarray, degree: int, threshold: float, feature_names: list[str] | None = None) -> SINDyFit:
    dt = float(np.mean(np.diff(times)))
    da = finite_difference(a, dt)
    theta, names = polynomial_library(a, degree=degree, include_bias=True, var_names=feature_names)
    xi = stlsq(theta, da, threshold=threshold)
    da_pred = theta @ xi
    derivative_rmse = float(np.sqrt(np.mean((da_pred - da) ** 2)))
    rollout = simulate_sindy(a[0], times, xi, degree=degree, feature_names=feature_names)
    rollout_rmse = float(np.sqrt(np.mean((rollout - a) ** 2)))
    stable = bool(np.all(np.isfinite(rollout)) and np.max(np.abs(rollout)) < 1e6)
    n_active = int(np.count_nonzero(np.abs(xi) > 0))
    return SINDyFit(
        coefficients=xi,
        feature_names=names,
        threshold=threshold,
        degree=degree,
        n_active=n_active,
        derivative_rmse=derivative_rmse,
        rollout_rmse=rollout_rmse,
        stable=stable,
    )


def coeff_table_rows(fit: SINDyFit, state_names: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, feature in enumerate(fit.feature_names):
        row: dict[str, float | str] = {"feature": feature}
        for j, state_name in enumerate(state_names):
            row[state_name] = float(fit.coefficients[i, j])
        rows.append(row)
    return rows
