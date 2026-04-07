"""
diagnostics.py — Dynamical Fidelity Diagnostics
================================================
Computes the full set of diagnostics for comparing true KSE vs surrogates.
All functions accept trajectory arrays of shape (T, N).

Diagnostics:
1. spatial_power_spectrum   — E(q) = <|û_q|^2> time-averaged
2. joint_pdf_derivatives    — joint PDF of (u_x, u_xx), key Linot 2022 figure
3. autocorrelation          — temporal autocorrelation of spatial mean
4. ensemble_error           — prediction error vs Lyapunov time (Linot Fig 7)
5. wasserstein_distance     — W1 between empirical distributions (Park 2024)
6. marginal_pdf             — 1D marginal PDF of u values
7. invariant_measure_stats  — mean, variance, skewness, kurtosis
8. clv_angle_distribution   — distribution of CLV angles (Ozalp 2024)
9. compare_systems          — full comparison table: true vs surrogate(s)
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from typing import Optional

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial Power Spectrum
# ─────────────────────────────────────────────────────────────────────────────

def spatial_power_spectrum(traj: np.ndarray, L: float = 22.0) -> tuple:
    """
    Time-averaged spatial power spectrum E(q) = <|û_q|^2>.

    Args:
        traj: trajectory in physical space, shape (T, N)
        L: domain length

    Returns:
        (q, E): wavenumbers and energy spectrum, shapes (N//2,)
    """
    T, N = traj.shape
    # FFT each snapshot
    traj_hat = np.fft.fft(traj, axis=1)  # (T, N)
    # Power spectrum (one-sided)
    power = np.abs(traj_hat) ** 2 / N ** 2  # normalize
    E_full = np.mean(power, axis=0)  # time-average

    # One-sided spectrum (positive wavenumbers only)
    n_modes = N // 2
    E = E_full[:n_modes]
    E[1:] *= 2  # double-sided to one-sided

    k_int = np.arange(n_modes)
    q = 2 * np.pi * k_int / L  # physical wavenumbers

    return q, E


# ─────────────────────────────────────────────────────────────────────────────
# 2. Joint PDF of Derivatives
# ─────────────────────────────────────────────────────────────────────────────

def compute_derivatives(traj: np.ndarray, L: float = 22.0) -> tuple:
    """
    Compute u_x and u_xx from trajectory via spectral differentiation.

    Returns:
        (ux, uxx): each shape (T, N)
    """
    T, N = traj.shape
    traj_hat = np.fft.fft(traj, axis=1)  # (T, N)

    k_int = np.fft.fftfreq(N, d=1.0 / N)
    q = 2 * np.pi * k_int / L

    ux_hat = 1j * q[None, :] * traj_hat
    uxx_hat = -q[None, :] ** 2 * traj_hat

    ux = np.fft.ifft(ux_hat, axis=1).real
    uxx = np.fft.ifft(uxx_hat, axis=1).real

    return ux, uxx


def joint_pdf_derivatives(traj: np.ndarray, L: float = 22.0,
                           n_bins: int = 64,
                           ux_range: tuple = (-4, 4),
                           uxx_range: tuple = (-8, 8)) -> tuple:
    """
    Joint PDF of (u_x, u_xx) — key diagnostic from Linot 2022 Fig 8.
    Standard metric for assessing if surrogate stays on correct attractor.

    Args:
        traj: (T, N) physical space trajectory
        n_bins: histogram bins per dimension
        ux_range: range for u_x axis
        uxx_range: range for u_xx axis

    Returns:
        (ux_edges, uxx_edges, pdf): 2D histogram
    """
    ux, uxx = compute_derivatives(traj, L)

    # Flatten all space-time points
    ux_flat = ux.ravel()
    uxx_flat = uxx.ravel()

    H, ux_edges, uxx_edges = np.histogram2d(
        ux_flat, uxx_flat,
        bins=n_bins,
        range=[ux_range, uxx_range],
        density=True
    )
    return ux_edges, uxx_edges, H


def kl_divergence_pdf(pdf_true: np.ndarray, pdf_approx: np.ndarray,
                      eps: float = 1e-10) -> float:
    """
    KL divergence D_KL(true || approx).
    Uses convention: 0 * log(0/q) = 0.
    """
    p = pdf_true + eps
    q = pdf_approx + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Autocorrelation
# ─────────────────────────────────────────────────────────────────────────────

def temporal_autocorrelation(traj: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """
    Temporal autocorrelation of spatial mean <u(t)>.

    Args:
        traj: (T, N) trajectory
        max_lag: maximum lag in time steps

    Returns:
        C: autocorrelation, shape (max_lag,), C[0] = 1
    """
    u_mean = traj.mean(axis=1)  # spatial mean at each time, shape (T,)
    u_mean -= u_mean.mean()
    var = np.var(u_mean)

    C = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            C[lag] = 1.0
        else:
            C[lag] = np.mean(u_mean[:-lag] * u_mean[lag:]) / (var + 1e-12)

    return C


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ensemble Prediction Error
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_error(true_trajs: np.ndarray, pred_trajs: np.ndarray,
                   dt: float = 0.25,
                   lyapunov_time: float = 22.0) -> tuple:
    """
    Ensemble-averaged prediction error vs Lyapunov time (Linot Fig 7).

    Args:
        true_trajs: (n_ensemble, T, N)
        pred_trajs: (n_ensemble, T, N)
        dt: time step
        lyapunov_time: τ_L for normalization

    Returns:
        (t_norm, error_mean, error_std)
        t_norm: time normalized by Lyapunov time
        error_mean: mean relative error
        error_std: std of relative error
    """
    T = true_trajs.shape[1]
    times = np.arange(T) * dt
    t_norm = times / lyapunov_time

    # Normalization: mean distance between random attractor states
    # Approximate as std of true trajectory
    D = np.std(true_trajs)

    errors = np.linalg.norm(true_trajs - pred_trajs, axis=2) / (D + 1e-12)
    # errors: (n_ensemble, T)

    return t_norm, errors.mean(axis=0), errors.std(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Wasserstein Distance (Park 2024)
# ─────────────────────────────────────────────────────────────────────────────

def wasserstein1_marginals(traj_true: np.ndarray,
                           traj_approx: np.ndarray,
                           n_dims: int = 8) -> float:
    """
    Approximate W1 distance between empirical measures via 1D marginals.
    Computes W1 for each of the first n_dims spatial modes and averages.

    Full W1 in high dimensions is intractable; 1D marginals give a lower bound
    and practical diagnostic (Park 2024 uses this approach).

    Args:
        traj_true: (T, N)
        traj_approx: (T, N)
        n_dims: number of modes to compare

    Returns:
        mean W1 across marginals
    """
    w1_vals = []
    for i in range(min(n_dims, traj_true.shape[1])):
        x = traj_true[:, i]
        y = traj_approx[:, i]
        # W1 for 1D = integral |F_X - F_Y|
        w1 = stats.wasserstein_distance(x, y)
        w1_vals.append(w1)

    return float(np.mean(w1_vals))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Marginal PDF
# ─────────────────────────────────────────────────────────────────────────────

def marginal_pdf(traj: np.ndarray, n_bins: int = 100,
                 u_range: tuple = (-5, 5)) -> tuple:
    """
    Marginal PDF of u values (all space-time points pooled).

    Returns:
        (bin_centers, pdf)
    """
    counts, edges = np.histogram(traj.ravel(), bins=n_bins,
                                  range=u_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


# ─────────────────────────────────────────────────────────────────────────────
# 7. Invariant Measure Statistics
# ─────────────────────────────────────────────────────────────────────────────

def invariant_measure_stats(traj: np.ndarray, label: str = "") -> dict:
    """
    Compute statistical moments of the invariant measure.

    Returns dict with mean, variance, skewness, kurtosis, energy.
    """
    flat = traj.ravel()
    energy = np.mean(traj ** 2)  # L2 energy

    return {
        "label": label,
        "mean": float(np.mean(flat)),
        "variance": float(np.var(flat)),
        "skewness": float(stats.skew(flat)),
        "kurtosis": float(stats.kurtosis(flat)),
        "energy": float(energy),
        "rms": float(np.sqrt(energy)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLV Angle Distribution
# ─────────────────────────────────────────────────────────────────────────────

def clv_angle_stats(angles: np.ndarray) -> dict:
    """
    Statistics of CLV angle distribution.
    Near-zero angles indicate non-hyperbolicity (tangencies of stable/unstable manifolds).

    Args:
        angles: (T, n_pairs) in degrees

    Returns:
        dict with mean, std, fraction near zero, distribution info
    """
    flat = angles.ravel()
    frac_near_zero = float(np.mean(flat < 5.0))  # fraction < 5 degrees

    return {
        "mean_angle": float(np.mean(flat)),
        "std_angle": float(np.std(flat)),
        "min_angle": float(np.min(flat)),
        "fraction_near_zero": frac_near_zero,
        "is_hyperbolic": frac_near_zero < 0.01,  # heuristic
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full System Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_systems(systems: dict, L: float = 22.0,
                    dt: float = 0.25) -> dict:
    """
    Run all diagnostics on multiple systems and return comparison table.

    Args:
        systems: dict mapping name -> trajectory array (T, N)
        L: domain length
        dt: time step

    Returns:
        dict of diagnostic results per system
    """
    results = {}

    for name, traj in systems.items():
        print(f"\nComputing diagnostics for: {name}")
        r = {}

        # Power spectrum
        q, E = spatial_power_spectrum(traj, L)
        r["power_spectrum"] = (q, E)

        # Joint PDF
        ux_e, uxx_e, pdf = joint_pdf_derivatives(traj, L)
        r["joint_pdf"] = (ux_e, uxx_e, pdf)

        # Autocorrelation
        r["autocorr"] = temporal_autocorrelation(traj)

        # Invariant measure statistics
        r["stats"] = invariant_measure_stats(traj, label=name)

        # Marginal PDF
        r["marginal_pdf"] = marginal_pdf(traj)

        results[name] = r
        print(f"  Energy: {r['stats']['energy']:.4f}, "
              f"RMS: {r['stats']['rms']:.4f}, "
              f"Skew: {r['stats']['skewness']:.4f}")

    # Wasserstein distances (all vs first system)
    names = list(systems.keys())
    ref_name = names[0]
    ref_traj = systems[ref_name]

    for name in names[1:]:
        w1 = wasserstein1_marginals(ref_traj, systems[name])
        results[name]["wasserstein_vs_true"] = w1
        print(f"  W1({name} vs {ref_name}): {w1:.4f}")

    # KL divergence of joint PDFs
    _, _, pdf_ref = results[ref_name]["joint_pdf"]
    for name in names[1:]:
        _, _, pdf_approx = results[name]["joint_pdf"]
        kl = kl_divergence_pdf(pdf_ref, pdf_approx)
        results[name]["kl_joint_pdf"] = kl
        print(f"  KL({name} joint PDF): {kl:.4f}")

    return results


def print_comparison_table(lyapunov_results: dict) -> None:
    """Print formatted comparison table of Lyapunov diagnostics."""
    print("\n" + "=" * 70)
    print(f"{'System':<20} {'L1':>8} {'TL':>8} {'n_pos':>6} "
          f"{'D_KY':>8} {'h_KS':>8}")
    print("=" * 70)

    for name, res in lyapunov_results.items():
        print(f"{name:<20} {res['lambda_1']:>8.4f} {res['lyapunov_time']:>8.2f} "
              f"{res['n_positive']:>6d} {res['kaplan_yorke_dim']:>8.2f} "
              f"{res['ks_entropy']:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    import os
    from ks_solver import KSSolver

    solver = KSSolver(L=22.0, N=64, dt=0.25)
    key = jax.random.PRNGKey(1)

    print("Generating test trajectory...")
    u0_hat = solver.random_ic(key)
    u0_hat = solver.warmup(u0_hat, n_warmup=2000)
    traj = np.array(solver.integrate(u0_hat, n_steps=4000))

    print(f"Trajectory: {traj.shape}")

    # Run all diagnostics
    q, E = spatial_power_spectrum(traj)
    print(f"Power spectrum: {len(q)} modes, peak at q={q[np.argmax(E)]:.3f}")

    ux_e, uxx_e, pdf = joint_pdf_derivatives(traj)
    print(f"Joint PDF: shape {pdf.shape}, max={pdf.max():.3f}")

    stats_out = invariant_measure_stats(traj, label="True KSE")
    print(f"Stats: energy={stats_out['energy']:.4f}, rms={stats_out['rms']:.4f}")

    # Save diagnostics
    os.makedirs("data", exist_ok=True)
    np.save("data/power_spectrum_q.npy", q)
    np.save("data/power_spectrum_E.npy", E)
    np.save("data/joint_pdf.npy", pdf)
    print("Diagnostics saved.")
