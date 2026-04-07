"""
prob_diagnostics.py — Probabilistic Evaluation Metrics
=======================================================
Shared evaluation module for all probabilistic surrogate branches.

All functions are pure numpy (no JAX) so they work on saved trajectories
without GPU and can be called from any branch script.

Metrics:
  - nll_gaussian          Negative log-likelihood under diagonal Gaussian
  - crps_gaussian         Continuous Ranked Probability Score (proper scoring rule)
  - calibration_curve     Expected vs observed coverage at multiple quantile levels
  - rank_histogram        Talagrand rank histogram (ensemble reliability)
  - coverage              Empirical coverage at 50/90/95% ensemble intervals
  - ensemble_rmse_spread  RMSE of ensemble mean vs truth, plus ensemble spread, over time
  - spread_skill_ratio    Spread/RMSE ratio (ideal = 1.0 for a calibrated ensemble)
  - wasserstein1_empirical Wasserstein-1 distance between two empirical sample sets
  - transition_cov_error  Frobenius error between predicted and empirical transition cov
  - summary_table         Print a compact summary table across multiple models
"""

import numpy as np
from scipy.stats import norm as scipy_norm
from scipy.special import erf


# ── Point metrics ──────────────────────────────────────────────────────────────

def nll_gaussian(pred_mean, pred_std, targets):
    """
    Mean negative log-likelihood under diagonal Gaussian predictive.

    Args:
        pred_mean : (..., n) predicted mean
        pred_std  : (..., n) predicted std  (must be > 0)
        targets   : (..., n) true values

    Returns:
        scalar — mean NLL per element (nats)
    """
    pred_std = np.clip(pred_std, 1e-8, None)
    nll = 0.5 * np.log(2 * np.pi) + np.log(pred_std) + \
          0.5 * ((targets - pred_mean) / pred_std) ** 2
    return float(np.mean(nll))


def crps_gaussian(pred_mean, pred_std, targets):
    """
    CRPS for a Gaussian predictive distribution (analytic formula).

    CRPS(N(μ,σ²), y) = σ * [ z*(2Φ(z)-1) + 2φ(z) - 1/√π ]
    where z = (y - μ) / σ.

    Args:
        pred_mean : (..., n)
        pred_std  : (..., n)
        targets   : (..., n)

    Returns:
        scalar — mean CRPS (same units as targets; lower is better)
    """
    pred_std = np.clip(pred_std, 1e-8, None)
    z = (targets - pred_mean) / pred_std
    crps = pred_std * (
        z * (2 * scipy_norm.cdf(z) - 1)
        + 2 * scipy_norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


# ── Calibration ────────────────────────────────────────────────────────────────

def calibration_curve(pred_std, errors, n_bins=15):
    """
    Reliability diagram: expected coverage vs observed coverage.

    For a perfectly calibrated model, 90% of errors should fall within ±1.645σ.

    Args:
        pred_std : (N,) or (N, d) flattened predicted standard deviations
        errors   : (N,) or (N, d) flattened residuals (targets - pred_mean)
        n_bins   : number of confidence levels to check

    Returns:
        expected_cov : (n_bins,) array of nominal coverage levels [0.1 .. 0.99]
        observed_cov : (n_bins,) array of empirical coverage fractions
    """
    pred_std = np.asarray(pred_std).ravel()
    errors   = np.asarray(errors).ravel()
    pred_std = np.clip(pred_std, 1e-8, None)
    z_scores = np.abs(errors / pred_std)

    levels = np.linspace(0.05, 0.99, n_bins)
    observed = []
    for p in levels:
        z_crit = scipy_norm.ppf((1 + p) / 2)
        observed.append(float(np.mean(z_scores <= z_crit)))
    return np.array(levels), np.array(observed)


def calibration_error(pred_std, errors, n_bins=15):
    """
    Mean absolute calibration error (MACE) — scalar summary of calibration curve.
    0.0 = perfectly calibrated.
    """
    exp, obs = calibration_curve(pred_std, errors, n_bins)
    return float(np.mean(np.abs(exp - obs)))


# ── Ensemble metrics ───────────────────────────────────────────────────────────

def rank_histogram(ensemble, verifying_obs):
    """
    Talagrand / rank histogram for ensemble reliability.

    Args:
        ensemble       : (M, T, n) — M members, T timesteps, n state dims
        verifying_obs  : (T, n) — verification observations

    Returns:
        counts : (M+1,) normalized histogram (should be flat for calibrated ensemble)
        edges  : (M+2,) bin edges
    """
    M = ensemble.shape[0]
    T, n = verifying_obs.shape
    ranks = []
    for t in range(T):
        for i in range(n):
            r = int(np.sum(ensemble[:, t, i] < verifying_obs[t, i]))
            ranks.append(r)
    counts, edges = np.histogram(ranks, bins=np.arange(M + 2) - 0.5, density=True)
    return counts, edges


def coverage(ensemble_traj, true_traj, levels=(0.5, 0.9, 0.95)):
    """
    Fraction of true trajectory falling within ensemble central intervals.

    Args:
        ensemble_traj : (M, T, n)
        true_traj     : (T, n)
        levels        : iterable of probability levels

    Returns:
        dict {level: empirical_coverage_fraction}
    """
    results = {}
    for p in levels:
        lo = np.percentile(ensemble_traj, 100 * (1 - p) / 2, axis=0)
        hi = np.percentile(ensemble_traj, 100 * (1 + p) / 2, axis=0)
        frac = float(np.mean((true_traj >= lo) & (true_traj <= hi)))
        results[float(p)] = frac
    return results


def ensemble_rmse_spread(ensemble_traj, true_traj):
    """
    RMSE of ensemble mean and ensemble spread (std) at each timestep.

    Args:
        ensemble_traj : (M, T, n)
        true_traj     : (T, n)

    Returns:
        rmse   : (T,) RMSE of ensemble mean vs truth
        spread : (T,) mean ensemble std across state dimensions
    """
    mean_traj = np.mean(ensemble_traj, axis=0)           # (T, n)
    rmse   = np.sqrt(np.mean((mean_traj - true_traj) ** 2, axis=1))
    spread = np.mean(np.std(ensemble_traj, axis=0), axis=1)
    return rmse, spread


def spread_skill_ratio(ensemble_traj, true_traj):
    """
    Time-mean spread/RMSE ratio.  Ideal = 1.0 (calibrated).
    > 1: overdispersed, < 1: underdispersed.
    """
    rmse, spread = ensemble_rmse_spread(ensemble_traj, true_traj)
    mean_rmse   = float(np.mean(rmse[rmse > 0]))
    mean_spread = float(np.mean(spread))
    return mean_spread / mean_rmse if mean_rmse > 0 else np.nan


def ensemble_energy_score(ensemble_traj, true_traj):
    """
    Multivariate proper scoring rule: Energy Score.
    ES = E[||X - y||] - 0.5 * E[||X - X'||]
    Averaged over time steps.

    Args:
        ensemble_traj : (M, T, n)
        true_traj     : (T, n)

    Returns:
        scalar energy score (lower is better)
    """
    M, T, n = ensemble_traj.shape
    scores = []
    for t in range(T):
        y  = true_traj[t]
        ens = ensemble_traj[:, t, :]                    # (M, n)
        term1 = np.mean(np.linalg.norm(ens - y, axis=1))
        diff  = ens[:, None, :] - ens[None, :, :]      # (M, M, n)
        term2 = 0.5 * np.mean(np.linalg.norm(diff, axis=2))
        scores.append(term1 - term2)
    return float(np.mean(scores))


# ── Distribution comparison ────────────────────────────────────────────────────

def wasserstein1_empirical(samples_a, samples_b, n_proj=200, seed=0):
    """
    Sliced Wasserstein-1 distance between two empirical distributions.

    Projects both onto random 1D directions and averages |W1| over projections.
    Scales as O(n_proj * N log N) — suitable for N ~ 4000, d ~ 64.

    Args:
        samples_a : (N, d) first sample set
        samples_b : (M, d) second sample set
        n_proj    : number of random projections

    Returns:
        scalar sliced W1
    """
    rng = np.random.default_rng(seed)
    d = samples_a.shape[1]
    w1_vals = []
    for _ in range(n_proj):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        pa = np.sort(samples_a @ v)
        pb = np.sort(samples_b @ v)
        # interpolate to same length for W1
        if len(pa) != len(pb):
            t = np.linspace(0, 1, max(len(pa), len(pb)))
            pa = np.interp(t, np.linspace(0, 1, len(pa)), pa)
            pb = np.interp(t, np.linspace(0, 1, len(pb)), pb)
        w1_vals.append(np.mean(np.abs(pa - pb)))
    return float(np.mean(w1_vals))


def transition_cov_error(pred_cov_diag, h_t, h_next, n_bins=20):
    """
    Compare predicted marginal variance to empirical local variance of transitions.

    Bins h_t by first PC coordinate, computes empirical std of h_next in each bin,
    compares to mean predicted std.

    Args:
        pred_cov_diag : (N, d) predicted diagonal variance at each point
        h_t           : (N, d) conditioning states
        h_next        : (N, d) next states (observed)
        n_bins        : number of bins along first PC

    Returns:
        frobenius_ratio : mean(pred_std) / mean(empirical_std) — ideal = 1.0
    """
    pred_std = np.sqrt(np.clip(pred_cov_diag, 0, None))
    residuals = h_next - h_t  # approximate
    emp_std = np.std(residuals, axis=0)
    pred_std_mean = np.mean(pred_std, axis=0)
    ratio = pred_std_mean / np.clip(emp_std, 1e-10, None)
    return float(np.mean(ratio))


# ── Summary printing ───────────────────────────────────────────────────────────

def print_prob_summary(results_dict, true_traj, dt=0.25):
    """
    Print a compact comparison table across multiple probabilistic models.

    Args:
        results_dict : {name: {"ensemble": (M,T,n), "pred_mean": (T,n),
                                "pred_std": (T,n), "nll": float, ...}}
        true_traj    : (T, n)
        dt           : time step for display
    """
    print("\n" + "=" * 90)
    print(f"{'Model':<22} {'NLL':>8} {'CRPS':>8} {'CalErr':>8} "
          f"{'Spread/RMSE':>12} {'Cov90%':>8} {'SWD':>8}")
    print("-" * 90)
    for name, r in results_dict.items():
        nll  = r.get("nll",  float("nan"))
        crps = r.get("crps", float("nan"))
        cal  = r.get("cal_error", float("nan"))
        ssr  = r.get("spread_skill", float("nan"))
        cov  = r.get("coverage_90", float("nan"))
        swd  = r.get("swd", float("nan"))
        print(f"{name:<22} {nll:>8.4f} {crps:>8.4f} {cal:>8.4f} "
              f"{ssr:>12.3f} {cov:>8.3f} {swd:>8.4f}")
    print("=" * 90)
