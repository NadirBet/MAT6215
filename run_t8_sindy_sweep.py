"""
run_t8_sindy_sweep.py - Tasks T8 + T9 + T10: SINDy Sweep
===========================================================
Combined sweep covering:
  T8: STLSQ threshold sweep
  T9: Library expansion (degree, n_modes, coordinate system)
  T10: Derivative method comparison (finite_diff vs exact RHS)

Grid:
  threshold:    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
  degree:       2, 3
  n_modes:      4, 6, 8, 10, 12
  deriv_method: "exact_rhs", "finite_diff", "spectral"

For each configuration:
  - Active term count (sparsity)
  - Rollout stability + energy
  - Lyapunov exponents (in modal space via finite-difference Benettin)
  - D_KY, h_KS, n_pos

A "Pareto sweep" section finds the best threshold at fixed degree=2, n_modes=8,
deriv=exact_rhs, sweeping only threshold (fast).
The "library sweep" section fixes best threshold and varies degree+n_modes.

Outputs:
  data/sindy_sweep_results.pkl
  figures/figT8_sindy_threshold.png
  figures/figT9_sindy_library.png
"""

import sys
sys.path = [p for p in sys.path if 'MAT6215' not in p]
sys.path.insert(0, '.')
import gpu_config

import numpy as np
import jax
import jax.numpy as jnp
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

from ks_solver import KSSolver
from sindy import (
    compute_galerkin_basis, project_trajectory,
    polynomial_library, stlsq, SINDyModel,
    compute_time_derivatives,
)
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
traj_train = np.load("data/traj_train.npy")
traj_test  = np.load("data/traj_analysis.npy")
le_true    = np.load("data/lyapunov_exponents_full.npy")
solver = KSSolver(L=22.0, N=64, dt=0.25)

true_energy = float(np.mean(np.sum(traj_test**2, axis=1)))


def kaplan_yorke(le):
    cs = np.cumsum(le)
    k = np.where(cs < 0)[0]
    if len(k) == 0:
        return float(len(le))
    k = k[0]
    return float(k) + (cs[k-1] if k > 0 else 0.0) / abs(le[k])


# ── Exact RHS derivative (no finite differences) ────────────────────────────
def compute_exact_derivatives(traj, solver, Phi, u_mean):
    """
    Compute dh/dt = Phi^T * (du/dt) exactly using KSE RHS.
    This is the cleanest signal for SINDy — no FD noise.
    """
    T = len(traj)
    dhdt = np.zeros((T, Phi.shape[1]))
    for i in range(T):
        u = traj[i]
        u_hat = np.fft.fft(u)
        rhs_hat = solver.rhs(u_hat)
        dudt = np.fft.ifft(rhs_hat).real
        dhdt[i] = dudt @ Phi   # project du/dt onto POD modes (no centering for derivatives)
    return dhdt


# ── Lyapunov in modal space via finite-difference Benettin ──────────────────
def sindy_lyapunov(model, a0, n_steps=800, dt=0.25):
    """
    Benettin QR via JAX JVP through SINDy RK4 step.
    Works in modal coordinates (r-dimensional).
    """
    r = a0.shape[0]
    Xi_jax = jnp.array(model.Xi, dtype=jnp.float64)

    def theta_jax(a):
        """JAX-native polynomial library for a single modal state."""
        feats = [jnp.array([1.0], dtype=a.dtype), a]

        if model.degree >= 2:
            quad = [a[i] * a[j] for i, j in combinations_with_replacement(range(r), 2)]
            feats.append(jnp.stack(quad))

        if model.degree >= 3:
            cubic = [
                a[i] * a[j] * a[k]
                for i, j, k in combinations_with_replacement(range(r), 3)
            ]
            feats.append(jnp.stack(cubic))

        return jnp.concatenate(feats)

    def rhs_jax(a):
        return theta_jax(a) @ Xi_jax

    def rk4(a):
        k1 = rhs_jax(a)
        k2 = rhs_jax(a + dt/2*k1)
        k3 = rhs_jax(a + dt/2*k2)
        k4 = rhs_jax(a + dt*k3)
        return a + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    rk4_jit = jax.jit(rk4)
    a0j = jnp.array(a0, dtype=jnp.float64)
    Q0  = jnp.eye(r, dtype=jnp.float64)
    log0 = jnp.zeros(r, dtype=jnp.float64)

    def benettin(carry, _):
        a, Q, ls = carry
        Q_raw = jax.vmap(lambda q: jax.jvp(rk4_jit, (a,), (q,))[1],
                         in_axes=1, out_axes=1)(Q)
        a_n = rk4_jit(a)
        Q_n, R = jnp.linalg.qr(Q_raw)
        s = jnp.sign(jnp.diag(R))
        Q_n = Q_n * s[None, :]
        R   = R   * s[:, None]
        return (a_n, Q_n, ls + jnp.log(jnp.abs(jnp.diag(R)))), None

    try:
        (_, _, log_tot), _ = jax.lax.scan(benettin, (a0j, Q0, log0), None, length=n_steps)
        return np.array(log_tot / (n_steps * dt))
    except Exception as e:
        print(f"    Lyapunov scan failed: {e}")
        return None


# ── Rollout stability test ──────────────────────────────────────────────────
def test_rollout(model, u0, n_steps=2000):
    """Returns (stable, energy, traj[:500])."""
    try:
        traj_phys = model.integrate(u0, n_steps=n_steps, dt=0.25)
        if np.any(np.isnan(traj_phys)) or np.any(np.isinf(traj_phys)):
            return False, float('nan'), None
        energy = float(np.mean(np.sum(traj_phys**2, axis=1)))
        stable = energy < 1e6
        return stable, energy, traj_phys[:500]
    except Exception as e:
        return False, float('nan'), None


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 1 — Threshold sweep (fixed: degree=2, n_modes=8, exact RHS deriv)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SWEEP 1: Threshold sweep (degree=2, n_modes=8, exact_rhs)")
print("="*60)

THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
N_MODES_BASE = 8

Phi8, u_mean8 = compute_galerkin_basis(solver, n_modes=N_MODES_BASE, traj=traj_train)
a_train8 = project_trajectory(traj_train, Phi8, u_mean8)
dhdt_exact8 = compute_exact_derivatives(traj_train, solver, Phi8, u_mean8)
Theta8, names8 = polynomial_library(a_train8, degree=2, include_bias=True)
a0_test8 = (traj_test[200] - u_mean8) @ Phi8

threshold_results = {}
for thresh in THRESHOLDS:
    t0 = time.time()
    print(f"\n  threshold={thresh}:")
    Xi, active = stlsq(Theta8, dhdt_exact8, threshold=thresh, max_iter=20, verbose=False)
    model = SINDyModel(Xi, Phi8, u_mean8, names8, degree=2, threshold=thresh)

    n_active = int(active.sum())
    n_total  = int(active.size)
    print(f"    Active: {n_active}/{n_total} ({100*n_active/n_total:.1f}%)")

    stable, energy, _ = test_rollout(model, traj_test[0], n_steps=2000)
    print(f"    Rollout: stable={stable}, energy={energy:.2f}")

    # Lyapunov in modal coords
    le = sindy_lyapunov(model, a0_test8, n_steps=800)
    if le is not None:
        dky = kaplan_yorke(le)
        h_ks = float(np.sum(le[le > 0]))
        n_pos = int(np.sum(le > 0))
        print(f"    Lyapunov: L1={le[0]:+.4f}, n_pos={n_pos}, D_KY={dky:.2f}")
    else:
        dky = h_ks = float('nan'); n_pos = 0

    threshold_results[thresh] = {
        "threshold": thresh,
        "n_active": n_active, "n_total": n_total,
        "sparsity": n_active / n_total,
        "stable": stable, "energy": energy,
        "lyapunov": le, "D_KY": dky, "h_KS": h_ks, "n_pos": n_pos,
        "L1": float(le[0]) if le is not None else float('nan'),
        "runtime": time.time() - t0,
    }
    log_event("T8", "threshold_run",
              config={"threshold": thresh, "degree": 2, "n_modes": N_MODES_BASE},
              metrics={"n_active": n_active, "stable": stable, "energy": energy,
                       "D_KY": dky, "h_KS": h_ks, "n_pos": n_pos})


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 2 — Library sweep (fixed best threshold from sweep 1; vary degree + n_modes)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SWEEP 2: Library sweep (threshold=0.01, exact_rhs)")
print("="*60)

DEGREES  = [2, 3]
N_MODES_LIST = [4, 6, 8, 10, 12]
THRESH_LIB = 0.01

library_results = {}
for n_modes in N_MODES_LIST:
    for degree in DEGREES:
        key = (n_modes, degree)
        t0 = time.time()
        print(f"\n  n_modes={n_modes}, degree={degree}:")
        Phi, u_mean = compute_galerkin_basis(solver, n_modes=n_modes, traj=traj_train)
        a_tr = project_trajectory(traj_train, Phi, u_mean)
        dhdt_ex = compute_exact_derivatives(traj_train, solver, Phi, u_mean)
        Theta, names = polynomial_library(a_tr, degree=degree, include_bias=True)
        Xi, active = stlsq(Theta, dhdt_ex, threshold=THRESH_LIB, max_iter=20, verbose=False)
        model = SINDyModel(Xi, Phi, u_mean, names, degree=degree, threshold=THRESH_LIB)

        n_active = int(active.sum())
        n_total  = int(active.size)
        print(f"    Active: {n_active}/{n_total}")

        stable, energy, _ = test_rollout(model, traj_test[0], n_steps=2000)
        a0 = (traj_test[200] - u_mean) @ Phi
        le = sindy_lyapunov(model, a0, n_steps=800)
        if le is not None:
            dky = kaplan_yorke(le); h_ks = float(np.sum(le[le > 0])); n_pos = int(np.sum(le > 0))
            print(f"    Lyapunov: L1={le[0]:+.4f}, n_pos={n_pos}, D_KY={dky:.2f}")
        else:
            dky = h_ks = float('nan'); n_pos = 0
        print(f"    Rollout: stable={stable}, energy={energy:.2f}")

        library_results[key] = {
            "n_modes": n_modes, "degree": degree,
            "n_active": n_active, "n_total": n_total,
            "stable": stable, "energy": energy,
            "lyapunov": le, "D_KY": dky, "h_KS": h_ks, "n_pos": n_pos,
            "L1": float(le[0]) if le is not None else float('nan'),
            "runtime": time.time() - t0,
        }
        log_event("T9", "library_run",
                  config={"n_modes": n_modes, "degree": degree, "threshold": THRESH_LIB},
                  metrics={"n_active": n_active, "stable": stable, "energy": energy,
                           "D_KY": dky, "h_KS": h_ks, "n_pos": n_pos})


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP 3 — Derivative method comparison (n_modes=8, degree=2, threshold=0.01)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SWEEP 3: Derivative method (n_modes=8, degree=2, threshold=0.01)")
print("="*60)

a_train8_sub = a_train8[::2]  # use subsampled for fd methods
METHODS = ["exact_rhs", "finite_diff", "spectral"]
deriv_results = {}

for method in METHODS:
    t0 = time.time()
    print(f"\n  method={method}:")
    if method == "exact_rhs":
        dhdt = dhdt_exact8
    elif method == "finite_diff":
        dhdt = compute_time_derivatives(a_train8, dt=0.25, method="finite_diff")
    else:
        dhdt = compute_time_derivatives(a_train8, dt=0.25, method="spectral")

    Theta_m, names_m = polynomial_library(a_train8, degree=2)
    Xi, active = stlsq(Theta_m, dhdt, threshold=0.01, max_iter=20, verbose=False)
    model = SINDyModel(Xi, Phi8, u_mean8, names_m, degree=2, threshold=0.01)

    stable, energy, _ = test_rollout(model, traj_test[0], n_steps=2000)
    le = sindy_lyapunov(model, a0_test8, n_steps=800)
    if le is not None:
        dky = kaplan_yorke(le); h_ks = float(np.sum(le[le > 0])); n_pos = int(np.sum(le > 0))
    else:
        dky = h_ks = float('nan'); n_pos = 0
    print(f"    Rollout: stable={stable}, energy={energy:.2f}")
    if le is not None:
        print(f"    Lyapunov: L1={le[0]:+.4f}, n_pos={n_pos}, D_KY={dky:.2f}")

    deriv_results[method] = {
        "method": method, "n_active": int(active.sum()),
        "stable": stable, "energy": energy,
        "lyapunov": le, "D_KY": dky, "h_KS": h_ks, "n_pos": n_pos,
        "L1": float(le[0]) if le is not None else float('nan'),
        "runtime": time.time() - t0,
    }
    log_event("T10", "deriv_method_run",
              config={"method": method, "n_modes": N_MODES_BASE, "threshold": 0.01},
              metrics={"stable": stable, "energy": energy, "D_KY": dky, "h_KS": h_ks})


# ── Save ───────────────────────────────────────────────────────────────────────
all_results = {
    "threshold_sweep": threshold_results,
    "library_sweep": library_results,
    "deriv_sweep": deriv_results,
    "true": {"L1": le_true[0], "D_KY": kaplan_yorke(le_true),
             "h_KS": float(np.sum(le_true[le_true > 0])), "energy": true_energy},
}
with open("data/sindy_sweep_results.pkl", "wb") as f:
    pickle.dump(all_results, f)
print("\nSaved: data/sindy_sweep_results.pkl")

# ── Summary tables ─────────────────────────────────────────────────────────────
print("\n=== THRESHOLD SWEEP ===")
print(f"{'thresh':>8} {'active%':>8} {'stable':>7} {'energy':>8} {'L1':>8} {'D_KY':>7} {'h_KS':>8}")
for thresh, r in sorted(threshold_results.items()):
    print(f"{thresh:>8.3f} {100*r['sparsity']:>7.1f}% {str(r['stable']):>7} "
          f"{r['energy']:>8.1f} {r['L1']:>+8.4f} {r['D_KY']:>7.2f} {r['h_KS']:>8.4f}")

print("\n=== LIBRARY SWEEP ===")
print(f"{'modes':>6} {'deg':>4} {'active':>7} {'stable':>7} {'energy':>8} {'L1':>8} {'D_KY':>7}")
for (nm, deg), r in sorted(library_results.items()):
    print(f"{nm:>6d} {deg:>4d} {r['n_active']:>7d} {str(r['stable']):>7} "
          f"{r['energy']:>8.1f} {r['L1']:>+8.4f} {r['D_KY']:>7.2f}")

print("\n=== DERIV METHOD ===")
for method, r in deriv_results.items():
    print(f"  {method:<15} stable={r['stable']} energy={r['energy']:.2f} "
          f"L1={r['L1']:+.4f} D_KY={r['D_KY']:.2f}")


# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

threshs = sorted(threshold_results.keys())

# Sparsity vs threshold
ax = axes[0, 0]
sparsities = [100 * threshold_results[t]["sparsity"] for t in threshs]
ax.semilogx(threshs, sparsities, 'o-', color='C0', lw=2, ms=7)
ax.set_xlabel("STLSQ Threshold")
ax.set_ylabel("Active Terms (%)")
ax.set_title("Sparsity vs Threshold")
ax.grid(True, alpha=0.3)

# D_KY vs threshold
ax = axes[0, 1]
dky_vals = [threshold_results[t]["D_KY"] for t in threshs]
ax.semilogx(threshs, dky_vals, 'o-', color='C1', lw=2, ms=7)
ax.axhline(kaplan_yorke(le_true), ls='--', color='k', lw=1.5, label=f'True ({kaplan_yorke(le_true):.2f})')
ax.set_xlabel("STLSQ Threshold")
ax.set_ylabel("D_KY")
ax.set_title("D_KY vs Threshold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Energy vs threshold
ax = axes[0, 2]
energies = [threshold_results[t]["energy"] for t in threshs]
ax.semilogx(threshs, energies, 'o-', color='C2', lw=2, ms=7)
ax.axhline(true_energy, ls='--', color='k', lw=1.5, label=f'True ({true_energy:.1f})')
ax.set_xlabel("STLSQ Threshold")
ax.set_ylabel("Rollout Energy")
ax.set_title("Energy vs Threshold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Library sweep: D_KY as heatmap
ax = axes[1, 0]
n_modes_list = sorted(set(k[0] for k in library_results))
degrees_list  = sorted(set(k[1] for k in library_results))
grid = np.full((len(degrees_list), len(n_modes_list)), np.nan)
for i, deg in enumerate(degrees_list):
    for j, nm in enumerate(n_modes_list):
        if (nm, deg) in library_results:
            grid[i, j] = library_results[(nm, deg)]["D_KY"]
im = ax.imshow(grid, aspect='auto', cmap='viridis')
ax.set_xticks(range(len(n_modes_list))); ax.set_xticklabels(n_modes_list)
ax.set_yticks(range(len(degrees_list))); ax.set_yticklabels(degrees_list)
ax.set_xlabel("n_modes"); ax.set_ylabel("Poly degree")
ax.set_title("D_KY (library sweep)")
plt.colorbar(im, ax=ax)

# Library sweep: stability
ax = axes[1, 1]
grid_stab = np.zeros((len(degrees_list), len(n_modes_list)))
for i, deg in enumerate(degrees_list):
    for j, nm in enumerate(n_modes_list):
        if (nm, deg) in library_results:
            grid_stab[i, j] = 1.0 if library_results[(nm, deg)]["stable"] else 0.0
im2 = ax.imshow(grid_stab, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(n_modes_list))); ax.set_xticklabels(n_modes_list)
ax.set_yticks(range(len(degrees_list))); ax.set_yticklabels(degrees_list)
ax.set_xlabel("n_modes"); ax.set_ylabel("Poly degree")
ax.set_title("Stable Rollout (library sweep)")

# Deriv method comparison
ax = axes[1, 2]
methods = list(deriv_results.keys())
vals = [deriv_results[m]["D_KY"] for m in methods]
bars = ax.bar(range(len(methods)), vals, color=['C0', 'C1', 'C2'])
ax.axhline(kaplan_yorke(le_true), ls='--', color='k', lw=1.5)
ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("D_KY")
ax.set_title("Derivative Method: D_KY")
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("T8-T10: SINDy Sweeps (threshold, library, deriv method)", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT8_sindy_sweep.png", dpi=120)
plt.close()
print("  Saved: figures/figT8_sindy_sweep.png")

log_event("T8", "script_complete",
          config={"thresholds": THRESHOLDS, "degrees": DEGREES, "n_modes": N_MODES_LIST},
          metrics={"n_threshold_runs": len(THRESHOLDS),
                   "n_library_runs": len(library_results),
                   "n_deriv_runs": len(METHODS)})
print("\nT8/T9/T10 complete.")
