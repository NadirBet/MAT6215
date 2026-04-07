"""
run_t11_discrete_sindy.py - Task T11: Discrete-Time SINDy
==========================================================
Learns a sparse map a_{n+1} = F(a_n) without computing time derivatives.
Avoids all FD noise — directly regresses next state from current state.

Comparison:
  - Continuous SINDy (T8 best config): needs dh/dt estimation
  - Discrete SINDy (this): learns F: R^r -> R^r directly from snapshot pairs

Library for discrete map: same polynomial library Θ(a) as continuous SINDy.
Regression: a_{n+1} ≈ Θ(a_n) * Ξ  (STLSQ on (a_n, a_{n+1}) pairs)

For Lyapunov: discrete map Lyapunov = log|eigenvalue of Jacobian of F| / dt

Sweep: threshold, subsample (stride), n_modes, degree

Outputs:
  data/discrete_sindy_results.pkl
  figures/figT11_discrete_sindy.png
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

from ks_solver import KSSolver
from sindy import (
    compute_galerkin_basis, project_trajectory,
    polynomial_library, stlsq,
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


# ── Discrete SINDy Model ────────────────────────────────────────────────────────

class DiscreteSINDyModel:
    """
    Sparse discrete map: a_{n+1} = Θ(a_n) @ Xi  (STLSQ)

    Rollout: iterate map starting from encoded u0.
    Lyapunov: Benettin via JAX JVP through the map step.
    """
    def __init__(self, Xi, Phi, u_mean, feature_names, degree=2, threshold=0.01):
        self.Xi = Xi
        self.Phi = Phi
        self.u_mean = u_mean
        self.feature_names = feature_names
        self.degree = degree
        self.threshold = threshold
        self.r = Phi.shape[1]

    def step_modes(self, a):
        """One map step in modal coordinates."""
        Theta, _ = polynomial_library(a[None, :], degree=self.degree)
        return (Theta @ self.Xi)[0]

    def integrate(self, u0, n_steps):
        """Rollout in physical space."""
        a = (u0 - self.u_mean) @ self.Phi
        traj = np.zeros((n_steps, len(u0)))
        for i in range(n_steps):
            Theta, _ = polynomial_library(a[None, :], degree=self.degree)
            a = (Theta @ self.Xi)[0]
            u = a @ self.Phi.T + self.u_mean
            if np.any(np.isnan(u)) or np.linalg.norm(u) > 1e6:
                return traj[:i]
            traj[i] = u
        return traj

    def lyapunov(self, a0, n_steps=800):
        """Benettin via JVP through map step. Returns LE = log(R)/dt per step."""
        r = self.r

        def step_jax(a):
            Theta, _ = polynomial_library(a[None, :], degree=self.degree)
            return jnp.array((Theta @ self.Xi)[0])

        step_jit = jax.jit(step_jax)
        a0j = jnp.array(a0, dtype=jnp.float64)
        Q0  = jnp.eye(r, dtype=jnp.float64)
        log0 = jnp.zeros(r, dtype=jnp.float64)

        def benettin(carry, _):
            a, Q, ls = carry
            Q_raw = jax.vmap(lambda q: jax.jvp(step_jit, (a,), (q,))[1],
                             in_axes=1, out_axes=1)(Q)
            a_n = step_jit(a)
            Q_n, R = jnp.linalg.qr(Q_raw)
            s = jnp.sign(jnp.diag(R))
            Q_n = Q_n * s[None, :]; R = R * s[:, None]
            return (a_n, Q_n, ls + jnp.log(jnp.abs(jnp.diag(R)))), None

        try:
            (_, _, log_tot), _ = jax.lax.scan(
                benettin, (a0j, Q0, log0), None, length=n_steps)
            # LE in continuous time = log(R) / (n_steps * dt)
            return np.array(log_tot / (n_steps * solver.dt))
        except Exception as e:
            print(f"    Lyapunov failed: {e}")
            return None


def fit_discrete_sindy(Phi, u_mean, traj, degree=2, threshold=0.01,
                       stride=1, verbose=False):
    """
    Fit discrete SINDy from snapshot pairs.
    stride=1: consecutive pairs (a_n, a_{n+1})
    stride=k: k-step pairs (a_n, a_{n+k})  — matches tau sweep
    """
    a_traj = project_trajectory(traj, Phi, u_mean)
    a_n  = a_traj[:-stride:stride]
    a_n1 = a_traj[stride::stride]

    Theta, names = polynomial_library(a_n, degree=degree, include_bias=True)
    Xi, active = stlsq(Theta, a_n1, threshold=threshold, max_iter=20, verbose=verbose)
    return DiscreteSINDyModel(Xi, Phi, u_mean, names, degree=degree, threshold=threshold), active


# ── Baseline: continuous SINDy best (degree=2, r=8, thresh=0.01) ────────────
print("\nFitting baseline continuous SINDy (deg=2, r=8, thresh=0.01) for comparison...")
Phi8, u_mean8 = compute_galerkin_basis(solver, n_modes=8, traj=traj_train)
a0_test = (traj_test[200] - u_mean8) @ Phi8

try:
    with open("data/sindy_sweep_results.pkl", "rb") as f:
        t8_data = pickle.load(f)
    cont_ref = t8_data["threshold_sweep"].get(0.01, {})
    cont_L1 = cont_ref.get("L1", float('nan'))
    cont_DKY = cont_ref.get("D_KY", float('nan'))
    cont_stable = cont_ref.get("stable", False)
    print(f"  Continuous SINDy (from T8): L1={cont_L1:+.4f} D_KY={cont_DKY:.2f} stable={cont_stable}")
except Exception:
    cont_L1 = cont_DKY = float('nan'); cont_stable = False
    print("  T8 results not available yet")


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP 1 — Threshold sweep (discrete, stride=1, deg=2, r=8)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SWEEP 1: Threshold sweep (discrete, stride=1, deg=2, r=8)")
print("="*60)

THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
thresh_results = {}

for thresh in THRESHOLDS:
    t0 = time.time()
    print(f"\n  threshold={thresh}:")
    model, active = fit_discrete_sindy(Phi8, u_mean8, traj_train,
                                        degree=2, threshold=thresh, stride=1)
    n_active = int(active.sum())
    print(f"    Active: {n_active}/{active.size}")

    # Rollout
    traj_out = model.integrate(traj_test[0], n_steps=2000)
    stable = len(traj_out) == 2000
    energy = float(np.mean(np.sum(traj_out**2, axis=1))) if len(traj_out) > 0 else float('nan')
    print(f"    Rollout: stable={stable}, energy={energy:.2f}")

    # Lyapunov
    le = model.lyapunov(a0_test, n_steps=800)
    if le is not None:
        dky = kaplan_yorke(le); h_ks = float(np.sum(le[le>0])); n_pos = int(np.sum(le>0))
        print(f"    Lyapunov: L1={le[0]:+.4f}, n_pos={n_pos}, D_KY={dky:.2f}")
    else:
        dky = h_ks = float('nan'); n_pos = 0

    thresh_results[thresh] = {
        "threshold": thresh, "n_active": n_active,
        "stable": stable, "energy": energy,
        "lyapunov": le, "D_KY": dky, "h_KS": h_ks, "n_pos": n_pos,
        "L1": float(le[0]) if le is not None else float('nan'),
        "runtime": time.time() - t0,
    }
    log_event("T11", "discrete_thresh",
              config={"threshold": thresh, "stride": 1, "degree": 2, "n_modes": 8},
              metrics={"n_active": n_active, "stable": stable, "D_KY": dky, "h_KS": h_ks})


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP 2 — Stride sweep (tau for discrete map — matches T6)
# Threshold=0.01, deg=2, r=8, stride in [1,2,4,8,16]
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SWEEP 2: Stride sweep (thresh=0.01, deg=2, r=8)")
print("="*60)

STRIDES = [1, 2, 4, 8, 16]
stride_results = {}

for stride in STRIDES:
    tau = stride * solver.dt
    t0 = time.time()
    print(f"\n  stride={stride} (tau={tau:.2f}):")
    model, active = fit_discrete_sindy(Phi8, u_mean8, traj_train,
                                        degree=2, threshold=0.01, stride=stride)
    n_pts = len(traj_train[:-stride:stride])
    print(f"    Training pairs: {n_pts}, active: {active.sum()}/{active.size}")

    traj_out = model.integrate(traj_test[0], n_steps=2000 // stride)
    stable = len(traj_out) == 2000 // stride
    energy = float(np.mean(np.sum(traj_out**2, axis=1))) if len(traj_out) > 0 else float('nan')

    # Lyapunov: log(R)/tau  (tau = stride * dt)
    le = model.lyapunov(a0_test, n_steps=800)
    if le is not None:
        # model.lyapunov divides by n_steps*dt but each "step" is stride*dt
        # The model was trained to predict a_{n+stride}, so the map is F^stride
        # log|DF^stride| / (stride*dt) = true LE estimate
        # But we already divide by dt inside lyapunov(), so need to correct:
        le_corr = le * solver.dt / tau  # correct for stride
        dky = kaplan_yorke(le_corr); h_ks = float(np.sum(le_corr[le_corr>0]))
        n_pos = int(np.sum(le_corr > 0))
        print(f"    Lyapunov: L1={le_corr[0]:+.4f}, n_pos={n_pos}, D_KY={dky:.2f}")
    else:
        le_corr = None; dky = h_ks = float('nan'); n_pos = 0
    print(f"    Rollout: stable={stable}, energy={energy:.2f}")

    stride_results[stride] = {
        "stride": stride, "tau": tau, "n_active": int(active.sum()),
        "stable": stable, "energy": energy,
        "lyapunov": le_corr, "D_KY": dky, "h_KS": h_ks, "n_pos": n_pos,
        "L1": float(le_corr[0]) if le_corr is not None else float('nan'),
    }
    log_event("T11", "discrete_stride",
              config={"stride": stride, "tau": tau, "threshold": 0.01, "degree": 2},
              metrics={"n_active": int(active.sum()), "stable": stable,
                       "D_KY": dky, "h_KS": h_ks})


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP 3 — Library: degree vs n_modes (stride=1, thresh=0.01)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SWEEP 3: Library sweep (stride=1, thresh=0.01)")
print("="*60)

N_MODES_LIST = [4, 6, 8, 10, 12]
DEGREES = [2, 3]
library_results = {}

for n_modes in N_MODES_LIST:
    for degree in DEGREES:
        t0 = time.time()
        Phi, u_mean = compute_galerkin_basis(solver, n_modes=n_modes, traj=traj_train)
        a0 = (traj_test[200] - u_mean) @ Phi
        model, active = fit_discrete_sindy(Phi, u_mean, traj_train,
                                            degree=degree, threshold=0.01, stride=1)
        traj_out = model.integrate(traj_test[0], n_steps=2000)
        stable = len(traj_out) == 2000
        energy = float(np.mean(np.sum(traj_out**2, axis=1))) if len(traj_out) > 0 else float('nan')
        le = model.lyapunov(a0, n_steps=800)
        if le is not None:
            dky = kaplan_yorke(le); h_ks = float(np.sum(le[le>0])); n_pos = int(np.sum(le>0))
        else:
            dky = h_ks = float('nan'); n_pos = 0
        print(f"  r={n_modes}, deg={degree}: stable={stable} D_KY={dky:.2f}")
        library_results[(n_modes, degree)] = {
            "n_modes": n_modes, "degree": degree,
            "n_active": int(active.sum()), "stable": stable,
            "energy": energy, "D_KY": dky, "h_KS": h_ks, "n_pos": n_pos,
            "L1": float(le[0]) if le is not None else float('nan'),
        }


# ── Save ───────────────────────────────────────────────────────────────────────
results = {
    "threshold_sweep": thresh_results,
    "stride_sweep": stride_results,
    "library_sweep": library_results,
    "continuous_sindy_ref": {"L1": cont_L1, "D_KY": cont_DKY, "stable": cont_stable},
    "true": {"L1": le_true[0], "D_KY": kaplan_yorke(le_true),
             "h_KS": float(np.sum(le_true[le_true>0])), "energy": true_energy},
}
with open("data/discrete_sindy_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: data/discrete_sindy_results.pkl")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== Discrete vs Continuous SINDy (thresh=0.01, r=8, deg=2, stride=1) ===")
disc = thresh_results.get(0.01, {})
print(f"  Continuous (T8): L1={cont_L1:+.4f} D_KY={cont_DKY:.2f} stable={cont_stable}")
print(f"  Discrete  (T11): L1={disc.get('L1',float('nan')):+.4f} "
      f"D_KY={disc.get('D_KY',float('nan')):.2f} stable={disc.get('stable',False)}")

# ── Figures ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

threshs = sorted(thresh_results.keys())

# D_KY vs threshold
ax = axes[0, 0]
dky_disc = [thresh_results[t]["D_KY"] for t in threshs]
ax.semilogx(threshs, dky_disc, 'o-', color='C0', lw=2, ms=7, label='Discrete SINDy')
ax.axhline(cont_DKY, ls='--', color='C1', lw=1.5, label=f'Continuous (T8, {cont_DKY:.2f})')
ax.axhline(kaplan_yorke(le_true), ls='--', color='k', lw=1.5, label=f'True ({kaplan_yorke(le_true):.2f})')
ax.set_xlabel("Threshold"); ax.set_ylabel("D_KY")
ax.set_title("Discrete SINDy: D_KY vs Threshold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Stability vs threshold
ax = axes[0, 1]
stable_frac = [1 if thresh_results[t]["stable"] else 0 for t in threshs]
ax.semilogx(threshs, [thresh_results[t]["energy"] for t in threshs],
            'o-', color='C0', lw=2, ms=7, label='Discrete SINDy')
ax.axhline(true_energy, ls='--', color='k', lw=1.5, label=f'True ({true_energy:.1f})')
ax.set_xlabel("Threshold"); ax.set_ylabel("Rollout Energy")
ax.set_title("Rollout Energy vs Threshold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Stride comparison
ax = axes[0, 2]
taus = [stride_results[s]["tau"] for s in STRIDES]
ax.semilogx(taus, [stride_results[s]["D_KY"] for s in STRIDES], 's-', color='C2',
            lw=2, ms=7, label='Discrete SINDy')
ax.axhline(kaplan_yorke(le_true), ls='--', color='k', lw=1.5)
ax.set_xlabel("tau (stride * dt)"); ax.set_ylabel("D_KY")
ax.set_title("D_KY vs Data Spacing (tau)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Head-to-head: continuous vs discrete at each threshold
ax = axes[1, 0]
try:
    with open("data/sindy_sweep_results.pkl", "rb") as f:
        t8_data = pickle.load(f)
    cont_dky = [t8_data["threshold_sweep"].get(t, {}).get("D_KY", float('nan')) for t in threshs]
    ax.semilogx(threshs, cont_dky, 'o-', color='C1', lw=2, ms=7, label='Continuous')
except Exception:
    pass
ax.semilogx(threshs, dky_disc, 's--', color='C0', lw=2, ms=7, label='Discrete')
ax.axhline(kaplan_yorke(le_true), ls='--', color='k', lw=1.5)
ax.set_xlabel("Threshold"); ax.set_ylabel("D_KY")
ax.set_title("Cont. vs Discrete SINDy: D_KY")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Library sweep heatmap: stability
ax = axes[1, 1]
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
ax.set_title("Discrete SINDy: D_KY (library sweep)")
plt.colorbar(im, ax=ax)

# L1 comparison
ax = axes[1, 2]
l1_disc = [thresh_results[t]["L1"] for t in threshs]
ax.semilogx(threshs, l1_disc, 'o-', color='C0', lw=2, ms=7, label='Discrete')
ax.axhline(le_true[0], ls='--', color='k', lw=1.5, label=f'True ({le_true[0]:+.4f})')
ax.set_xlabel("Threshold"); ax.set_ylabel("L1 (leading LE)")
ax.set_title("Leading Lyapunov vs Threshold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle("T11: Discrete-Time SINDy (snapshot pairs, no derivative estimation)",
             fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT11_discrete_sindy.png", dpi=120)
plt.close()
print("  Saved: figures/figT11_discrete_sindy.png")

log_event("T11", "script_complete",
          config={"thresholds": THRESHOLDS, "strides": STRIDES},
          metrics={"n_threshold_runs": len(THRESHOLDS)})
print("\nT11 complete.")
