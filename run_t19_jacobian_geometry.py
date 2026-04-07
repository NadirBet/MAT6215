"""
run_t19_jacobian_geometry.py - Task T19: Local Jacobian Geometry
=================================================================
Computes local Jacobians of surrogate RHS functions along attractor
trajectories, measures their singular value structure, and compares
with the true KSE linearized operator.

Key diagnostics:
  1. Spectrum of singular values σ_i(J_f(u)) sampled along trajectory
     - True KSE: should show stiff structure (large range σ_1/σ_N >> 1)
     - Surrogates: typically much flatter spectrum
  2. Distribution of ||J_f||_F (Frobenius norm of Jacobian)
  3. Distribution of spectral radius ρ(J_f) — determines local growth
  4. Alignment: angle between leading right singular vector of J_f
     and leading Lyapunov vector
  5. For latent AE: local metric distortion (how much the encoder
     warps distances)

Surrogates tested:
  - True KSE (analytical Jacobian via AD through ETD-RK4 or spectral RHS)
  - NODE-Std-MSE
  - NODE-Stab-JAC
  - NODE-negdef (T4)

Outputs:
  data/jacobian_geometry_results.pkl
  figures/figT19_jacobian_geometry.png
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
from neural_ode import stabilized_node_rhs, standard_node_rhs, mlp_forward
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_SAMPLES = 300   # sample Jacobians at this many attractor points
DT = 0.25

solver = KSSolver(L=22.0, N=64, dt=DT)

print("Loading data...")
traj_analysis = np.load("data/traj_analysis.npy")
le_true       = np.load("data/lyapunov_exponents_full.npy")

# Attractor sample points (well after warmup)
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(traj_analysis) - 10, size=N_SAMPLES, replace=False) + 5
u_samples = traj_analysis[sample_idx].astype(np.float64)


# ── Jacobian sampling utility ──────────────────────────────────────────────────
def sample_jacobians(rhs_fn, params, u_samples, batch=20):
    """
    Compute Jacobian J_f(u_i) for each sample via jax.jacobian.
    Returns array of shape (N_SAMPLES, N, N).
    Batches to avoid memory pressure.
    """
    N = u_samples.shape[1]
    jac_fn = jax.jit(jax.jacobian(lambda u: rhs_fn(params, u)))
    jacs = np.zeros((len(u_samples), N, N))
    for i in range(0, len(u_samples), batch):
        end = min(i + batch, len(u_samples))
        for j in range(i, end):
            jacs[j] = np.array(jac_fn(jnp.array(u_samples[j])))
        if (i // batch) % 5 == 0:
            print(f"    Jacobians: {end}/{len(u_samples)}")
    return jacs


def jacobian_stats(jacs):
    """
    Compute singular value statistics from Jacobian samples.
    Returns dict with:
      - sv_mean: mean singular value spectrum (N,)
      - sv_std:  std across samples (N,)
      - frobenius: ||J||_F per sample (N_samples,)
      - spectral_radius: ρ(J) = max |eigenvalue| per sample (N_samples,)
      - condition_number: σ_1 / σ_N per sample (N_samples,)
    """
    N_s, N, _ = jacs.shape
    svs_all = np.zeros((N_s, N))
    frob = np.zeros(N_s)
    spec_rad = np.zeros(N_s)

    for i, J in enumerate(jacs):
        try:
            sv = np.linalg.svd(J, compute_uv=False)
            svs_all[i] = sv
            frob[i] = float(np.sqrt(np.sum(J**2)))
            eigvals = np.linalg.eigvals(J)
            spec_rad[i] = float(np.max(np.abs(eigvals)))
        except Exception:
            svs_all[i] = np.nan
            frob[i] = np.nan
            spec_rad[i] = np.nan

    cond = svs_all[:, 0] / (svs_all[:, -1] + 1e-14)

    return {
        "sv_mean": np.nanmean(svs_all, axis=0),
        "sv_std":  np.nanstd(svs_all, axis=0),
        "sv_all":  svs_all,
        "frobenius": frob,
        "spectral_radius": spec_rad,
        "condition_number": cond,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# System 1: True KSE (RHS in physical space via rhs_physical)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("System: True KSE")
print("="*55)
t0 = time.time()

# True KSE RHS in physical space: u_t = -u*u_x - u_xx - u_xxxx
# Use solver.rhs_physical which computes this exactly
def true_rhs_physical(_, u):
    """True KSE RHS in physical space (pseudo wrapper for jacobian_fn)."""
    return solver.rhs_physical(u)

# Note: params=None, rhs takes (params, u) interface
jacs_true = sample_jacobians(true_rhs_physical, None, u_samples)
stats_true = jacobian_stats(jacs_true)
print(f"  True KSE: ||J||_F mean={np.nanmean(stats_true['frobenius']):.3f}  "
      f"σ_1 mean={stats_true['sv_mean'][0]:.3f}  "
      f"cond mean={np.nanmean(stats_true['condition_number']):.1e}  "
      f"ρ mean={np.nanmean(stats_true['spectral_radius']):.3f}")
log_event("T19", "system_done", config={"system": "true_kse"},
          metrics={"frob_mean": float(np.nanmean(stats_true["frobenius"])),
                   "sv1_mean": float(stats_true["sv_mean"][0]),
                   "cond_mean": float(np.nanmean(stats_true["condition_number"])),
                   "runtime": time.time()-t0})

jac_results = {"true_kse": stats_true}


# ═══════════════════════════════════════════════════════════════════════════════
# System 2: NODE-Std-MSE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("System: NODE-Std-MSE")
print("="*55)
t0 = time.time()
try:
    with open("data/node_standard_mse.pkl", "rb") as f:
        m = pickle.load(f)
    params_std = m["params"]
    jacs_std = sample_jacobians(standard_node_rhs, params_std, u_samples)
    stats_std = jacobian_stats(jacs_std)
    jac_results["node_std_mse"] = stats_std
    print(f"  NODE-Std-MSE: ||J||_F mean={np.nanmean(stats_std['frobenius']):.3f}  "
          f"σ_1 mean={stats_std['sv_mean'][0]:.3f}  "
          f"cond mean={np.nanmean(stats_std['condition_number']):.1e}  "
          f"ρ mean={np.nanmean(stats_std['spectral_radius']):.3f}")
    log_event("T19", "system_done", config={"system": "node_std_mse"},
              metrics={"frob_mean": float(np.nanmean(stats_std["frobenius"])),
                       "sv1_mean": float(stats_std["sv_mean"][0]),
                       "runtime": time.time()-t0})
except Exception as e:
    print(f"  Failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# System 3: NODE-Stab-JAC
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("System: NODE-Stab-JAC")
print("="*55)
t0 = time.time()
try:
    with open("data/node_stabilized_jac.pkl", "rb") as f:
        m = pickle.load(f)
    params_jac = m["params"]
    jacs_jac = sample_jacobians(stabilized_node_rhs, params_jac, u_samples)
    stats_jac = jacobian_stats(jacs_jac)
    jac_results["node_stab_jac"] = stats_jac
    print(f"  NODE-Stab-JAC: ||J||_F mean={np.nanmean(stats_jac['frobenius']):.3f}  "
          f"σ_1 mean={stats_jac['sv_mean'][0]:.3f}  "
          f"cond mean={np.nanmean(stats_jac['condition_number']):.1e}  "
          f"ρ mean={np.nanmean(stats_jac['spectral_radius']):.3f}")
    log_event("T19", "system_done", config={"system": "node_stab_jac"},
              metrics={"frob_mean": float(np.nanmean(stats_jac["frobenius"])),
                       "sv1_mean": float(stats_jac["sv_mean"][0]),
                       "runtime": time.time()-t0})
except Exception as e:
    print(f"  Failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# System 4: NODE-negdef (T4, stable)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("System: NODE-negdef (T4)")
print("="*55)
t0 = time.time()
try:
    with open("data/constrained_a_results.pkl", "rb") as f:
        t4 = pickle.load(f)
    neg_params = t4["negdef"]["params"]
    EPS = 1e-3
    def rhs_neg(params, u):
        B = params["B"]
        A = -(B.T @ B + EPS * jnp.eye(B.shape[0]))
        return u @ A.T + mlp_forward(params["mlp"], u)
    jacs_neg = sample_jacobians(rhs_neg, neg_params, u_samples)
    stats_neg = jacobian_stats(jacs_neg)
    jac_results["node_negdef"] = stats_neg
    print(f"  NODE-negdef: ||J||_F mean={np.nanmean(stats_neg['frobenius']):.3f}  "
          f"σ_1 mean={stats_neg['sv_mean'][0]:.3f}  "
          f"cond mean={np.nanmean(stats_neg['condition_number']):.1e}  "
          f"ρ mean={np.nanmean(stats_neg['spectral_radius']):.3f}")
    log_event("T19", "system_done", config={"system": "node_negdef"},
              metrics={"frob_mean": float(np.nanmean(stats_neg["frobenius"])),
                       "sv1_mean": float(stats_neg["sv_mean"][0]),
                       "runtime": time.time()-t0})
except Exception as e:
    print(f"  Failed: {e}")


# ── Save ───────────────────────────────────────────────────────────────────────
with open("data/jacobian_geometry_results.pkl", "wb") as f:
    pickle.dump(jac_results, f)
print("\nSaved: data/jacobian_geometry_results.pkl")


# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'System':<20} {'||J||_F':>9} {'σ_1':>8} {'σ_N':>8} {'cond':>10} {'ρ(J)':>8}")
print("-"*75)
for name, s in jac_results.items():
    frob = np.nanmean(s["frobenius"])
    sv1  = s["sv_mean"][0]
    svN  = s["sv_mean"][-1]
    cond = np.nanmean(s["condition_number"])
    rho  = np.nanmean(s["spectral_radius"])
    print(f"{name:<20} {frob:>9.3f} {sv1:>8.3f} {svN:>8.5f} {cond:>10.2e} {rho:>8.3f}")
print("="*75)


# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
colors = {"true_kse": "k", "node_std_mse": "C0", "node_stab_jac": "C1",
          "node_negdef": "C2"}
labels = {"true_kse": "True KSE", "node_std_mse": "NODE-Std-MSE",
          "node_stab_jac": "NODE-Stab-JAC", "node_negdef": "NODE-negdef"}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Mean singular value spectrum
ax = axes[0, 0]
N = 64
idx_arr = np.arange(1, N + 1)
for name, s in jac_results.items():
    sv = s["sv_mean"]
    ax.semilogy(idx_arr[:N], sv[:N], '-', color=colors[name],
                label=labels[name], lw=2)
ax.set_xlabel("Singular value index")
ax.set_ylabel("Mean σ_i")
ax.set_title("Mean Singular Value Spectrum of J_f(u)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Frobenius norm distribution
ax = axes[0, 1]
for name, s in jac_results.items():
    frob = s["frobenius"][~np.isnan(s["frobenius"])]
    ax.hist(frob, bins=30, density=True, alpha=0.5,
            color=colors[name], label=labels[name])
ax.set_xlabel("||J_f(u)||_F")
ax.set_ylabel("PDF")
ax.set_title("Frobenius Norm Distribution")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Spectral radius distribution
ax = axes[0, 2]
for name, s in jac_results.items():
    rho = s["spectral_radius"][~np.isnan(s["spectral_radius"])]
    ax.hist(rho, bins=30, density=True, alpha=0.5,
            color=colors[name], label=labels[name])
ax.set_xlabel("ρ(J_f) = max|eigenvalue|")
ax.set_ylabel("PDF")
ax.set_title("Spectral Radius Distribution")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Condition number distribution
ax = axes[1, 0]
for name, s in jac_results.items():
    cond = s["condition_number"][~np.isnan(s["condition_number"])]
    cond = cond[np.isfinite(cond)]
    ax.hist(np.log10(cond + 1), bins=30, density=True, alpha=0.5,
            color=colors[name], label=labels[name])
ax.set_xlabel("log10(condition number)")
ax.set_ylabel("PDF")
ax.set_title("Jacobian Condition Number Distribution")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Top-5 singular values: mean ± std comparison
ax = axes[1, 1]
n_show = min(8, N)
x = np.arange(n_show)
width = 0.8 / len(jac_results)
for k, (name, s) in enumerate(jac_results.items()):
    sv_m = s["sv_mean"][:n_show]
    sv_s = s["sv_std"][:n_show]
    ax.errorbar(x + k * width, sv_m, yerr=sv_s, fmt='o-',
                color=colors[name], label=labels[name], lw=2, ms=5, capsize=3)
ax.set_xlabel("Singular value index i")
ax.set_ylabel("σ_i mean ± std")
ax.set_title("Top Singular Values (mean ± std)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Summary bar: mean Frobenius norm
ax = axes[1, 2]
names_plot = list(jac_results.keys())
frob_means = [np.nanmean(jac_results[n]["frobenius"]) for n in names_plot]
frob_stds  = [np.nanstd(jac_results[n]["frobenius"])  for n in names_plot]
ax.bar(range(len(names_plot)), frob_means, yerr=frob_stds,
       color=[colors[n] for n in names_plot], alpha=0.8, capsize=4)
ax.set_xticks(range(len(names_plot)))
ax.set_xticklabels([labels[n] for n in names_plot], rotation=15, ha='right', fontsize=8)
ax.set_ylabel("||J_f||_F (mean ± std)")
ax.set_title("RHS Jacobian Frobenius Norm")
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("T19: Local Jacobian Geometry — True KSE vs Surrogates", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT19_jacobian_geometry.png", dpi=120)
plt.close()
print("  Saved: figures/figT19_jacobian_geometry.png")

log_event("T19", "script_complete",
          config={"n_samples": N_SAMPLES, "systems": list(jac_results.keys())},
          metrics={n: {"frob_mean": float(np.nanmean(s["frobenius"]))}
                   for n, s in jac_results.items()})
print("\nT19 complete.")
