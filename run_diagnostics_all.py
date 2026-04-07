"""
run_diagnostics_all.py - Full Diagnostics for All Surrogate Models
====================================================================
Generates rollout trajectories for every available surrogate and runs
the full diagnostics suite (power spectrum, joint PDF, autocorrelation,
invariant measure stats, W1 distance vs true KSE).

Fills STATUS.md gap: existing diagnostics.pkl only covers True KSE,
NODE-Std-MSE, and SINDy PI.  This script adds:
  - NODE-Stab-JAC
  - T4 constrained-A variants (negdef, diag_neg)
  - Latent NODE (POD d=10)
  - SINDy PI (re-verifies)

Outputs:
  data/diagnostics_all.pkl          -- full results dict
  figures/figDiag_power_spectra.png
  figures/figDiag_joint_pdfs.png
  figures/figDiag_autocorr.png
  figures/figDiag_w1_table.png
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
from diagnostics import compare_systems, print_comparison_table
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

DT = 0.25
N_DIAG_STEPS = 4000   # 1000 time units — long enough for good statistics

solver = KSSolver(L=22.0, N=64, dt=DT)

print("Loading reference data...")
traj_analysis = np.load("data/traj_analysis.npy")
le_true = np.load("data/lyapunov_exponents_full.npy")

# Use the first 4000 steps of traj_analysis as "true KSE"
true_traj = traj_analysis[:N_DIAG_STEPS].astype(np.float64)
u0 = traj_analysis[100].astype(np.float64)   # start from attractor

systems = {"True KSE": true_traj}


# ── Generic RK4 rollout ────────────────────────────────────────────────────────
def rollout_rk4(rhs_fn, params, u0, n_steps=N_DIAG_STEPS):
    step = jax.jit(lambda u: _rk4(rhs_fn, params, u))
    u = jnp.array(u0, dtype=jnp.float64)
    traj = []
    for i in range(n_steps):
        u = step(u)
        if jnp.any(jnp.isnan(u)) or jnp.linalg.norm(u) > 1e6:
            print(f"    Diverged at step {i}")
            break
        traj.append(np.array(u))
    return np.array(traj) if traj else None

def _rk4(rhs_fn, params, u):
    k1 = rhs_fn(params, u)
    k2 = rhs_fn(params, u + DT/2*k1)
    k3 = rhs_fn(params, u + DT/2*k2)
    k4 = rhs_fn(params, u + DT*k3)
    return u + DT/6*(k1 + 2*k2 + 2*k3 + k4)


# ── NODE-Std-MSE ─────────────────────────────────────────────────────────────
print("\nRolling out NODE-Std-MSE...")
try:
    with open("data/node_standard_mse.pkl", "rb") as f:
        m = pickle.load(f)
    traj = rollout_rk4(standard_node_rhs, m["params"], u0)
    if traj is not None:
        systems["NODE-Std-MSE"] = traj
        print(f"  OK: {len(traj)} steps, energy={float(np.mean(np.sum(traj**2,axis=1))):.2f}")
except Exception as e:
    print(f"  Failed: {e}")


# ── NODE-Stab-JAC ─────────────────────────────────────────────────────────────
print("\nRolling out NODE-Stab-JAC...")
try:
    with open("data/node_stabilized_jac.pkl", "rb") as f:
        m = pickle.load(f)
    traj = rollout_rk4(stabilized_node_rhs, m["params"], u0)
    if traj is not None and len(traj) > 200:
        systems["NODE-Stab-JAC"] = traj
        print(f"  OK: {len(traj)} steps, energy={float(np.mean(np.sum(traj**2,axis=1))):.2f}")
    else:
        print(f"  Diverged after {len(traj) if traj is not None else 0} steps — skipping")
except Exception as e:
    print(f"  Failed: {e}")


# ── T4 negdef variant (stable) ────────────────────────────────────────────────
print("\nRolling out NODE-negdef (T4)...")
try:
    with open("data/constrained_a_results.pkl", "rb") as f:
        t4 = pickle.load(f)
    neg_params = t4["negdef"]["params"]
    EPS = 1e-3
    def rhs_neg(params, u):
        B = params["B"]
        A = -(B.T @ B + EPS * jnp.eye(B.shape[0]))
        return u @ A.T + mlp_forward(params["mlp"], u)
    traj = rollout_rk4(rhs_neg, neg_params, u0)
    if traj is not None:
        systems["NODE-negdef"] = traj
        print(f"  OK: {len(traj)} steps, energy={float(np.mean(np.sum(traj**2,axis=1))):.2f}")
except Exception as e:
    print(f"  Failed: {e}")


# ── T4 diag_neg variant (stable) ──────────────────────────────────────────────
print("\nRolling out NODE-diag-neg (T4)...")
try:
    diag_params = t4["diag_neg"]["params"]
    def rhs_diag(params, u):
        diag_A = -jax.nn.softplus(params["d_vec"])
        return diag_A * u + mlp_forward(params["mlp"], u)
    traj = rollout_rk4(rhs_diag, diag_params, u0)
    if traj is not None:
        systems["NODE-diag-neg"] = traj
        print(f"  OK: {len(traj)} steps, energy={float(np.mean(np.sum(traj**2,axis=1))):.2f}")
except Exception as e:
    print(f"  Failed: {e}")


# ── Latent NODE (POD d=10) ────────────────────────────────────────────────────
print("\nRolling out Latent NODE (d=10)...")
try:
    from latent_node import (
        fit_pod_autoencoder, pod_encode, pod_decode,
        init_latent_ode, train_latent_ode,
        prepare_latent_data_pod, rollout_latent_node,
    )
    traj_train = np.load("data/traj_train.npy")
    D_LAT = 10
    pod = fit_pod_autoencoder(traj_train, D_LAT)
    encode_fn = lambda u: pod_encode(pod, u)
    decode_fn = lambda h: pod_decode(pod, h)
    lat_data = prepare_latent_data_pod(traj_train, pod, solver, subsample=2)
    key_ode = jax.random.PRNGKey(42)
    ode_params = init_latent_ode(key_ode, d=D_LAT, hidden=128, n_layers=3)
    print("  Training latent ODE (300 epochs)...")
    ode_params, _ = train_latent_ode(
        ode_params, lat_data["h"], lat_data["dhdt"],
        n_epochs=300, batch_size=256, key=key_ode)
    traj, _ = rollout_latent_node(
        pod, ode_params, u0, n_steps=N_DIAG_STEPS, dt=DT,
        encode_fn=encode_fn, decode_fn=decode_fn)
    if not np.any(np.isnan(traj)):
        systems["Latent-NODE-d10"] = traj
        print(f"  OK: {len(traj)} steps, energy={float(np.mean(np.sum(traj**2,axis=1))):.2f}")
except Exception as e:
    print(f"  Failed: {e}")
    import traceback; traceback.print_exc()


# ── SINDy PI ──────────────────────────────────────────────────────────────────
print("\nRolling out SINDy PI...")
try:
    with open("data/sindy_model.pkl", "rb") as f:
        sindy_model = pickle.load(f)
    traj = sindy_model.integrate(u0, n_steps=N_DIAG_STEPS, dt=DT)
    if not np.any(np.isnan(traj)) and not np.any(np.isinf(traj)):
        systems["SINDy-PI"] = traj
        print(f"  OK: {len(traj)} steps, energy={float(np.mean(np.sum(traj**2,axis=1))):.2f}")
    else:
        print("  NaN/Inf in rollout")
except Exception as e:
    print(f"  Failed: {e}")


# ── Run diagnostics ────────────────────────────────────────────────────────────
print(f"\nRunning diagnostics on {len(systems)} systems: {list(systems.keys())}")
diag_results = compare_systems(systems, L=22.0, dt=DT)

with open("data/diagnostics_all.pkl", "wb") as f:
    pickle.dump(diag_results, f)
print("\nSaved: data/diagnostics_all.pkl")


# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'System':<22} {'Energy':>8} {'RMS':>7} {'Skew':>7} {'W1_true':>9}")
print("-"*75)
for name, r in diag_results.items():
    s = r["stats"]
    w1 = r.get("wasserstein_vs_true", 0.0 if name == "True KSE" else float('nan'))
    print(f"{name:<22} {s['energy']:>8.2f} {s['rms']:>7.4f} "
          f"{s['skewness']:>7.3f} {w1:>9.4f}")
print("="*75)


# ── Figures ───────────────────────────────────────────────────────────────────
print("\nGenerating diagnostic figures...")
colors_map = {
    "True KSE": "k", "NODE-Std-MSE": "C0", "NODE-Stab-JAC": "C1",
    "NODE-negdef": "C2", "NODE-diag-neg": "C3",
    "Latent-NODE-d10": "C4", "SINDy-PI": "C5",
}

# --- Power spectra ---
fig, ax = plt.subplots(figsize=(8, 5))
for name, r in diag_results.items():
    q, E = r["power_spectrum"]
    lw = 2.5 if name == "True KSE" else 1.5
    ls = '-' if name == "True KSE" else '--'
    ax.semilogy(q[1:], E[1:], lw=lw, ls=ls,
                color=colors_map.get(name, "gray"), label=name)
ax.set_xlabel("Wavenumber q")
ax.set_ylabel("E(q) = |û_q|²")
ax.set_title("Spatial Power Spectra: True KSE vs All Surrogates")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/figDiag_power_spectra.png", dpi=120)
plt.close()
print("  Saved: figures/figDiag_power_spectra.png")

# --- Joint PDFs (u_x, u_xx) ---
n_sys = len(diag_results)
ncols = min(n_sys, 4)
nrows = (n_sys + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
axes = np.array(axes).flatten() if n_sys > 1 else [axes]
for idx, (name, r) in enumerate(diag_results.items()):
    ax = axes[idx]
    ux_e, uxx_e, pdf = r["joint_pdf"]
    vmax = np.percentile(pdf, 98)
    ax.pcolormesh(ux_e, uxx_e, pdf.T, cmap='hot_r', vmin=0, vmax=vmax,
                  shading='auto', rasterized=True)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("u_x"); ax.set_ylabel("u_xx")
for ax in axes[len(diag_results):]:
    ax.set_visible(False)
plt.suptitle("Joint PDF (u_x, u_xx) — Linot 2022 diagnostic", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figDiag_joint_pdfs.png", dpi=120)
plt.close()
print("  Saved: figures/figDiag_joint_pdfs.png")

# --- Autocorrelation ---
fig, ax = plt.subplots(figsize=(8, 5))
for name, r in diag_results.items():
    ac = r["autocorr"]
    t_lag = np.arange(len(ac)) * DT
    lw = 2.5 if name == "True KSE" else 1.5
    ax.plot(t_lag[:200], ac[:200], lw=lw,
            color=colors_map.get(name, "gray"), label=name)
ax.axhline(0, ls='--', color='gray', lw=0.8)
ax.set_xlabel("Lag time")
ax.set_ylabel("Autocorrelation")
ax.set_title("Temporal Autocorrelation")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/figDiag_autocorr.png", dpi=120)
plt.close()
print("  Saved: figures/figDiag_autocorr.png")

# --- W1 bar chart ---
fig, ax = plt.subplots(figsize=(8, 4))
names_w1 = [n for n in diag_results if "wasserstein_vs_true" in diag_results[n]]
w1_vals  = [diag_results[n]["wasserstein_vs_true"] for n in names_w1]
ax.bar(range(len(names_w1)), w1_vals,
       color=[colors_map.get(n, "gray") for n in names_w1])
ax.set_xticks(range(len(names_w1)))
ax.set_xticklabels(names_w1, rotation=20, ha='right', fontsize=9)
ax.set_ylabel("W1 distance (vs True KSE)")
ax.set_title("Wasserstein-1 Distance from True KSE Marginals")
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("figures/figDiag_w1.png", dpi=120)
plt.close()
print("  Saved: figures/figDiag_w1.png")

log_event("diagnostics_all", "script_complete",
          config={"systems": list(systems.keys()), "n_steps": N_DIAG_STEPS},
          metrics={"n_systems": len(diag_results)})
print("\nDiagnostics complete.")
