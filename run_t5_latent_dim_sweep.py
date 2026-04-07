"""
run_t5_latent_dim_sweep.py - Task T5: Latent Dimension Sweep
=============================================================
Sweeps latent dimension d for the reduced-manifold NODE.
Uses POD AE (linear, fast, reliable) as the representation layer
so we isolate the effect of d on dynamics fidelity.

Sweep: d = 4, 6, 8, 10, 12, 16

For each d:
  - POD reconstruction MSE
  - Latent ODE training loss
  - Rollout stability + energy
  - Lyapunov spectrum in latent space
  - D_KY, h_KS, n_pos
  - Runtime

Output:
  data/latent_dim_sweep.pkl
  figures/figT5_latent_dim_sweep.png
"""

import sys
sys.path = [p for p in sys.path if 'MAT6215' not in p]
sys.path.insert(0, '.')
import gpu_config  # sets XLA thread flags before jax import

import numpy as np
import jax
import jax.numpy as jnp
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ks_solver import KSSolver
from experiment_log import log_event
from latent_node import (
    fit_pod_autoencoder, pod_encode, pod_decode,
    init_latent_ode, train_latent_ode,
    prepare_latent_data_pod, rollout_latent_node,
    compute_latent_lyapunov, reconstruction_diagnostics, kaplan_yorke
)

jax.config.update("jax_enable_x64", True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
traj_train = np.load("data/traj_train.npy")
traj_test  = np.load("data/traj_analysis.npy")
le_true    = np.load("data/lyapunov_exponents_full.npy")
solver = KSSolver(L=22.0, N=64, dt=0.25)

true_energy = float(np.mean(np.sum(traj_test**2, axis=1)))
dky_true = kaplan_yorke(le_true)
h_ks_true = float(np.sum(le_true[le_true > 0]))
print(f"True KSE: energy={true_energy:.2f}, D_KY={dky_true:.2f}, h_KS={h_ks_true:.4f}")

D_SWEEP = [4, 6, 8, 10, 12, 16]
results = {}


def latent_lyap_needs_refinement(le, d):
    """Flag obviously low-convergence spectra before trusting D_KY."""
    if le is None:
        return False
    le = np.asarray(le)
    dky = kaplan_yorke(le)
    near_zero = np.any(np.abs(le) < 0.02)
    tiny_spectrum = np.max(np.abs(le)) < 1e-2
    saturates_dim = abs(dky - d) < 1e-6
    return near_zero or (tiny_spectrum and saturates_dim)

for d in D_SWEEP:
    print(f"\n{'='*55}")
    print(f"d = {d}")
    print(f"{'='*55}")
    t_start = time.time()

    # --- POD AE ---
    pod = fit_pod_autoencoder(traj_train, d)
    encode_fn = lambda u: pod_encode(pod, u)
    decode_fn = lambda h: pod_decode(pod, h)

    ae_diag = reconstruction_diagnostics(pod, traj_test[:1000], encode_fn, decode_fn)
    print(f"  POD recon MSE={ae_diag['mse']:.5f}, rel_err={ae_diag['rel_error']:.4f}, "
          f"energy_ratio={ae_diag['energy_ratio']:.4f}")

    # --- Latent data ---
    lat_data = prepare_latent_data_pod(traj_train, pod, solver, subsample=2)

    # --- Train latent ODE ---
    key_ode = jax.random.PRNGKey(d * 100)
    ode_params = init_latent_ode(key_ode, d=d, hidden=128, n_layers=3)
    ode_params, ode_loss = train_latent_ode(
        ode_params, lat_data["h"], lat_data["dhdt"],
        n_epochs=500, batch_size=256, key=key_ode
    )
    print(f"  ODE final loss: {ode_loss[-1]:.5f}")

    # --- Rollout ---
    u0 = traj_test[0].astype(np.float64)
    try:
        traj_node, _ = rollout_latent_node(
            pod, ode_params, u0, n_steps=2000, dt=0.25,
            encode_fn=encode_fn, decode_fn=decode_fn
        )
        energy = float(np.mean(np.sum(traj_node**2, axis=1)))
        stable = not (np.any(np.isnan(traj_node)) or energy > 1e6)
        print(f"  Rollout: energy={energy:.2f}, stable={stable}")
    except Exception as e:
        print(f"  Rollout failed: {e}")
        traj_node = None
        energy = float('nan')
        stable = False

    # --- Lyapunov ---
    h0 = np.array(encode_fn(jnp.array(traj_test[200], dtype=jnp.float64)))
    try:
        le_lat = compute_latent_lyapunov(
            ode_params, h0, n_steps=800, dt=0.25, n_warmup=200
        )
        if latent_lyap_needs_refinement(le_lat, d):
            print("  Latent spectrum is near-zero or saturates d -- escalating to 2500 steps...")
            le_lat = compute_latent_lyapunov(
                ode_params, h0, n_steps=2500, dt=0.25, n_warmup=400
            )
        lyap_warning = latent_lyap_needs_refinement(le_lat, d)
        dky_lat = kaplan_yorke(le_lat)
        h_ks_lat = float(np.sum(le_lat[le_lat > 0]))
        n_pos_lat = int(np.sum(le_lat > 0))
        if lyap_warning:
            print("  WARNING: latent spectrum remains near-neutral; treat D_KY with caution.")
        print(f"  Lyapunov: L1={le_lat[0]:+.4f}, n_pos={n_pos_lat}, "
              f"D_KY={dky_lat:.2f}, h_KS={h_ks_lat:.4f}")
    except Exception as e:
        print(f"  Lyapunov failed: {e}")
        le_lat = None
        dky_lat = n_pos_lat = h_ks_lat = float('nan')
        lyap_warning = True

    runtime = time.time() - t_start
    print(f"  Runtime: {runtime:.1f}s")

    results[d] = {
        "d": d,
        "ae_diagnostics": ae_diag,
        "ode_loss_history": ode_loss,
        "final_ode_loss": ode_loss[-1],
        "energy": energy,
        "stable": stable,
        "lyapunov": le_lat,
        "D_KY": dky_lat,
        "h_KS": h_ks_lat,
        "n_pos": n_pos_lat,
        "lyap_warning": lyap_warning,
        "traj": traj_node[:500] if traj_node is not None else None,
        "runtime": runtime,
    }
    log_event(
        "T5",
        "latent_dim_complete",
        config={
            "latent_dim": d,
            "ode_epochs": 500,
            "batch_size": 256,
            "subsample": 2,
        },
        metrics={
            "ae_mse": ae_diag["mse"],
            "ae_rel_error": ae_diag["rel_error"],
            "ae_energy_ratio": ae_diag["energy_ratio"],
            "final_ode_loss": ode_loss[-1],
            "energy": energy,
            "stable": stable,
            "D_KY": dky_lat,
            "h_KS": h_ks_lat,
            "n_pos": n_pos_lat,
            "lyap_warning": lyap_warning,
            "L1": float(le_lat[0]) if le_lat is not None else None,
        },
        timings={
            "total_wall_s": runtime,
        },
    )

with open("data/latent_dim_sweep.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: data/latent_dim_sweep.pkl")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"{'d':>4} {'AE_MSE':>8} {'ODE_loss':>9} {'Energy':>8} {'Stable':>7} "
      f"{'D_KY':>7} {'h_KS':>8} {'n_pos':>6} {'Runtime':>8}")
print("-"*80)
print(f"{'True':>4} {'':>8} {'':>9} {true_energy:>8.1f} {'Yes':>7} "
      f"{dky_true:>7.2f} {h_ks_true:>8.4f} {int(np.sum(le_true>0)):>6d}")
for d in D_SWEEP:
    r = results[d]
    stable_s = "Yes" if r["stable"] else "No"
    print(f"{d:>4d} {r['ae_diagnostics']['mse']:>8.5f} {r['final_ode_loss']:>9.5f} "
          f"{r['energy']:>8.1f} {stable_s:>7} "
          f"{r['D_KY']:>7.2f} {r['h_KS']:>8.4f} {r['n_pos']:>6} "
          f"{r['runtime']:>7.1f}s")
print("="*80)

# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
d_vals = D_SWEEP

# AE reconstruction MSE vs d
ax = axes[0, 0]
ae_mses = [results[d]["ae_diagnostics"]["mse"] for d in d_vals]
ax.semilogy(d_vals, ae_mses, 'o-', color='C0', lw=2, ms=8)
ax.set_xlabel("Latent dimension d")
ax.set_ylabel("AE Reconstruction MSE")
ax.set_title("POD Reconstruction Error vs d")
ax.grid(True, alpha=0.3)
ax.set_xticks(d_vals)

# Energy vs d
ax = axes[0, 1]
energies = [results[d]["energy"] for d in d_vals]
ax.plot(d_vals, energies, 'o-', color='C1', lw=2, ms=8)
ax.axhline(true_energy, ls='--', color='k', lw=2, label=f'True ({true_energy:.1f})')
ax.set_xlabel("Latent dimension d")
ax.set_ylabel("Rollout Mean Energy ||u||^2")
ax.set_title("Rollout Energy vs d")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(d_vals)

# D_KY vs d
ax = axes[0, 2]
dky_vals = [results[d]["D_KY"] for d in d_vals]
ax.plot(d_vals, dky_vals, 'o-', color='C2', lw=2, ms=8, label='Latent NODE')
ax.axhline(dky_true, ls='--', color='k', lw=2, label=f'True ({dky_true:.2f})')
ax.set_xlabel("Latent dimension d")
ax.set_ylabel("Kaplan-Yorke Dimension D_KY")
ax.set_title("Dynamical Fidelity (D_KY) vs d")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(d_vals)

# h_KS vs d
ax = axes[1, 0]
h_ks_vals = [results[d]["h_KS"] for d in d_vals]
ax.plot(d_vals, h_ks_vals, 'o-', color='C3', lw=2, ms=8)
ax.axhline(h_ks_true, ls='--', color='k', lw=2, label=f'True ({h_ks_true:.4f})')
ax.set_xlabel("Latent dimension d")
ax.set_ylabel("KS Entropy h_KS")
ax.set_title("KS Entropy vs d")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(d_vals)

# Lyapunov spectra comparison
ax = axes[1, 1]
ax.axhline(0, color='k', lw=0.8, ls='--')
colors_d = plt.cm.viridis(np.linspace(0, 1, len(d_vals)))
for i, d in enumerate(d_vals):
    le = results[d]["lyapunov"]
    if le is not None:
        ax.plot(np.arange(1, d+1), le, 'o-', color=colors_d[i],
                label=f'd={d}', ms=4, lw=1.5)
ax.set_xlabel("Index i")
ax.set_ylabel("Lyapunov exponent")
ax.set_title("Latent Lyapunov Spectra")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ODE loss vs d
ax = axes[1, 2]
ode_losses = [results[d]["final_ode_loss"] for d in d_vals]
ax.semilogy(d_vals, ode_losses, 'o-', color='C4', lw=2, ms=8)
ax.set_xlabel("Latent dimension d")
ax.set_ylabel("Final Latent ODE Loss")
ax.set_title("ODE Training Loss vs d")
ax.grid(True, alpha=0.3)
ax.set_xticks(d_vals)

plt.suptitle("T5: Latent Dimension Sweep (POD + Latent NODE)", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT5_latent_dim_sweep.png", dpi=120)
plt.close()
print("  Saved: figures/figT5_latent_dim_sweep.png")

print("\nT5 complete.")
log_event(
    "T5",
    "script_complete",
    config={"latent_dims": D_SWEEP},
    metrics={"n_runs": len(D_SWEEP)},
)
