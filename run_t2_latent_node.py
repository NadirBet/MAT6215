"""
run_t2_latent_node.py - Task T2: Linot-style Reduced-Manifold NODE
==================================================================
Trains and evaluates:
  1. Nonlinear AE + Latent ODE (d=8, main mode)
  2. POD AE + Latent ODE (d=8, linear comparison)
  3. Nonlinear AE + Discrete Map (d=8, comparison T7)

Saves results to data/latent_node_results.pkl
Saves figures to figures/
"""

import sys
sys.path = [p for p in sys.path if 'MAT6215' not in p]
sys.path.insert(0, '.')

import numpy as np
import jax
import jax.numpy as jnp
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ks_solver import KSSolver
from latent_node import (
    run_latent_node_pipeline, kaplan_yorke,
    rollout_latent_node, rollout_discrete_map,
    encode, decode, pod_encode, pod_decode
)

jax.config.update("jax_enable_x64", True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading trajectory data...")
traj_train = np.load("data/traj_train.npy")  # (20000, 64)
traj_test  = np.load("data/traj_analysis.npy")  # (8000, 64)
le_true    = np.load("data/lyapunov_exponents_full.npy")

solver = KSSolver(L=22.0, N=64, dt=0.25)

true_energy = float(np.mean(np.sum(traj_test**2, axis=1)))
print(f"True KSE energy: {true_energy:.3f}")

# Normalize training data: zero mean, unit std per component
# This is critical for nonlinear AE — raw KSE data has RMS ~1.18 which saturates tanh
u_mean_global = traj_train.mean(axis=0)         # (N,)
u_std_global  = traj_train.std(axis=0) + 1e-8  # (N,)
traj_train_norm = (traj_train - u_mean_global) / u_std_global
traj_test_norm  = (traj_test  - u_mean_global) / u_std_global
print(f"Data normalized: mean~0, std~1 per component")
print(f"  train RMS after norm: {float(np.sqrt(np.mean(traj_train_norm**2))):.4f}")
print(f"True KSE: L1={le_true[0]:.4f}, n_pos={int(np.sum(le_true>0))}, D_KY={kaplan_yorke(le_true):.2f}")

# Wrapper: solver in normalized coordinates
# Latent data preparation needs du/dt in the same coordinate system as u
# For normalized u_norm = (u - mean)/std, du_norm/dt = (du/dt) / std
class NormSolver:
    """Thin wrapper that returns RHS in normalized coordinates."""
    def __init__(self, solver, u_mean, u_std):
        self.solver = solver
        self.u_mean = u_mean
        self.u_std = u_std

    def rhs(self, u_hat_norm):
        # u_hat_norm is FFT of normalized u; recover physical RHS
        # u_phys = u_norm * std + mean; du_phys/dt = rhs_phys(u_phys)
        # du_norm/dt = (du_phys/dt) / std
        # For RHS in Fourier: rhs_hat_norm_k = rhs_hat_phys_k / std_k
        # But std is per-component in physical space, not diagonal in Fourier space
        # Simpler: work entirely in physical space
        raise NotImplementedError("Use rhs_physical_norm instead")

    def rhs_physical_norm(self, u_norm):
        """Physical-space RHS in normalized coordinates."""
        u_phys = u_norm * self.u_std + self.u_mean
        u_hat = jnp.fft.fft(jnp.array(u_phys))
        rhs_hat = self.solver.rhs(u_hat)
        rhs_phys = jnp.fft.ifft(rhs_hat).real
        return np.array(rhs_phys / self.u_std)  # normalize the derivative


norm_solver = NormSolver(solver, u_mean_global, u_std_global)

# ── Run 1: Nonlinear AE + Latent NODE, d=8 ────────────────────────────────────
print("\n" + "="*60)
print("RUN 1: Nonlinear AE + Latent NODE (d=8, normalized data)")
print("="*60)
key1 = jax.random.PRNGKey(42)
results_nl = run_latent_node_pipeline(
    traj_train_norm, traj_test_norm, norm_solver,
    d=8,
    ae_epochs=500,
    ode_epochs=500,
    ae_hidden=256, ae_layers=3,
    ode_hidden=128, ode_layers=3,
    batch_size=256, subsample=2,
    mode='nonlinear',
    key=key1
)
results_nl["label"] = "Nonlinear-AE+NODE d=8"
results_nl["u_mean"] = u_mean_global
results_nl["u_std"] = u_std_global

# ── Run 2: POD AE + Latent NODE, d=8 ─────────────────────────────────────────
print("\n" + "="*60)
print("RUN 2: POD AE + Latent NODE (d=8, normalized data)")
print("="*60)
key2 = jax.random.PRNGKey(43)
results_pod = run_latent_node_pipeline(
    traj_train_norm, traj_test_norm, norm_solver,
    d=8,
    ae_epochs=0,  # POD requires no training
    ode_epochs=500,
    ode_hidden=128, ode_layers=3,
    batch_size=256, subsample=2,
    mode='pod',
    key=key2
)
results_pod["label"] = "POD-AE+NODE d=8"
results_pod["u_mean"] = u_mean_global
results_pod["u_std"] = u_std_global

# ── Save all results ───────────────────────────────────────────────────────────
all_results = {
    "nonlinear": results_nl,
    "pod": results_pod,
    "true_energy": true_energy,
    "le_true": le_true,
}

with open("data/latent_node_results.pkl", "wb") as f:
    pickle.dump(all_results, f)
print("\nSaved: data/latent_node_results.pkl")

# ── Print summary table ────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'System':<28} {'L1':>7} {'n_pos':>6} {'D_KY':>7} {'h_KS':>7} {'Energy':>8}")
print("-"*70)

# True KSE
dky_true = kaplan_yorke(le_true)
h_ks_true = float(np.sum(le_true[le_true > 0]))
print(f"{'True KSE':<28} {le_true[0]:>+7.4f} {int(np.sum(le_true>0)):>6d} {dky_true:>7.2f} {h_ks_true:>7.4f} {true_energy:>8.3f}")

for key_r, res in [("nonlinear", results_nl), ("pod", results_pod)]:
    label = res.get("label", key_r)
    le_lat = res.get("lyapunov_latent")
    energy = res.get("rollout_node_energy", float('nan'))
    ae_mse = res["ae_diagnostics"]["mse"]
    if le_lat is not None:
        print(f"{'  '+label:<28} {le_lat[0]:>+7.4f} {res['n_pos_latent']:>6d} {res['dky_latent']:>7.2f} {res['h_ks_latent']:>7.4f} {energy:>8.3f}  AE_MSE={ae_mse:.5f}")
    else:
        print(f"{'  '+label:<28} {'N/A':>7} {'N/A':>6} {'N/A':>7} {'N/A':>7} {energy:>8.3f}  AE_MSE={ae_mse:.5f}")

print("="*70)

# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Fig A: AE loss curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
if results_nl.get("ae_loss_history"):
    ax.semilogy(results_nl["ae_loss_history"], label="Nonlinear AE")
ax.set_xlabel("Epoch")
ax.set_ylabel("AE Reconstruction Loss")
ax.set_title("Autoencoder Training")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(results_nl["ode_loss_history"], label="Nonlinear AE + NODE", color="C0")
ax.semilogy(results_pod["ode_loss_history"], label="POD + NODE", color="C1")
ax.set_xlabel("Epoch")
ax.set_ylabel("Latent ODE Loss")
ax.set_title("Latent ODE Training")
ax.legend()
ax.grid(True, alpha=0.3)
plt.suptitle("T2: Latent NODE Training Curves", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT2_latent_training.png", dpi=120)
plt.close()
print("  Saved: figures/figT2_latent_training.png")

# Fig B: Space-time comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
x = np.linspace(0, 22, 64)
t_plot = np.arange(500) * 0.25

for ax, traj, title in zip(
    axes,
    [traj_test[:500], results_nl.get("traj_latent_node", np.zeros((500, 64)))[:500],
     results_pod.get("traj_latent_node", np.zeros((500, 64)))[:500]],
    ["True KSE", "Nonlinear AE + NODE", "POD + NODE"]
):
    if traj is not None and traj.shape[0] > 0:
        vmax = max(np.percentile(np.abs(traj[:500]), 98), 0.1)
        ax.pcolormesh(x, t_plot, traj[:500], cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, shading='auto', rasterized=True)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
plt.suptitle("T2: Space-Time Comparison (500 steps)", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT2_spacetime.png", dpi=120)
plt.close()
print("  Saved: figures/figT2_spacetime.png")

# Fig C: Power spectrum comparison
fig, ax = plt.subplots(figsize=(7, 5))
q = np.fft.rfftfreq(64, d=22.0/64) * (2 * np.pi)

def power_spectrum(traj):
    return np.mean(np.abs(np.fft.rfft(traj, axis=1))**2, axis=0)

ax.semilogy(q[1:], power_spectrum(traj_test[:2000])[1:], 'k-', lw=2, label="True KSE")
if results_nl.get("traj_latent_node") is not None:
    ax.semilogy(q[1:], power_spectrum(results_nl["traj_latent_node"][:2000])[1:],
                'b--', lw=1.5, label="Nonlinear AE + NODE")
if results_pod.get("traj_latent_node") is not None:
    ax.semilogy(q[1:], power_spectrum(results_pod["traj_latent_node"][:2000])[1:],
                'r-.', lw=1.5, label="POD + NODE")
ax.set_xlabel("Wavenumber q")
ax.set_ylabel("E(q)")
ax.set_title("T2: Power Spectrum")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/figT2_power_spectrum.png", dpi=120)
plt.close()
print("  Saved: figures/figT2_power_spectrum.png")

# Fig D: Latent Lyapunov spectrum
fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(0, color='k', lw=0.8, ls='--')
for res, color, label in [
    (results_nl, 'C0', 'Nonlinear AE + NODE'),
    (results_pod, 'C1', 'POD + NODE'),
]:
    le = res.get("lyapunov_latent")
    if le is not None:
        ax.plot(np.arange(1, len(le)+1), le, 'o-', color=color, label=label, ms=5)

ax.set_xlabel("Index i")
ax.set_ylabel("Lyapunov exponent")
ax.set_title("T2: Latent Lyapunov Spectrum")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/figT2_lyapunov.png", dpi=120)
plt.close()
print("  Saved: figures/figT2_lyapunov.png")

print("\nT2 complete.")
