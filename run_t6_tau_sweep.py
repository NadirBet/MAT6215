"""
run_t6_tau_sweep.py - Task T6: Data-Spacing (Tau) Sweep
=========================================================
Trains latent ODE and discrete map at different temporal spacings tau.
For each tau: short-time prediction error, autocorr, joint PDF, Lyapunov metrics.

Tau values (in solver steps, dt=0.25):
  strides = [1, 2, 4, 8, 16, 32]  -> tau = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
  In Lyapunov times (tau_L=22):     -> [0.011, 0.023, 0.045, 0.091, 0.18, 0.36]

For each tau:
  - Discrete map G: h_{n+1} = G(h_n)  trained on subsampled pairs
  - Latent NODE dh/dt = g(h)           trained on instantaneous (h, dh/dt)
    with the same subsampled data density

Fixed: POD, d=10, hidden=128, n_layers=3

Outputs:
  data/tau_sweep_results.pkl
  figures/figT6_tau_sweep.png
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
from experiment_log import log_event
from latent_node import (
    fit_pod_autoencoder, pod_encode, pod_decode,
    init_latent_ode, train_latent_ode,
    prepare_latent_data_pod, prepare_discrete_latent_data,
    rollout_latent_node, rollout_discrete_map,
    init_discrete_map, train_discrete_map,
    compute_latent_lyapunov, compute_discrete_map_lyapunov,
    reconstruction_diagnostics,
    kaplan_yorke,
)

jax.config.update("jax_enable_x64", True)

# ── Constants ──────────────────────────────────────────────────────────────────
LYAP_TIME = 22.0   # Lyapunov time tau_L for KSE L=22
DT = 0.25
D_LATENT = 10      # fixed latent dim (from T5 tradeoff)
STRIDES = [1, 2, 4, 8, 16, 32]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
traj_train = np.load("data/traj_train.npy")
traj_test  = np.load("data/traj_analysis.npy")
le_true    = np.load("data/lyapunov_exponents_full.npy")
solver = KSSolver(L=22.0, N=64, dt=DT)

true_energy = float(np.mean(np.sum(traj_test**2, axis=1)))
dky_true = kaplan_yorke(le_true)
h_ks_true = float(np.sum(le_true[le_true > 0]))
print(f"True KSE: energy={true_energy:.2f}, D_KY={dky_true:.2f}, h_KS={h_ks_true:.4f}")

# ── Shared POD basis (fit once on all data) ─────────────────────────────────
print(f"\nFitting POD (d={D_LATENT})...")
pod = fit_pod_autoencoder(traj_train, D_LATENT)
encode_fn = lambda u: pod_encode(pod, u)
decode_fn = lambda h: pod_decode(pod, h)

ae_diag = reconstruction_diagnostics(pod, traj_test[:1000], encode_fn, decode_fn)
print(f"POD recon MSE={ae_diag['mse']:.5f}, energy_ratio={ae_diag['energy_ratio']:.4f}")


# ── Helper: Lyapunov exponents in latent space for map ─────────────────────
def lyap_needs_refinement(le, dim):
    """Flag spectra that are too close to the numerical noise floor."""
    if le is None:
        return False
    le = np.asarray(le)
    dky = kaplan_yorke(le)
    near_zero = np.any(np.abs(le) < 0.02)
    tiny_spectrum = np.max(np.abs(le)) < 1e-2
    saturates_dim = abs(dky - dim) < 1e-6
    return near_zero or (tiny_spectrum and saturates_dim)


def rollout_traj_latent(pod, ode_params, u0, n_steps):
    """Rollout latent NODE and return physical trajectory."""
    try:
        traj, _ = rollout_latent_node(
            pod, ode_params, u0, n_steps, dt=DT,
            encode_fn=encode_fn, decode_fn=decode_fn)
        return traj
    except Exception:
        return None


def rollout_traj_map(pod, map_params, u0, n_steps):
    """Rollout discrete map and return physical trajectory (one step = 1 stride)."""
    try:
        traj, _ = rollout_discrete_map(
            pod, map_params, u0, n_steps,
            encode_fn=encode_fn, decode_fn=decode_fn)
        return traj
    except Exception:
        return None


# ── Precompute dense latent data once ─────────────────────────────────────────
print("\nPreparing dense latent data (stride=1)...")
dense_lat  = prepare_latent_data_pod(traj_train, pod, solver, subsample=1)
print(f"  {len(dense_lat['h'])} latent points")

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = {}

for stride in STRIDES:
    tau_eff = stride * DT
    tau_lyap = tau_eff / LYAP_TIME
    print(f"\n{'='*60}")
    print(f"stride={stride}  tau={tau_eff:.2f}  tau/tau_L={tau_lyap:.3f}")
    print(f"{'='*60}")
    t0 = time.time()

    # Subsample data for this stride
    h_s    = dense_lat["h"][::stride]
    dhdt_s = dense_lat["dhdt"][::stride]
    print(f"  Training points (ODE): {len(h_s)}")

    # ── Latent ODE ─────────────────────────────────────────────────────────
    key_ode = jax.random.PRNGKey(stride * 7)
    ode_params = init_latent_ode(key_ode, d=D_LATENT, hidden=128, n_layers=3)
    ode_params, ode_loss = train_latent_ode(
        ode_params, h_s, dhdt_s,
        n_epochs=400, batch_size=256, key=key_ode)
    print(f"  ODE final loss: {ode_loss[-1]:.6f}")

    # ODE Rollout
    u0 = traj_test[0].astype(np.float64)
    traj_ode = rollout_traj_latent(pod, ode_params, u0, n_steps=2000)
    ode_energy = float(np.mean(np.sum(traj_ode**2, axis=1))) if traj_ode is not None else float('nan')
    ode_stable = (traj_ode is not None and not np.isnan(ode_energy) and ode_energy < 1e6)
    print(f"  ODE rollout: energy={ode_energy:.2f}, stable={ode_stable}")

    # ODE Lyapunov (in latent space)
    h0_lyap = np.array(encode_fn(jnp.array(traj_test[200], dtype=jnp.float64)))
    try:
        le_ode = compute_latent_lyapunov(
            ode_params, h0_lyap, n_steps=800, dt=DT, n_warmup=200
        )
        if lyap_needs_refinement(le_ode, D_LATENT):
            print("  ODE spectrum is near-zero or saturates d -- escalating to 2500 steps...")
            le_ode = compute_latent_lyapunov(
                ode_params, h0_lyap, n_steps=2500, dt=DT, n_warmup=400
            )
        ode_lyap_warning = lyap_needs_refinement(le_ode, D_LATENT)
        dky_ode = kaplan_yorke(le_ode)
        h_ks_ode = float(np.sum(le_ode[le_ode > 0]))
        if ode_lyap_warning:
            print("  WARNING: ODE spectrum remains near-neutral; treat D_KY with caution.")
    except Exception as e:
        print(f"  ODE Lyapunov failed: {e}")
        le_ode = None; dky_ode = h_ks_ode = float('nan')
        ode_lyap_warning = True

    # ── Discrete Map ───────────────────────────────────────────────────────
    disc_data = prepare_discrete_latent_data(traj_train, encode_fn, subsample=stride)
    print(f"  Training points (map): {len(disc_data['h_n'])}")

    key_map = jax.random.PRNGKey(stride * 13)
    map_params = init_discrete_map(key_map, d=D_LATENT, hidden=128, n_layers=3)
    map_params, map_loss = train_discrete_map(
        map_params, disc_data["h_n"], disc_data["h_n1"],
        n_epochs=400, batch_size=256, key=key_map)
    print(f"  Map final loss: {map_loss[-1]:.6f}")

    # Map Rollout (n_steps in units of tau_eff)
    traj_map = rollout_traj_map(pod, map_params, u0, n_steps=2000 // max(stride, 1))
    map_energy = float(np.mean(np.sum(traj_map**2, axis=1))) if traj_map is not None else float('nan')
    map_stable = (traj_map is not None and not np.isnan(map_energy) and map_energy < 1e6)
    print(f"  Map rollout: energy={map_energy:.2f}, stable={map_stable}")

    # Map Lyapunov (per-time-unit exponents; tau_eff handled inside the utility)
    try:
        le_map = compute_discrete_map_lyapunov(
            map_params, h0_lyap, n_steps=800, tau=tau_eff, n_warmup=200
        )
        if lyap_needs_refinement(le_map, D_LATENT):
            print("  Map spectrum is near-zero or saturates d -- escalating to 2500 steps...")
            le_map = compute_discrete_map_lyapunov(
                map_params, h0_lyap, n_steps=2500, tau=tau_eff, n_warmup=400
            )
        map_lyap_warning = lyap_needs_refinement(le_map, D_LATENT)
        dky_map = kaplan_yorke(le_map)
        h_ks_map = float(np.sum(le_map[le_map > 0]))
        if map_lyap_warning:
            print("  WARNING: map spectrum remains near-neutral; treat D_KY with caution.")
    except Exception as e:
        print(f"  Map Lyapunov failed: {e}")
        le_map = None; dky_map = h_ks_map = float('nan')
        map_lyap_warning = True

    runtime = time.time() - t0
    print(f"  Runtime: {runtime:.1f}s")

    results[stride] = {
        "stride": stride,
        "tau": tau_eff,
        "tau_lyap": tau_lyap,
        # ODE
        "ode_loss": ode_loss[-1],
        "ode_energy": ode_energy,
        "ode_stable": ode_stable,
        "ode_lyapunov": le_ode,
        "ode_D_KY": dky_ode,
        "ode_h_KS": h_ks_ode,
        "ode_n_pos": int(np.sum(le_ode > 0)) if le_ode is not None else 0,
        "ode_L1": float(le_ode[0]) if le_ode is not None else float('nan'),
        "ode_lyap_warning": ode_lyap_warning,
        # Map
        "map_loss": map_loss[-1],
        "map_energy": map_energy,
        "map_stable": map_stable,
        "map_lyapunov": le_map,
        "map_D_KY": dky_map,
        "map_h_KS": h_ks_map,
        "map_n_pos": int(np.sum(le_map > 0)) if le_map is not None else 0,
        "map_L1": float(le_map[0]) if le_map is not None else float('nan'),
        "map_lyap_warning": map_lyap_warning,
        "runtime": runtime,
    }
    log_event("T6", "tau_step_complete",
              config={"stride": stride, "tau": tau_eff, "d": D_LATENT},
              metrics={
                  "ode_loss": ode_loss[-1], "ode_D_KY": dky_ode, "ode_h_KS": h_ks_ode,
                  "ode_lyap_warning": ode_lyap_warning,
                  "map_loss": map_loss[-1], "map_D_KY": dky_map, "map_h_KS": h_ks_map,
                  "map_lyap_warning": map_lyap_warning,
              })

with open("data/tau_sweep_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: data/tau_sweep_results.pkl")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'stride':>7} {'tau':>6} {'tau/tL':>7} | "
      f"{'ODE_L1':>8} {'ODE_DKY':>8} {'ODE_stab':>9} | "
      f"{'MAP_L1':>8} {'MAP_DKY':>8} {'MAP_stab':>9}")
print("-"*90)
print(f"{'TRUE':>7} {'':>6} {'':>7} | {le_true[0]:>+8.4f} {dky_true:>8.2f} {'Yes':>9} | {'':>8} {'':>8}")
for s in STRIDES:
    r = results[s]
    print(f"{s:>7d} {r['tau']:>6.2f} {r['tau_lyap']:>7.3f} | "
          f"{r['ode_L1']:>+8.4f} {r['ode_D_KY']:>8.2f} {str(r['ode_stable']):>9} | "
          f"{r['map_L1']:>+8.4f} {r['map_D_KY']:>8.2f} {str(r['map_stable']):>9}")
print("="*90)

# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
taus = [results[s]["tau"] for s in STRIDES]
tau_lyaps = [results[s]["tau_lyap"] for s in STRIDES]

def safe(r, k): return r[k] if not (isinstance(r[k], float) and np.isnan(r[k])) else None

# D_KY vs tau
ax = axes[0, 0]
ode_dky = [results[s]["ode_D_KY"] for s in STRIDES]
map_dky = [results[s]["map_D_KY"] for s in STRIDES]
ax.semilogx(taus, ode_dky, 'o-', color='C0', lw=2, ms=7, label='Latent ODE')
ax.semilogx(taus, map_dky, 's--', color='C1', lw=2, ms=7, label='Discrete Map')
ax.axhline(dky_true, ls='--', color='k', lw=1.5, label=f'True ({dky_true:.2f})')
ax.set_xlabel("tau (time units)")
ax.set_ylabel("D_KY")
ax.set_title("Kaplan-Yorke Dim vs Tau")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# h_KS vs tau
ax = axes[0, 1]
ax.semilogx(taus, [results[s]["ode_h_KS"] for s in STRIDES], 'o-', color='C0', lw=2, ms=7, label='Latent ODE')
ax.semilogx(taus, [results[s]["map_h_KS"] for s in STRIDES], 's--', color='C1', lw=2, ms=7, label='Discrete Map')
ax.axhline(h_ks_true, ls='--', color='k', lw=1.5, label=f'True ({h_ks_true:.4f})')
ax.set_xlabel("tau (time units)")
ax.set_ylabel("h_KS (KS entropy)")
ax.set_title("KS Entropy vs Tau")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Energy vs tau
ax = axes[0, 2]
ax.semilogx(taus, [results[s]["ode_energy"] for s in STRIDES], 'o-', color='C0', lw=2, ms=7, label='Latent ODE')
ax.semilogx(taus, [results[s]["map_energy"] for s in STRIDES], 's--', color='C1', lw=2, ms=7, label='Discrete Map')
ax.axhline(true_energy, ls='--', color='k', lw=1.5, label=f'True ({true_energy:.1f})')
ax.set_xlabel("tau (time units)")
ax.set_ylabel("Rollout Energy")
ax.set_title("Rollout Energy vs Tau")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# L1 vs tau
ax = axes[1, 0]
ax.semilogx(taus, [results[s]["ode_L1"] for s in STRIDES], 'o-', color='C0', lw=2, ms=7, label='Latent ODE')
ax.semilogx(taus, [results[s]["map_L1"] for s in STRIDES], 's--', color='C1', lw=2, ms=7, label='Discrete Map')
ax.axhline(le_true[0], ls='--', color='k', lw=1.5, label=f'True ({le_true[0]:.4f})')
ax.set_xlabel("tau (time units)")
ax.set_ylabel("L1 (leading Lyapunov)")
ax.set_title("Leading Lyapunov vs Tau")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Loss vs tau
ax = axes[1, 1]
ax.semilogx(taus, [results[s]["ode_loss"] for s in STRIDES], 'o-', color='C0', lw=2, ms=7, label='ODE loss')
ax.semilogx(taus, [results[s]["map_loss"] for s in STRIDES], 's--', color='C1', lw=2, ms=7, label='Map loss')
ax.set_xlabel("tau (time units)")
ax.set_ylabel("Final Training Loss")
ax.set_title("Training Loss vs Tau")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# n_pos vs tau
ax = axes[1, 2]
ax.semilogx(taus, [results[s]["ode_n_pos"] for s in STRIDES], 'o-', color='C0', lw=2, ms=7, label='Latent ODE')
ax.semilogx(taus, [results[s]["map_n_pos"] for s in STRIDES], 's--', color='C1', lw=2, ms=7, label='Discrete Map')
ax.axhline(int(np.sum(le_true > 0)), ls='--', color='k', lw=1.5)
ax.set_xlabel("tau (time units)")
ax.set_ylabel("n_pos (# positive LEs)")
ax.set_title("Positive Exponents vs Tau")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(f"T6: Tau Sweep - Latent ODE vs Discrete Map (d={D_LATENT})", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT6_tau_sweep.png", dpi=120)
plt.close()
print("  Saved: figures/figT6_tau_sweep.png")

log_event("T6", "script_complete",
          config={"strides": STRIDES, "d": D_LATENT},
          metrics={"n_runs": len(STRIDES)})
print("\nT6 complete.")
