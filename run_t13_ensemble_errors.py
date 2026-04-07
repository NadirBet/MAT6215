"""
run_t13_ensemble_errors.py - Task T13: Ensemble Short-Time Error Curves
========================================================================
Computes ensemble-averaged forecast error curves for all surrogates.

Protocol:
  - Load 100 initial conditions from ensemble_ic.npy (attractor samples)
  - Roll out true KSE from each IC for T_max=400 steps (100 time units)
  - Roll out each surrogate from the same IC
  - Compute RMSE(t) = sqrt(mean_over_ICs ||u_surrogate(t) - u_true(t)||^2 / N)
  - Normalize by attractor RMS to get dimensionless error
  - Find "predictability horizon" = time where RMSE crosses 0.5 of attractor RMS

Surrogates tested:
  1. NODE-Std-MSE (full physical space)
  2. NODE-Stab-MSE negdef (stable, from T4)
  3. NODE-Stab-JAC (full space)
  4. Latent NODE (POD d=10)
  5. SINDy PI (physics-informed)

Outputs:
  data/ensemble_error_results.pkl
  figures/figT13_ensemble_errors.png
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
from neural_ode import stabilized_node_rhs, standard_node_rhs
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

# ── Config ─────────────────────────────────────────────────────────────────────
T_MAX_STEPS = 400    # 100 time units
DT = 0.25
LYAP_TIME = 22.0     # τ_L for KSE L=22

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
solver = KSSolver(L=22.0, N=64, dt=DT)

# Ensemble ICs
try:
    ensemble_ic = np.load("data/ensemble_ic.npy")
    print(f"Loaded {len(ensemble_ic)} ensemble ICs")
except FileNotFoundError:
    # Generate from attractor samples
    print("ensemble_ic.npy not found, sampling from traj_analysis...")
    traj_analysis = np.load("data/traj_analysis.npy")
    T = len(traj_analysis)
    rng = np.random.default_rng(99)
    idx = rng.choice(T, size=100, replace=False)
    ensemble_ic = traj_analysis[idx].astype(np.float64)
    np.save("data/ensemble_ic.npy", ensemble_ic)

N_IC = len(ensemble_ic)

# Compute attractor RMS for normalization
traj_analysis = np.load("data/traj_analysis.npy")
attractor_rms = float(np.sqrt(np.mean(traj_analysis**2)))
print(f"Attractor RMS: {attractor_rms:.4f}")

# ── True KSE rollout ──────────────────────────────────────────────────────────
print(f"\nComputing true KSE rollouts ({N_IC} ICs, {T_MAX_STEPS} steps)...")
true_trajs = np.zeros((N_IC, T_MAX_STEPS, 64))
t0 = time.time()
for ic_idx in range(N_IC):
    u = ensemble_ic[ic_idx].copy()
    u_hat = np.fft.fft(u)
    for t in range(T_MAX_STEPS):
        u_hat = solver.step(u_hat)
        true_trajs[ic_idx, t] = np.fft.ifft(u_hat).real
    if (ic_idx + 1) % 20 == 0:
        print(f"  IC {ic_idx+1}/{N_IC}")
print(f"  Done in {time.time()-t0:.1f}s")


# ── Generic RMSE curve function ────────────────────────────────────────────────
def compute_rmse_curve(rollout_fn, name):
    """
    Computes ensemble-averaged RMSE(t) for a surrogate.

    rollout_fn: callable(u0: np.ndarray) -> traj of shape (T_MAX_STEPS, 64) or None
    Returns: rmse_curve of shape (T_MAX_STEPS,), normalized by attractor_rms
    """
    errors_sq = np.zeros((N_IC, T_MAX_STEPS))
    n_valid = 0
    print(f"\n  Computing {name} ({N_IC} ICs)...")
    t0 = time.time()
    for ic_idx in range(N_IC):
        u0 = ensemble_ic[ic_idx]
        try:
            traj_sur = rollout_fn(u0)
            if traj_sur is None or np.any(np.isnan(traj_sur)) or np.any(np.isinf(traj_sur)):
                # Diverged — clip error to ceiling
                errors_sq[ic_idx] = attractor_rms ** 2
                continue
            # Align length
            T_use = min(len(traj_sur), T_MAX_STEPS)
            diff = traj_sur[:T_use] - true_trajs[ic_idx, :T_use]
            errors_sq[ic_idx, :T_use] = np.mean(diff**2, axis=1)
            # Fill remaining with ceiling
            errors_sq[ic_idx, T_use:] = attractor_rms ** 2
            n_valid += 1
        except Exception:
            errors_sq[ic_idx] = attractor_rms ** 2

    rmse = np.sqrt(np.mean(errors_sq, axis=0)) / attractor_rms
    print(f"  {name}: valid rollouts={n_valid}/{N_IC}, elapsed={time.time()-t0:.1f}s")
    return rmse


# ── RK4 integrator for NODE variants ──────────────────────────────────────────
def make_rk4_rollout(rhs_fn, params, n_steps=T_MAX_STEPS):
    step = jax.jit(lambda u: _rk4_step(rhs_fn, params, u, DT))
    def rollout(u0):
        traj = np.zeros((n_steps, 64))
        u = jnp.array(u0, dtype=jnp.float64)
        for t in range(n_steps):
            u = step(u)
            if jnp.any(jnp.isnan(u)) or jnp.linalg.norm(u) > 1e6:
                return None
            traj[t] = np.array(u)
        return traj
    return rollout

def _rk4_step(rhs_fn, params, u, dt):
    k1 = rhs_fn(params, u)
    k2 = rhs_fn(params, u + dt/2*k1)
    k3 = rhs_fn(params, u + dt/2*k2)
    k4 = rhs_fn(params, u + dt*k3)
    return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate 1: NODE-Std-MSE
# ─────────────────────────────────────────────────────────────────────────────
rmse_curves = {}

try:
    with open("data/node_standard_mse.pkl", "rb") as f:
        node_std = pickle.load(f)
    # params structure: {"nonlinear": mlp_params}  (from init_standard_node)
    params_std = node_std["params"]
    rollout_std = make_rk4_rollout(standard_node_rhs, params_std)
    rmse_curves["NODE-Std-MSE"] = compute_rmse_curve(rollout_std, "NODE-Std-MSE")
    log_event("T13", "surrogate_done", config={"surrogate": "NODE-Std-MSE"},
              metrics={"rmse_at_1_lyap": float(rmse_curves["NODE-Std-MSE"][
                  min(int(LYAP_TIME/DT), T_MAX_STEPS-1)])})
except Exception as e:
    print(f"  NODE-Std-MSE failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Surrogate 2: NODE-Stab-negdef (T4, stable)
# ─────────────────────────────────────────────────────────────────────────────
try:
    with open("data/constrained_a_results.pkl", "rb") as f:
        t4 = pickle.load(f)
    neg_params = t4["negdef"]["params"]
    EPS = 1e-3
    from neural_ode import mlp_forward as _mlp_fwd
    def rhs_neg(params, u):
        B = params["B"]
        A = -(B.T @ B + EPS * jnp.eye(B.shape[0]))
        return u @ A.T + _mlp_fwd(params["mlp"], u)
    rollout_neg = make_rk4_rollout(rhs_neg, neg_params)
    rmse_curves["NODE-negdef"] = compute_rmse_curve(rollout_neg, "NODE-negdef")
    log_event("T13", "surrogate_done", config={"surrogate": "NODE-negdef"},
              metrics={"rmse_at_1_lyap": float(rmse_curves["NODE-negdef"][
                  min(int(LYAP_TIME/DT), T_MAX_STEPS-1)])})
except Exception as e:
    print(f"  NODE-negdef failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Surrogate 3: NODE-Stab-JAC
# ─────────────────────────────────────────────────────────────────────────────
try:
    with open("data/node_stabilized_jac.pkl", "rb") as f:
        node_jac = pickle.load(f)
    # params structure: {"linear": ..., "nonlinear": ...}  (from init_stabilized_node)
    params_jac = node_jac["params"]
    rollout_jac = make_rk4_rollout(stabilized_node_rhs, params_jac)
    rmse_curves["NODE-Stab-JAC"] = compute_rmse_curve(rollout_jac, "NODE-Stab-JAC")
    log_event("T13", "surrogate_done", config={"surrogate": "NODE-Stab-JAC"},
              metrics={"rmse_at_1_lyap": float(rmse_curves["NODE-Stab-JAC"][
                  min(int(LYAP_TIME/DT), T_MAX_STEPS-1)])})
except Exception as e:
    print(f"  NODE-Stab-JAC failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Surrogate 4: Latent NODE (POD d=10)
# ─────────────────────────────────────────────────────────────────────────────
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
    ode_params, _ = train_latent_ode(
        ode_params, lat_data["h"], lat_data["dhdt"],
        n_epochs=300, batch_size=256, key=key_ode)
    print("  Latent NODE trained.")

    def rollout_lat(u0):
        try:
            traj, _ = rollout_latent_node(
                pod, ode_params, u0, n_steps=T_MAX_STEPS, dt=DT,
                encode_fn=encode_fn, decode_fn=decode_fn)
            return traj
        except Exception:
            return None

    rmse_curves["Latent-NODE-d10"] = compute_rmse_curve(rollout_lat, "Latent-NODE-d10")
    log_event("T13", "surrogate_done", config={"surrogate": "Latent-NODE-d10"},
              metrics={"rmse_at_1_lyap": float(rmse_curves["Latent-NODE-d10"][
                  min(int(LYAP_TIME/DT), T_MAX_STEPS-1)])})
except Exception as e:
    print(f"  Latent NODE failed: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# Surrogate 5: SINDy PI
# ─────────────────────────────────────────────────────────────────────────────
try:
    with open("data/sindy_model.pkl", "rb") as f:
        sindy_data = pickle.load(f)
    # sindy_data might be a SINDyModel or a dict
    from sindy import SINDyModel
    if isinstance(sindy_data, SINDyModel):
        sindy_model = sindy_data
    else:
        # Reconstruct
        sindy_model = sindy_data.get("model") or sindy_data

    def rollout_sindy(u0):
        try:
            traj = sindy_model.integrate(u0, n_steps=T_MAX_STEPS, dt=DT)
            if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                return None
            return traj
        except Exception:
            return None

    rmse_curves["SINDy-PI"] = compute_rmse_curve(rollout_sindy, "SINDy-PI")
    log_event("T13", "surrogate_done", config={"surrogate": "SINDy-PI"},
              metrics={"rmse_at_1_lyap": float(rmse_curves["SINDy-PI"][
                  min(int(LYAP_TIME/DT), T_MAX_STEPS-1)])})
except Exception as e:
    print(f"  SINDy failed: {e}")


# ── Predictability horizons ────────────────────────────────────────────────────
THRESHOLD = 0.5   # RMSE / attractor_rms
t_vec = np.arange(T_MAX_STEPS) * DT
t_lyap = t_vec / LYAP_TIME

def predictability_horizon(rmse, threshold=THRESHOLD):
    """Time steps where RMSE first exceeds threshold."""
    idx = np.where(rmse > threshold)[0]
    return t_lyap[idx[0]] if len(idx) > 0 else t_lyap[-1]

print("\n=== Predictability Horizons ===")
horizons = {}
for name, rmse in rmse_curves.items():
    h = predictability_horizon(rmse)
    horizons[name] = h
    print(f"  {name:<20}: {h:.3f} τ_L  "
          f"(RMSE at τ_L={rmse[min(int(LYAP_TIME/DT), T_MAX_STEPS-1)]:.3f})")

# ── Save ───────────────────────────────────────────────────────────────────────
results = {
    "rmse_curves": rmse_curves,
    "t_vec": t_vec,
    "t_lyap": t_lyap,
    "horizons": horizons,
    "attractor_rms": attractor_rms,
    "n_ic": N_IC,
    "T_max_steps": T_MAX_STEPS,
}
with open("data/ensemble_error_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: data/ensemble_error_results.pkl")


# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_map = {
    "NODE-Std-MSE": "C0", "NODE-negdef": "C1", "NODE-Stab-JAC": "C2",
    "Latent-NODE-d10": "C3", "SINDy-PI": "C4",
}

ax = axes[0]
for name, rmse in rmse_curves.items():
    ax.semilogy(t_lyap, rmse, lw=2, color=colors_map.get(name, "C5"), label=name)
ax.axhline(THRESHOLD, ls='--', color='k', lw=1, label=f'Threshold ({THRESHOLD})')
ax.axvline(1.0, ls=':', color='gray', lw=1, label='1 τ_L')
ax.set_xlabel("t / τ_L (Lyapunov times)")
ax.set_ylabel("Normalized RMSE")
ax.set_title(f"Ensemble Forecast Error ({N_IC} ICs)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
names_h = list(horizons.keys())
h_vals  = [horizons[n] for n in names_h]
ax.barh(range(len(names_h)), h_vals,
        color=[colors_map.get(n, "C5") for n in names_h])
ax.set_yticks(range(len(names_h)))
ax.set_yticklabels(names_h, fontsize=9)
ax.set_xlabel("Predictability horizon (τ_L)")
ax.set_title(f"Predictability Horizon (RMSE > {THRESHOLD})")
ax.grid(True, alpha=0.3, axis='x')

plt.suptitle("T13: Ensemble Short-Time Forecast Errors", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT13_ensemble_errors.png", dpi=120)
plt.close()
print("  Saved: figures/figT13_ensemble_errors.png")

log_event("T13", "script_complete",
          config={"n_ic": N_IC, "T_max_steps": T_MAX_STEPS, "threshold": THRESHOLD},
          metrics={"horizons": {k: float(v) for k, v in horizons.items()}})
print("\nT13 complete.")
