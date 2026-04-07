"""
run_t20_l44.py - Task T20: Scope Expansion to L=44
====================================================
Repeats the main experiments (Neural ODE + latent NODE + SINDy) at L=44
to check whether the MSE vs dynamical-fidelity gap worsens at larger systems.

L=44 has roughly twice as many positive Lyapunov exponents as L=22.
True KSE at L=44: n_pos ≈ 8-10, D_KY ≈ 12-14.

Experiments:
  1. Simulate true KSE at L=44 (N=128 modes, dt=0.25)
  2. Compute Lyapunov spectrum (first 20 exponents)
  3. Train NODE-Std-MSE at L=44
  4. Train Latent-NODE (d=16) at L=44
  5. Train SINDy at L=44 (Galerkin, degree=2)
  6. Compare Lyapunov spectra and D_KY across methods
  7. Compare W1 distance from invariant measure

Output:
  data/l44_results.pkl
  figures/figT20_l44_lyapunov_spectra.png
  figures/figT20_l44_summary.png
"""

import sys
sys.path = [p for p in sys.path if 'MAT6215' not in p]
sys.path.insert(0, '.')
import gpu_config

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ks_solver import KSSolver
from neural_ode import init_standard_node, standard_node_rhs
from latent_node import (
    fit_pod_autoencoder, pod_encode, pod_decode,
    init_latent_ode, train_latent_ode,
    prepare_latent_data_pod, rollout_latent_node,
)
from sindy import (
    compute_galerkin_basis, project_trajectory,
    polynomial_library, compute_time_derivatives, stlsq, SINDyModel,
)
from diagnostics import compare_systems
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

# ── Configuration ──────────────────────────────────────────────────────────────
L44 = 44.0
N44 = 128        # 2/3 dealiasing → ~85 active modes
DT = 0.25
N_WARMUP = 2000   # 500 tu warmup
N_TRAIN  = 40000  # 10000 tu training
N_LYAP   = 8000   # 2000 tu for Lyapunov
N_DIAG   = 4000   # 1000 tu for diagnostics
N_CLV    = 20     # first 20 Lyapunov vectors

NODE_EPOCHS  = 300
NODE_HIDDEN  = 128
NODE_LAYERS  = 3
LAT_D        = 16    # larger latent dim for larger system
LAT_EPOCHS   = 300
SINDY_MODES  = 10
SINDY_THRESH = 0.05

print(f"T20: L=44 scope expansion")
print(f"  N={N44} modes, dt={DT}, train={N_TRAIN} steps")

results = {}

# ──────────────────────────────────────────────────────────────────────────────
# 1. Simulate KSE at L=44
# ──────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Simulating KSE at L=44...")
solver44 = KSSolver(L=L44, N=N44, dt=DT)

# Warmup
rng = np.random.default_rng(0)
u0_init = rng.standard_normal(N44) * 0.1
u_hat = np.fft.rfft(u0_init)
for _ in range(N_WARMUP):
    u_hat = solver44.step(u_hat)
u_warmup = np.fft.irfft(u_hat, n=N44)
print(f"  Warmup done. Initial energy: {float(np.mean(u_warmup**2)):.2f}")

# Training trajectory
t0 = time.time()
traj_train_44 = np.zeros((N_TRAIN, N44))
u_hat = np.fft.rfft(u_warmup)
for i in range(N_TRAIN):
    u_hat = solver44.step(u_hat)
    traj_train_44[i] = np.fft.irfft(u_hat, n=N44)
print(f"  Training traj: {N_TRAIN} steps, {time.time()-t0:.1f}s")
print(f"  Energy: {float(np.mean(np.sum(traj_train_44**2, axis=1))):.2f}")

# Analysis trajectory (for Lyapunov + diagnostics)
u_hat2 = np.fft.rfft(traj_train_44[-100])
traj_analysis_44 = np.zeros((N_LYAP, N44))
for i in range(N_LYAP):
    u_hat2 = solver44.step(u_hat2)
    traj_analysis_44[i] = np.fft.irfft(u_hat2, n=N44)
print(f"  Analysis traj: {N_LYAP} steps")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Lyapunov spectrum at L=44 (Benettin method, using JAX JVP)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Computing Lyapunov spectrum for true KSE at L=44...")
t0 = time.time()

def step_physical(u):
    u_hat = jnp.fft.rfft(u)
    u_hat_next = solver44.step(np.array(u_hat))
    return jnp.fft.irfft(jnp.array(u_hat_next), n=N44)

step_jit = jax.jit(step_physical)

# Initialize QR iteration
u_benettin = traj_analysis_44[50].copy()
Q = np.eye(N44, N_CLV)
les = np.zeros(N_CLV)
n_steps_lyap = min(N_LYAP - 100, 3000)
transient = 500

for i in range(n_steps_lyap):
    # Push state
    _, Jvp_fn = jax.linearize(step_jit, jnp.array(u_benettin))
    u_benettin = np.array(step_jit(jnp.array(u_benettin)))

    # Evolve perturbations via JVP
    Q_new = np.column_stack([np.array(Jvp_fn(jnp.array(Q[:, k])))
                              for k in range(N_CLV)])
    # QR
    Q, R = np.linalg.qr(Q_new)
    if i >= transient:
        les += np.log(np.abs(np.diag(R)))

les /= (n_steps_lyap - transient) * DT

def kaplan_yorke(le):
    cs = np.cumsum(le)
    k_arr = np.where(cs < 0)[0]
    if len(k_arr) == 0:
        return float(len(le))
    k = k_arr[0]
    return float(k) + (cs[k-1] if k > 0 else 0.0) / abs(le[k])

dky_44 = kaplan_yorke(les)
h_ks_44 = float(np.sum(les[les > 0]))
n_pos_44 = int(np.sum(les > 0))

print(f"  L44 Lyapunov: L1={les[0]:+.4f}, n_pos={n_pos_44}, D_KY={dky_44:.2f}, h_KS={h_ks_44:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

results["true_kse_l44"] = {
    "lyapunov_spectrum": les.tolist(),
    "L1": float(les[0]),
    "n_pos": n_pos_44,
    "D_KY": dky_44,
    "h_KS": h_ks_44,
    "n_computed": N_CLV,
}

u0_diag = traj_analysis_44[100].copy()

# ──────────────────────────────────────────────────────────────────────────────
# 3. NODE-Std-MSE at L=44
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[3/6] Training NODE-Std-MSE at L=44 ({NODE_EPOCHS} epochs)...")
t0 = time.time()

# Data: (u_t, u_{t+1}) pairs for one-step prediction
X_node = traj_train_44[:-1]
Y_node = traj_train_44[1:]

# Use simple RK4 + MSE loss (direct vectorfield matching via finite diff)
key = jax.random.PRNGKey(42)
node_params = init_standard_node(key, n=N44, hidden=NODE_HIDDEN, n_layers=NODE_LAYERS)

# Compute approximate dudt via finite differences
dudt_train = (Y_node - X_node) / DT   # shape (N_TRAIN-1, N44)

@jax.jit
def mse_loss(params, x_batch, dudt_batch):
    pred = jax.vmap(lambda x: standard_node_rhs(params, x))(x_batch)
    return jnp.mean((pred - dudt_batch)**2)

@jax.jit
def update(params, opt_state, x_batch, dudt_batch):
    loss, grads = jax.value_and_grad(mse_loss)(params, x_batch, dudt_batch)
    updates, opt_state_new = optimizer.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, loss

schedule = optax.exponential_decay(1e-3, transition_steps=5000, decay_rate=0.5)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(node_params)

BATCH = 512
n_data = len(X_node)
loss_history = []

for epoch in range(NODE_EPOCHS):
    perm = np.random.permutation(n_data)
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, n_data - BATCH, BATCH):
        idx = perm[i:i+BATCH]
        node_params, opt_state, loss = update(
            node_params, opt_state,
            jnp.array(X_node[idx]), jnp.array(dudt_train[idx])
        )
        epoch_loss += float(loss)
        n_batches += 1
    avg_loss = epoch_loss / max(n_batches, 1)
    loss_history.append(avg_loss)
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{NODE_EPOCHS}: loss={avg_loss:.5f}")

print(f"  Training done in {time.time()-t0:.1f}s, final loss={loss_history[-1]:.5f}")

# Rollout
def rollout_rk4_node(params, u0, n_steps):
    @jax.jit
    def step(u):
        k1 = standard_node_rhs(params, u)
        k2 = standard_node_rhs(params, u + DT/2*k1)
        k3 = standard_node_rhs(params, u + DT/2*k2)
        k4 = standard_node_rhs(params, u + DT*k3)
        return u + DT/6*(k1 + 2*k2 + 2*k3 + k4)
    u = jnp.array(u0, dtype=jnp.float64)
    traj = []
    for i in range(n_steps):
        u = step(u)
        if jnp.any(jnp.isnan(u)) or jnp.linalg.norm(u) > 1e6:
            print(f"    Diverged at step {i}")
            break
        traj.append(np.array(u))
    return np.array(traj) if traj else None

traj_node = rollout_rk4_node(node_params, u0_diag, N_DIAG)
node_stable = (traj_node is not None and len(traj_node) >= N_DIAG // 2)
print(f"  Rollout: {'stable' if node_stable else 'DIVERGED'}, {len(traj_node) if traj_node is not None else 0} steps")

# Quick Lyapunov for NODE
node_les = np.full(N_CLV, np.nan)
if node_stable:
    def node_step(u):
        k1 = standard_node_rhs(node_params, u)
        k2 = standard_node_rhs(node_params, u + DT/2*k1)
        k3 = standard_node_rhs(node_params, u + DT/2*k2)
        k4 = standard_node_rhs(node_params, u + DT*k3)
        return u + DT/6*(k1 + 2*k2 + 2*k3 + k4)
    node_step_jit = jax.jit(node_step)
    u_b = jnp.array(u0_diag)
    Q_n = np.eye(N44, N_CLV)
    le_n = np.zeros(N_CLV)
    n_lyap_node = min(1500, len(traj_node) - 100)
    for i in range(n_lyap_node):
        _, Jvp_fn = jax.linearize(node_step_jit, u_b)
        u_b = node_step_jit(u_b)
        Q_new = np.column_stack([np.array(Jvp_fn(jnp.array(Q_n[:, k])))
                                  for k in range(N_CLV)])
        Q_n, R_n = np.linalg.qr(Q_new)
        if i >= 300:
            le_n += np.log(np.abs(np.diag(R_n)))
    le_n /= (n_lyap_node - 300) * DT
    node_les = le_n

node_dky = kaplan_yorke(node_les[~np.isnan(node_les)]) if not np.all(np.isnan(node_les)) else np.nan
node_npos = int(np.sum(node_les[~np.isnan(node_les)] > 0))
print(f"  NODE L44: L1={node_les[0]:+.4f}, n_pos={node_npos}, D_KY={node_dky:.2f}")

results["node_std_mse_l44"] = {
    "params": node_params,
    "loss_history": loss_history,
    "stable": node_stable,
    "lyapunov_spectrum": node_les.tolist(),
    "L1": float(node_les[0]),
    "n_pos": node_npos,
    "D_KY": float(node_dky),
    "traj": traj_node,
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. Latent NODE (d=16) at L=44
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[4/6] Training Latent-NODE (d={LAT_D}) at L=44 ({LAT_EPOCHS} epochs)...")
t0 = time.time()

try:
    pod44 = fit_pod_autoencoder(traj_train_44, LAT_D)
    lat_data44 = prepare_latent_data_pod(traj_train_44, pod44, solver44, subsample=2)
    key_lat = jax.random.PRNGKey(7)
    ode_params44 = init_latent_ode(key_lat, d=LAT_D, hidden=128, n_layers=3)
    ode_params44, lat_loss = train_latent_ode(
        ode_params44, lat_data44["h"], lat_data44["dhdt"],
        n_epochs=LAT_EPOCHS, batch_size=256, key=key_lat)

    traj_lat44, _ = rollout_latent_node(
        pod44, ode_params44, u0_diag, n_steps=N_DIAG, dt=DT,
        encode_fn=lambda u: pod_encode(pod44, u),
        decode_fn=lambda h: pod_decode(pod44, h))

    lat_stable = not np.any(np.isnan(traj_lat44))
    print(f"  Latent-NODE rollout: {'stable' if lat_stable else 'DIVERGED'}, {len(traj_lat44)} steps")
    print(f"  Energy: {float(np.mean(np.sum(traj_lat44**2, axis=1))):.2f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    results["latent_node_l44"] = {
        "stable": lat_stable,
        "traj": traj_lat44,
        "final_loss": float(lat_loss[-1]) if hasattr(lat_loss, '__len__') else float(lat_loss),
        "d": LAT_D,
    }
except Exception as e:
    print(f"  Latent-NODE failed: {e}")
    import traceback; traceback.print_exc()
    traj_lat44 = None
    results["latent_node_l44"] = {"stable": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# 5. SINDy at L=44
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[5/6] Training SINDy at L=44 (n_modes={SINDY_MODES}, thresh={SINDY_THRESH})...")
t0 = time.time()

try:
    Phi, sigma = compute_galerkin_basis(traj_train_44, n_modes=SINDY_MODES)
    a_train = project_trajectory(traj_train_44, Phi)
    dhdt_exact = compute_time_derivatives(a_train, DT, method="finite_diff")
    Theta = polynomial_library(a_train, degree=2, include_bias=True)
    Xi, active = stlsq(Theta, dhdt_exact, threshold=SINDY_THRESH, max_iter=10)
    sindy44 = SINDyModel(Xi, Phi, sigma, degree=2, include_bias=True,
                         threshold=SINDY_THRESH, solver_dt=DT)
    n_terms = int(np.sum(np.abs(Xi) > 0))
    print(f"  SINDy: {n_terms} active terms")

    try:
        traj_sindy44 = sindy44.integrate(u0_diag, n_steps=N_DIAG, dt=DT)
        sindy_ok = not (np.any(np.isnan(traj_sindy44)) or np.any(np.isinf(traj_sindy44)))
        print(f"  Rollout: {'stable' if sindy_ok else 'DIVERGED'}, {len(traj_sindy44)} steps")
    except Exception as e2:
        print(f"  Rollout failed: {e2}")
        traj_sindy44 = None
        sindy_ok = False

    results["sindy_l44"] = {
        "stable": sindy_ok,
        "n_active_terms": n_terms,
        "traj": traj_sindy44 if sindy_ok else None,
    }
    print(f"  Time: {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  SINDy failed: {e}")
    import traceback; traceback.print_exc()
    results["sindy_l44"] = {"stable": False, "error": str(e)}
    traj_sindy44 = None
    sindy_ok = False

# ──────────────────────────────────────────────────────────────────────────────
# 6. Diagnostics comparison
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[6/6] Running diagnostics comparison...")
systems_44 = {"True KSE L=44": traj_analysis_44[:N_DIAG]}

if node_stable and traj_node is not None and len(traj_node) >= 500:
    systems_44["NODE-Std-MSE L=44"] = traj_node[:N_DIAG]

if traj_lat44 is not None and not np.any(np.isnan(traj_lat44)):
    systems_44["Latent-NODE-d16 L=44"] = traj_lat44[:N_DIAG]

if traj_sindy44 is not None and sindy_ok:
    systems_44["SINDy L=44"] = traj_sindy44[:N_DIAG]

print(f"  Comparing {len(systems_44)} systems: {list(systems_44.keys())}")
diag44 = compare_systems(systems_44, L=L44, dt=DT)
results["diagnostics"] = diag44

# ── Save results ───────────────────────────────────────────────────────────────
with open("data/l44_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: data/l44_results.pkl")

# ── Figures ────────────────────────────────────────────────────────────────────
import os
os.makedirs("figures", exist_ok=True)

# --- Lyapunov spectrum comparison ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
les_true_22 = np.load("data/lyapunov_exponents_full.npy")
ax.plot(range(1, len(les_true_22)+1), les_true_22, 'k-o', ms=4, lw=1.5, label="True KSE L=22")
les_44 = np.array(results["true_kse_l44"]["lyapunov_spectrum"])
ax.plot(range(1, len(les_44)+1), les_44, 'b-s', ms=4, lw=1.5, label=f"True KSE L=44")
if not np.all(np.isnan(node_les)):
    ax.plot(range(1, N_CLV+1), node_les, 'r--^', ms=4, lw=1.2, label="NODE-MSE L=44")
ax.axhline(0, ls='--', color='gray', lw=0.8)
ax.set_xlabel("Index i")
ax.set_ylabel("Lyapunov exponent λᵢ")
ax.set_title("Lyapunov Spectrum: L=22 vs L=44")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- D_KY and n_pos comparison ---
ax2 = axes[1]
labels = ["True KSE\nL=22", "True KSE\nL=44"]
dky_vals = [
    float(np.load("data/lyapunov_exponents_full.npy")[0]),  # placeholder
    dky_44,
]
# Recompute L=22 KY from file
les22 = np.load("data/lyapunov_exponents_full.npy")
dky22 = kaplan_yorke(les22)
n_pos22 = int(np.sum(les22 > 0))

bar_data = [
    ("True KSE L=22", dky22, n_pos22),
    ("True KSE L=44", dky_44, n_pos_44),
]
if not np.all(np.isnan(node_les)):
    bar_data.append(("NODE-MSE L=44", float(node_dky), node_npos))
if results.get("sindy_l44", {}).get("stable"):
    pass  # no Lyapunov for SINDy yet

x_pos = range(len(bar_data))
names = [b[0] for b in bar_data]
dkys  = [b[1] for b in bar_data]
npos  = [b[2] for b in bar_data]

ax2.bar(x_pos, dkys, color=['k', 'b', 'r'][:len(bar_data)], alpha=0.7, label="D_KY")
for xi, (dky_v, np_v) in enumerate(zip(dkys, npos)):
    ax2.text(xi, dky_v + 0.1, f"n+={np_v}", ha='center', fontsize=8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
ax2.set_ylabel("Kaplan-Yorke Dimension D_KY")
ax2.set_title("Attractor Complexity: L=22 vs L=44")
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("figures/figT20_l44_lyapunov_spectra.png", dpi=120)
plt.close()
print("  Saved: figures/figT20_l44_lyapunov_spectra.png")

# --- Power spectra ---
fig, ax = plt.subplots(figsize=(9, 5))
colors = {'True KSE L=44': 'k', 'NODE-Std-MSE L=44': 'C0',
          'Latent-NODE-d16 L=44': 'C2', 'SINDy L=44': 'C5'}
for name, r in diag44.items():
    q, E = r["power_spectrum"]
    lw = 2.5 if "True" in name else 1.5
    ls = '-' if "True" in name else '--'
    ax.semilogy(q[1:], E[1:], lw=lw, ls=ls,
                color=colors.get(name, "gray"), label=name)
ax.set_xlabel("Wavenumber q")
ax.set_ylabel("E(q) = |û_q|²")
ax.set_title("Spatial Power Spectra at L=44")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/figT20_l44_summary.png", dpi=120)
plt.close()
print("  Saved: figures/figT20_l44_summary.png")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("T20 SUMMARY: L=44 Results")
print("="*65)
print(f"  True KSE L=22: n_pos={n_pos22}, D_KY={dky22:.2f}")
print(f"  True KSE L=44: n_pos={n_pos_44}, D_KY={dky_44:.2f}")
if not np.all(np.isnan(node_les)):
    print(f"  NODE-MSE  L=44: n_pos={node_npos}, D_KY={node_dky:.2f}, stable={node_stable}")
print(f"  SINDy     L=44: stable={results['sindy_l44']['stable']}")
print("="*65)
print(f"\nKey finding: D_KY doubled ({dky22:.1f} → {dky_44:.1f}) — is surrogate gap also larger?")
if not np.all(np.isnan(node_les)):
    dky_err22 = abs(0.0381 - float(les_true_22[0])) / abs(float(les_true_22[0]))   # rough
    dky_err44 = abs(dky_44 - node_dky) / abs(dky_44) if dky_44 != 0 else np.nan
    print(f"  D_KY relative error (NODE): L=44 → {dky_err44:.1%}")

log_event("t20_l44", "script_complete",
          config={"L": L44, "N": N44, "node_epochs": NODE_EPOCHS, "lat_d": LAT_D},
          metrics={"dky_true": dky_44, "n_pos_true": n_pos_44,
                   "node_stable": node_stable,
                   "sindy_stable": results["sindy_l44"]["stable"]})

print("\nT20 complete.")
