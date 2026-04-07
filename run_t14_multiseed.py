"""
run_t14_multiseed.py - Task T14: Multi-Seed Robustness
=======================================================
Retrains the two most informative models with 5 seeds each:
  1. NODE-Std-MSE (hidden=128, n_layers=3, 600 epochs) — main full-space model
  2. Latent NODE (POD d=10, hidden=128, n_layers=3, 500 epochs) — best ROM

For each model+seed, computes:
  - Final training loss
  - L1, n_pos, D_KY, h_KS (Lyapunov)
  - Rollout energy + stability

Reports: mean ± std across seeds for all metrics.

Outputs:
  data/multiseed_results.pkl
  figures/figT14_multiseed.png
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
from neural_ode import init_mlp, mlp_forward, prepare_training_data
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

# ── Config ─────────────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
N_EPOCHS_NODE = 600
N_EPOCHS_LAT  = 500
DT = 0.25

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
traj_train   = np.load("data/traj_train.npy")
traj_analysis = np.load("data/traj_analysis.npy")
le_true      = np.load("data/lyapunov_exponents_full.npy")
solver = KSSolver(L=22.0, N=64, dt=DT)
u0_lyap = traj_analysis[500].astype(np.float64)


def kaplan_yorke(le):
    cs = np.cumsum(le)
    k = np.where(cs < 0)[0]
    if len(k) == 0:
        return float(len(le))
    k = k[0]
    return float(k) + (cs[k-1] if k > 0 else 0.0) / abs(le[k])


# ── Lyapunov via RK4+JVP ─────────────────────────────────────────────────────
def rk4_step(rhs_fn, params, u, dt):
    k1 = rhs_fn(params, u)
    k2 = rhs_fn(params, u + dt/2*k1)
    k3 = rhs_fn(params, u + dt/2*k2)
    k4 = rhs_fn(params, u + dt*k3)
    return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def compute_lyapunov(rhs_fn, params, u0, n_modes=20, n_steps=1500, dt=0.25):
    N = u0.shape[0]
    step = jax.jit(lambda u: rk4_step(rhs_fn, params, u, dt))
    Q0  = jnp.eye(N, n_modes, dtype=jnp.float64)
    log0 = jnp.zeros(n_modes, dtype=jnp.float64)
    u0j  = jnp.array(u0, dtype=jnp.float64)

    def benettin(carry, _):
        u, Q, ls = carry
        Q_raw = jax.vmap(lambda q: jax.jvp(step, (u,), (q,))[1],
                         in_axes=1, out_axes=1)(Q)
        u_n = step(u)
        Q_n, R = jnp.linalg.qr(Q_raw)
        s = jnp.sign(jnp.diag(R))
        Q_n = Q_n * s[None, :]; R = R * s[:, None]
        return (u_n, Q_n, ls + jnp.log(jnp.abs(jnp.diag(R)))), None

    (_, _, log_tot), _ = jax.lax.scan(benettin, (u0j, Q0, log0), None, length=n_steps)
    return np.array(log_tot / (n_steps * dt))


def lyap_summary(le):
    return {
        "L1":    float(le[0]),
        "n_pos": int(np.sum(le > 0)),
        "D_KY":  kaplan_yorke(le),
        "h_KS":  float(np.sum(le[le > 0])),
    }


# ── Prepare MSE training data ──────────────────────────────────────────────────
print("\nPreparing MSE training data...")
mse_data = prepare_training_data(traj_train, solver, compute_jacobians=False,
                                  subsample=2, cache_path="data/mse_training_cache.npz")
u_d   = jnp.array(mse_data["u"],   dtype=jnp.float64)
rhs_d = jnp.array(mse_data["rhs"], dtype=jnp.float64)
print(f"  {len(u_d)} training points")


# ═══════════════════════════════════════════════════════════════════════════════
# Model 1: NODE-Std-MSE (plain MLP, no stabilization)
# ═══════════════════════════════════════════════════════════════════════════════

def init_node_std(key, N=64, hidden=128, n_layers=3):
    sizes = [N] + [hidden]*n_layers + [N]
    return {"mlp": init_mlp(key, sizes, scale=0.01)}

def rhs_node_std(params, u):
    return mlp_forward(params["mlp"], u)

def train_node_std(key, n_epochs=N_EPOCHS_NODE, batch_size=256):
    params = init_node_std(key)
    n_data = len(u_d)
    lr_sched = optax.exponential_decay(1e-3, n_epochs, 1e-5/1e-3)
    opt = optax.adam(lr_sched)
    opt_state = opt.init(params)
    rng = np.random.default_rng(int(key[0]))
    loss_hist = []

    @jax.jit
    def step_fn(params, opt_state, ub, rb):
        def loss_fn(p):
            pred = jax.vmap(lambda u: rhs_node_std(p, u))(ub)
            return jnp.mean((pred - rb)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    for epoch in range(n_epochs):
        idx = rng.permutation(n_data)
        el = []
        for s in range(0, n_data - batch_size + 1, batch_size):
            b = idx[s:s+batch_size]
            params, opt_state, loss = step_fn(params, opt_state, u_d[b], rhs_d[b])
            el.append(float(loss))
        loss_hist.append(np.mean(el))
        if (epoch+1) % (n_epochs//5) == 0 or epoch == n_epochs-1:
            print(f"      Epoch {epoch+1}/{n_epochs} loss={loss_hist[-1]:.5f}")
    return params, loss_hist


print("\n" + "="*60)
print("Model 1: NODE-Std-MSE (5 seeds)")
print("="*60)
node_std_runs = []
for seed in SEEDS:
    print(f"\n  Seed {seed}:")
    key = jax.random.PRNGKey(seed * 100 + 1)
    t0 = time.time()
    params, loss_hist = train_node_std(key)
    train_wall = time.time() - t0

    # Rollout energy
    step_jit = jax.jit(lambda u: rk4_step(rhs_node_std, params, u, DT))
    u_curr = jnp.array(traj_analysis[0], dtype=jnp.float64)
    traj_out = []
    stable = True
    for _ in range(2000):
        u_curr = step_jit(u_curr)
        if jnp.any(jnp.isnan(u_curr)) or jnp.linalg.norm(u_curr) > 1e6:
            stable = False; break
        traj_out.append(float(jnp.sum(u_curr**2)))
    energy = float(np.mean(traj_out)) if traj_out else float('nan')

    # Lyapunov (20-mode screen, escalate if needed)
    le = compute_lyapunov(rhs_node_std, params, u0_lyap, n_modes=20, n_steps=800)
    if np.any(np.abs(le) < 0.01):
        le = compute_lyapunov(rhs_node_std, params, u0_lyap, n_modes=20, n_steps=1500)
    summary = lyap_summary(le)

    run = {"seed": seed, "final_loss": loss_hist[-1], "stable": stable,
           "energy": energy, **summary, "train_wall": train_wall}
    node_std_runs.append(run)
    print(f"    loss={run['final_loss']:.5f} L1={run['L1']:+.4f} "
          f"D_KY={run['D_KY']:.2f} stable={stable}")
    log_event("T14", "node_std_seed_done",
              config={"model": "NODE-Std-MSE", "seed": seed, "epochs": N_EPOCHS_NODE},
              metrics=run)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 2: Latent NODE (POD d=10)
# ═══════════════════════════════════════════════════════════════════════════════
from latent_node import (
    fit_pod_autoencoder, pod_encode, pod_decode,
    init_latent_ode, train_latent_ode,
    prepare_latent_data_pod, rollout_latent_node,
    compute_latent_lyapunov, kaplan_yorke as lat_ky,
)

D_LAT = 10
print(f"\n\nFitting POD (d={D_LAT})...")
pod = fit_pod_autoencoder(traj_train, D_LAT)
encode_fn = lambda u: pod_encode(pod, u)
decode_fn = lambda h: pod_decode(pod, h)

print("Preparing latent data...")
lat_data = prepare_latent_data_pod(traj_train, pod, solver, subsample=2)
h_d   = lat_data["h"]
dhdt_d = lat_data["dhdt"]

print("\n" + "="*60)
print("Model 2: Latent NODE (POD d=10, 5 seeds)")
print("="*60)
lat_runs = []
h0_lyap = np.array(encode_fn(jnp.array(u0_lyap)))

for seed in SEEDS:
    print(f"\n  Seed {seed}:")
    key = jax.random.PRNGKey(seed * 100 + 2)
    t0 = time.time()
    ode_params, loss_hist = train_latent_ode(
        init_latent_ode(key, d=D_LAT, hidden=128, n_layers=3),
        h_d, dhdt_d, n_epochs=N_EPOCHS_LAT, batch_size=256, key=key)
    train_wall = time.time() - t0

    # Rollout
    try:
        traj_lat, _ = rollout_latent_node(
            pod, ode_params, traj_analysis[0].astype(np.float64),
            n_steps=2000, dt=DT, encode_fn=encode_fn, decode_fn=decode_fn)
        energy = float(np.mean(np.sum(traj_lat**2, axis=1)))
        stable = not (np.any(np.isnan(traj_lat)) or energy > 1e6)
    except Exception:
        energy = float('nan'); stable = False

    # Lyapunov
    le_lat = compute_latent_lyapunov(ode_params, h0_lyap, n_steps=800, dt=DT)
    if np.any(np.abs(le_lat) < 0.01):
        le_lat = compute_latent_lyapunov(ode_params, h0_lyap, n_steps=1500, dt=DT)
    s = lyap_summary(le_lat)

    run = {"seed": seed, "final_loss": loss_hist[-1], "stable": stable,
           "energy": energy, **s, "train_wall": train_wall}
    lat_runs.append(run)
    print(f"    loss={run['final_loss']:.5f} L1={run['L1']:+.4f} "
          f"D_KY={run['D_KY']:.2f} stable={stable}")
    log_event("T14", "latent_node_seed_done",
              config={"model": "Latent-NODE-d10", "seed": seed, "epochs": N_EPOCHS_LAT},
              metrics=run)


# ── Summary stats ──────────────────────────────────────────────────────────────
def stats(runs, key):
    vals = [r[key] for r in runs if not np.isnan(r[key])]
    return np.mean(vals), np.std(vals)

print("\n" + "="*70)
print("MULTI-SEED SUMMARY")
print("="*70)
print(f"\nTrue KSE: L1={le_true[0]:+.4f}, D_KY={kaplan_yorke(le_true):.2f}, "
      f"h_KS={float(np.sum(le_true[le_true>0])):.4f}")

for model_name, runs in [("NODE-Std-MSE", node_std_runs), ("Latent-NODE-d10", lat_runs)]:
    print(f"\n{model_name}:")
    for metric in ["final_loss", "L1", "D_KY", "h_KS", "energy"]:
        m, s = stats(runs, metric)
        print(f"  {metric:<15}: {m:.4f} ± {s:.4f}")
    n_stable = sum(r["stable"] for r in runs)
    print(f"  stable: {n_stable}/{len(runs)}")


# ── Save ───────────────────────────────────────────────────────────────────────
results = {
    "node_std_mse": node_std_runs,
    "latent_node_d10": lat_runs,
    "seeds": SEEDS,
    "true_kse": {
        "L1": float(le_true[0]),
        "D_KY": kaplan_yorke(le_true),
        "h_KS": float(np.sum(le_true[le_true > 0])),
    }
}
with open("data/multiseed_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: data/multiseed_results.pkl")


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
metrics = ["final_loss", "L1", "D_KY", "h_KS", "energy"]
true_vals = [None, float(le_true[0]), kaplan_yorke(le_true),
             float(np.sum(le_true[le_true>0])), None]
ylabels  = ["Final Loss", "L1", "D_KY", "h_KS", "Energy"]

ax_flat = axes.flatten()
for col, (metric, true_val, ylabel) in enumerate(zip(metrics, true_vals, ylabels)):
    ax = ax_flat[col]
    for run_set, color, label in [
        (node_std_runs, "C0", "NODE-Std-MSE"),
        (lat_runs,      "C1", "Latent-NODE-d10"),
    ]:
        vals = [r[metric] for r in run_set]
        x_offset = 0 if color == "C0" else 0.3
        x_pos = [s + x_offset for s in SEEDS]
        ax.scatter(x_pos, vals, color=color, s=60, zorder=3, label=label)
        m, s = stats(run_set, metric)
        ax.axhline(m, color=color, lw=2, ls='-', alpha=0.5)
        ax.fill_between([-0.5, len(SEEDS)-0.5], m-s, m+s,
                        alpha=0.1, color=color)
    if true_val is not None:
        ax.axhline(true_val, color='k', lw=1.5, ls='--', label='True KSE')
    ax.set_xlabel("Seed"); ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (5 seeds)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xticks(SEEDS)

# Seed-to-seed consistency radar (last panel: std/mean for each metric)
ax = ax_flat[5]
metrics_short = metrics[1:5]   # skip loss and energy
for run_set, color, label in [
    (node_std_runs, "C0", "NODE-Std-MSE"),
    (lat_runs,      "C1", "Latent-NODE-d10"),
]:
    cvs = []
    for m in metrics_short:
        mean, std = stats(run_set, m)
        cvs.append(std / (abs(mean) + 1e-10))
    ax.bar([i + (0.35 if color=="C1" else 0) for i in range(len(metrics_short))],
           cvs, width=0.3, color=color, label=label, alpha=0.8)
ax.set_xticks(range(len(metrics_short)))
ax.set_xticklabels(metrics_short, fontsize=9)
ax.set_ylabel("CV (std/|mean|)")
ax.set_title("Seed-to-Seed Variability")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("T14: Multi-Seed Robustness (5 seeds)", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT14_multiseed.png", dpi=120)
plt.close()
print("  Saved: figures/figT14_multiseed.png")

log_event("T14", "script_complete",
          config={"seeds": SEEDS, "models": ["NODE-Std-MSE", "Latent-NODE-d10"]},
          metrics={
              "node_std_L1_mean": float(stats(node_std_runs, "L1")[0]),
              "node_std_L1_std":  float(stats(node_std_runs, "L1")[1]),
              "lat_L1_mean":      float(stats(lat_runs, "L1")[0]),
              "lat_L1_std":       float(stats(lat_runs, "L1")[1]),
          })
print("\nT14 complete.")
