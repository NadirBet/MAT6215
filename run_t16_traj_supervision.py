"""
run_t16_traj_supervision.py - Task T16: Trajectory-Supervision Training
========================================================================
Trains NODE variants where the loss is on rollout states, not the
instantaneous vector field.  Contrasts with vector-field supervision.

Two paradigms:
  VF:   L = mean ||f_θ(u_i) - (du/dt)_i||^2         (current approach)
  TRAJ: L = mean ||Φ_θ(u_0, T_i) - u_{T_i}||^2      (trajectory loss)

where Φ_θ(u_0, T) = integrate f_θ from t=0 to T with Dopri5 (diffrax).

For TRAJ, we use short segments: T ∈ {1, 2, 4, 8} steps (0.25 to 2.0 time units).
Longer segments capture more temporal structure but are more expensive.

Comparison:
  - Same architecture (64->128->128->128->64, tanh, no A term)
  - Same number of epochs (300)
  - VF vs TRAJ-T1 vs TRAJ-T4 vs TRAJ-T8

Metrics: training loss, Lyapunov spectrum, rollout energy, stability.

Outputs:
  data/traj_supervision_results.pkl
  figures/figT16_traj_supervision.png
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

import diffrax

from ks_solver import KSSolver
from experiment_log import log_event
from latent_node import init_mlp, mlp_forward   # reuse MLP utilities

jax.config.update("jax_enable_x64", True)

DT = 0.25
N_EPOCHS = 300
HIDDEN = 128
N_LAYERS = 3

solver = KSSolver(L=22.0, N=64, dt=DT)

print("Loading data...")
traj_train   = np.load("data/traj_train.npy")
traj_analysis = np.load("data/traj_analysis.npy")
le_true       = np.load("data/lyapunov_exponents_full.npy")
u0_lyap = traj_analysis[500].astype(np.float64)


def kaplan_yorke(le):
    cs = np.cumsum(le)
    k = np.where(cs < 0)[0]
    if len(k) == 0:
        return float(len(le))
    k = k[0]
    return float(k) + (cs[k-1] if k > 0 else 0.0) / abs(le[k])


# ── Model: standard MLP NODE ──────────────────────────────────────────────────
def init_node(key, N=64, hidden=HIDDEN, n_layers=N_LAYERS):
    sizes = [N] + [hidden] * n_layers + [N]
    return {"mlp": init_mlp(key, sizes, scale=0.01)}

def node_rhs(params, u):
    return mlp_forward(params["mlp"], u)


# ── Lyapunov via Benettin+JVP ─────────────────────────────────────────────────
def compute_lyapunov(params, u0, n_modes=20, n_steps=1200, dt=DT):
    N = u0.shape[0]
    def rk4(u):
        k1 = node_rhs(params, u)
        k2 = node_rhs(params, u + dt/2*k1)
        k3 = node_rhs(params, u + dt/2*k2)
        k4 = node_rhs(params, u + dt*k3)
        return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    step = jax.jit(rk4)
    Q0   = jnp.eye(N, n_modes, dtype=jnp.float64)
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


# ── VF Training data ─────────────────────────────────────────────────────────
print("\nPreparing VF training data...")
# Build (u, du/dt) pairs from traj_train using exact RHS
subsample = 2
traj_sub = traj_train[::subsample]
u_vf  = jnp.array(traj_sub, dtype=jnp.float64)
rhs_vf = np.zeros_like(traj_sub)
for i in range(len(traj_sub)):
    u_hat = np.fft.fft(traj_sub[i])
    rhs_hat = solver.rhs(u_hat)
    rhs_vf[i] = np.fft.ifft(rhs_hat).real
rhs_vf = jnp.array(rhs_vf, dtype=jnp.float64)
print(f"  VF pairs: {len(u_vf)}")


# ── TRAJ Training data ────────────────────────────────────────────────────────
def build_traj_data(traj, segment_len):
    """
    Build (u0, u_T) pairs for trajectory-supervision loss.
    u0 = traj[i], u_T = traj[i + segment_len]
    """
    N_seq = len(traj) - segment_len
    u0s = jnp.array(traj[:N_seq],             dtype=jnp.float64)
    uTs = jnp.array(traj[segment_len:N_seq + segment_len], dtype=jnp.float64)
    return u0s, uTs


# ── Trajectory integration via Dopri5 ─────────────────────────────────────────
def integrate_node(params, u0_batch, T_steps):
    """
    Integrate NODE forward for T_steps using Dopri5 (diffrax).
    Returns predicted state at t = T_steps * DT.
    """
    T_time = T_steps * DT

    def integrate_one(u0):
        term = diffrax.ODETerm(lambda t, y, args: node_rhs(args, y))
        solver_dop = diffrax.Dopri5()
        sol = diffrax.diffeqsolve(
            term, solver_dop,
            t0=0.0, t1=T_time, dt0=DT,
            y0=u0, args=params,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=T_steps + 10,
        )
        return sol.ys[-1]

    return jax.vmap(integrate_one)(u0_batch)


# ── Training functions ─────────────────────────────────────────────────────────
def train_vf(key, n_epochs=N_EPOCHS, batch_size=256):
    """Train with vector-field loss."""
    params = init_node(key)
    n_data = len(u_vf)
    lr_sched = optax.exponential_decay(1e-3, n_epochs, 1e-5/1e-3)
    opt = optax.adam(lr_sched)
    opt_state = opt.init(params)
    rng = np.random.default_rng(int(key[0]))
    loss_hist = []

    @jax.jit
    def step_fn(params, opt_state, ub, rb):
        def loss_fn(p):
            pred = jax.vmap(lambda u: node_rhs(p, u))(ub)
            return jnp.mean((pred - rb)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    for epoch in range(n_epochs):
        idx = rng.permutation(n_data)
        el = []
        for s in range(0, n_data - batch_size + 1, batch_size):
            b = idx[s:s+batch_size]
            params, opt_state, loss = step_fn(params, opt_state, u_vf[b], rhs_vf[b])
            el.append(float(loss))
        loss_hist.append(np.mean(el))
        if (epoch+1) % (n_epochs//5) == 0 or epoch == n_epochs-1:
            print(f"    Epoch {epoch+1}/{n_epochs} loss={loss_hist[-1]:.6f}")
    return params, loss_hist


def train_traj(key, segment_len, n_epochs=N_EPOCHS, batch_size=64):
    """Train with trajectory-supervision loss."""
    params = init_node(key)
    u0s, uTs = build_traj_data(traj_train[::1], segment_len)
    n_data = len(u0s)
    lr_sched = optax.exponential_decay(1e-3, n_epochs, 1e-5/1e-3)
    opt = optax.adam(lr_sched)
    opt_state = opt.init(params)
    rng = np.random.default_rng(int(key[0]))
    loss_hist = []

    @jax.jit
    def step_fn(params, opt_state, u0_b, uT_b):
        def loss_fn(p):
            pred = integrate_node(p, u0_b, segment_len)
            return jnp.mean((pred - uT_b)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    for epoch in range(n_epochs):
        idx = rng.permutation(n_data)
        el = []
        for s in range(0, n_data - batch_size + 1, batch_size):
            b = idx[s:s+batch_size]
            params, opt_state, loss = step_fn(params, opt_state, u0s[b], uTs[b])
            el.append(float(loss))
        loss_hist.append(np.mean(el))
        if (epoch+1) % (n_epochs//5) == 0 or epoch == n_epochs-1:
            print(f"    Epoch {epoch+1}/{n_epochs} loss={loss_hist[-1]:.6f}")
    return params, loss_hist


def eval_model(params, n_rollout=2000):
    """Evaluate: rollout energy, stability, Lyapunov."""
    # Rollout
    def rk4_step(u):
        k1 = node_rhs(params, u)
        k2 = node_rhs(params, u + DT/2*k1)
        k3 = node_rhs(params, u + DT/2*k2)
        k4 = node_rhs(params, u + DT*k3)
        return u + DT/6*(k1 + 2*k2 + 2*k3 + k4)
    step = jax.jit(rk4_step)
    u = jnp.array(traj_analysis[0], dtype=jnp.float64)
    energies = []
    stable = True
    for _ in range(n_rollout):
        u = step(u)
        if jnp.any(jnp.isnan(u)) or jnp.linalg.norm(u) > 1e6:
            stable = False; break
        energies.append(float(jnp.sum(u**2)))
    energy = float(np.mean(energies)) if energies else float('nan')

    # Lyapunov
    try:
        le = compute_lyapunov(params, u0_lyap, n_modes=20, n_steps=800)
        if np.any(np.abs(le) < 0.01):
            le = compute_lyapunov(params, u0_lyap, n_modes=20, n_steps=1500)
        return {"stable": stable, "energy": energy,
                "L1": float(le[0]), "n_pos": int(np.sum(le>0)),
                "D_KY": kaplan_yorke(le), "h_KS": float(np.sum(le[le>0])),
                "lyapunov": le}
    except Exception as e:
        print(f"    Lyapunov failed: {e}")
        return {"stable": stable, "energy": energy,
                "L1": float('nan'), "n_pos": 0,
                "D_KY": float('nan'), "h_KS": float('nan'), "lyapunov": None}


# ═══════════════════════════════════════════════════════════════════════════════
# Run experiments
# ═══════════════════════════════════════════════════════════════════════════════
results = {}

# --- VF (baseline) ---
print("\n" + "="*55)
print("VF supervision (baseline)")
print("="*55)
t0 = time.time()
key = jax.random.PRNGKey(0)
params_vf, loss_hist_vf = train_vf(key)
metrics_vf = eval_model(params_vf)
results["VF"] = {"loss_history": loss_hist_vf,
                 "segment_len": 0, "tau": 0, **metrics_vf,
                 "runtime": time.time() - t0}
print(f"  VF: L1={metrics_vf['L1']:+.4f} D_KY={metrics_vf['D_KY']:.2f} "
      f"stable={metrics_vf['stable']}")
log_event("T16", "vf_done",
          config={"mode": "VF", "epochs": N_EPOCHS},
          metrics={k: v for k, v in metrics_vf.items() if k != "lyapunov"})

# --- TRAJ variants ---
SEGMENT_LENS = [1, 4, 8, 16]   # in solver steps (tau = seg * DT)

for seg in SEGMENT_LENS:
    tau = seg * DT
    print(f"\n" + "="*55)
    print(f"TRAJ supervision (seg={seg}, tau={tau:.2f})")
    print("="*55)
    t0 = time.time()
    key = jax.random.PRNGKey(seg * 10)
    try:
        params_t, loss_hist_t = train_traj(key, segment_len=seg)
        metrics_t = eval_model(params_t)
        results[f"TRAJ-T{seg}"] = {
            "loss_history": loss_hist_t,
            "segment_len": seg, "tau": tau, **metrics_t,
            "runtime": time.time() - t0,
        }
        print(f"  TRAJ-T{seg}: L1={metrics_t['L1']:+.4f} D_KY={metrics_t['D_KY']:.2f} "
              f"stable={metrics_t['stable']}")
        log_event("T16", "traj_done",
                  config={"mode": f"TRAJ-T{seg}", "seg": seg, "tau": tau, "epochs": N_EPOCHS},
                  metrics={k: v for k, v in metrics_t.items() if k != "lyapunov"})
    except Exception as e:
        print(f"  TRAJ-T{seg} failed: {e}")
        import traceback; traceback.print_exc()

# ── Save ───────────────────────────────────────────────────────────────────────
with open("data/traj_supervision_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: data/traj_supervision_results.pkl")

# ── Summary table ──────────────────────────────────────────────────────────────
true_dky = kaplan_yorke(le_true)
true_hks = float(np.sum(le_true[le_true>0]))
print("\n" + "="*75)
print(f"{'Mode':<12} {'tau':>6} {'Loss':>10} {'L1':>8} {'D_KY':>7} {'h_KS':>8} {'stable':>7}")
print("-"*75)
print(f"{'True KSE':<12} {'':>6} {'':>10} {le_true[0]:>+8.4f} {true_dky:>7.2f} {true_hks:>8.4f} {'Yes':>7}")
for name, r in results.items():
    print(f"{name:<12} {r['tau']:>6.2f} {r['loss_history'][-1]:>10.6f} "
          f"{r['L1']:>+8.4f} {r['D_KY']:>7.2f} {r['h_KS']:>8.4f} {str(r['stable']):>7}")
print("="*75)

# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
colors = {"VF": "C0", "TRAJ-T1": "C1", "TRAJ-T4": "C2", "TRAJ-T8": "C3", "TRAJ-T16": "C4"}

# Training loss curves
ax = axes[0, 0]
for name, r in results.items():
    ax.semilogy(r["loss_history"], color=colors.get(name, "C5"), label=name, lw=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Training Loss Curves")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# D_KY vs supervision horizon
ax = axes[0, 1]
names_s = list(results.keys())
taus_s = [results[n]["tau"] for n in names_s]
dkys_s = [results[n]["D_KY"] for n in names_s]
ax.plot(taus_s, dkys_s, 'o-', color='C0', lw=2, ms=8)
for i, n in enumerate(names_s):
    ax.annotate(n, (taus_s[i], dkys_s[i]), fontsize=7, ha='left', va='bottom')
ax.axhline(true_dky, ls='--', color='k', lw=1.5, label=f'True ({true_dky:.2f})')
ax.set_xlabel("Supervision horizon tau (time units)")
ax.set_ylabel("D_KY")
ax.set_title("D_KY vs Supervision Horizon")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# h_KS vs tau
ax = axes[0, 2]
h_ks_s = [results[n]["h_KS"] for n in names_s]
ax.plot(taus_s, h_ks_s, 'o-', color='C1', lw=2, ms=8)
ax.axhline(true_hks, ls='--', color='k', lw=1.5, label=f'True ({true_hks:.4f})')
ax.set_xlabel("tau"); ax.set_ylabel("h_KS")
ax.set_title("KS Entropy vs Supervision Horizon")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Lyapunov spectra
ax = axes[1, 0]
ax.axhline(0, color='gray', lw=0.8, ls='--')
for name, r in results.items():
    if r["lyapunov"] is not None:
        le = r["lyapunov"]
        ax.plot(range(1, len(le)+1), le, 'o-', color=colors.get(name, "C5"),
                label=name, ms=4, lw=1.5)
ax.plot(range(1, 21), le_true[:20], 'k^-', label='True KSE', ms=5, lw=2)
ax.set_xlabel("Index"); ax.set_ylabel("Lyapunov exponent")
ax.set_title("Lyapunov Spectra (20 modes)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# L1 vs tau
ax = axes[1, 1]
l1s_s = [results[n]["L1"] for n in names_s]
ax.plot(taus_s, l1s_s, 'o-', color='C2', lw=2, ms=8)
ax.axhline(le_true[0], ls='--', color='k', lw=1.5, label=f'True ({le_true[0]:+.4f})')
ax.set_xlabel("tau"); ax.set_ylabel("L1")
ax.set_title("Leading LE vs Supervision Horizon")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Energy comparison
ax = axes[1, 2]
true_energy = float(np.mean(np.sum(traj_analysis[:2000]**2, axis=1)))
energies_s = [results[n]["energy"] for n in names_s]
ax.bar(range(len(names_s)), energies_s,
       color=[colors.get(n, "C5") for n in names_s], alpha=0.8)
ax.axhline(true_energy, ls='--', color='k', lw=1.5, label=f'True ({true_energy:.1f})')
ax.set_xticks(range(len(names_s)))
ax.set_xticklabels(names_s, fontsize=8)
ax.set_ylabel("Rollout Energy"); ax.set_title("Rollout Energy by Mode")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("T16: Vector-Field vs Trajectory-Supervision Training", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT16_traj_supervision.png", dpi=120)
plt.close()
print("  Saved: figures/figT16_traj_supervision.png")

log_event("T16", "script_complete",
          config={"segment_lens": SEGMENT_LENS, "epochs": N_EPOCHS},
          metrics={n: {"D_KY": r["D_KY"], "L1": r["L1"]} for n, r in results.items()})
print("\nT16 complete.")
