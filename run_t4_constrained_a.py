"""
run_t4_constrained_a.py - Task T4: Constrained Stabilized NODE
===============================================================
Tests three parameterizations of the linear term A in the stabilized NODE:

  1. Unconstrained A (baseline, as in current model)
  2. Negative-definite: A = -(B^T B + eps*I)  [always dissipative]
  3. Diagonal-negative: A = -softplus(diag_vec) [elementwise negative]
  4. Fourier-diagonal: A = diag(L_k) where L_k are KSE eigenvalues
     (Physics-informed initialization of A, unconstrained after that)

Each variant is trained with MSE loss for 600 epochs, then:
  - Rollout stability tested (4000 steps)
  - Lyapunov spectrum computed (20-mode)
  - Power spectrum compared

Outputs:
  data/constrained_a_results.pkl
  figures/figT4_constrained_a.png
"""

import sys
sys.path = [p for p in sys.path if 'MAT6215' not in p]
sys.path.insert(0, '.')
import gpu_config  # sets XLA thread flags before jax import

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from ks_solver import KSSolver
from neural_ode import init_mlp, mlp_forward, prepare_training_data
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

# ── Lyapunov utility (same as T3) ──────────────────────────────────────────────
def rk4_step_generic(rhs_fn, params, u, dt):
    k1 = rhs_fn(params, u)
    k2 = rhs_fn(params, u + dt/2*k1)
    k3 = rhs_fn(params, u + dt/2*k2)
    k4 = rhs_fn(params, u + dt*k3)
    return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def kaplan_yorke(le):
    cs = np.cumsum(le)
    k = np.where(cs < 0)[0]
    if len(k) == 0:
        return float(len(le))
    k = k[0]
    return float(k) + (cs[k-1] if k > 0 else 0.0) / abs(le[k])


def compute_lyapunov(rhs_fn, params, u0, n_modes=20, n_steps=1500, dt=0.25):
    N = u0.shape[0]
    u0j = jnp.array(u0, dtype=jnp.float64)
    Q0 = jnp.eye(N, n_modes, dtype=jnp.float64)
    log0 = jnp.zeros(n_modes, dtype=jnp.float64)
    step = jax.jit(lambda u: rk4_step_generic(rhs_fn, params, u, dt))

    def benettin(carry, _):
        u, Q, ls = carry
        Q_raw = jax.vmap(lambda q: jax.jvp(step, (u,), (q,))[1],
                         in_axes=1, out_axes=1)(Q)
        u_n = step(u)
        Q_n, R = jnp.linalg.qr(Q_raw)
        s = jnp.sign(jnp.diag(R))
        Q_n = Q_n * s[None, :]
        R = R * s[:, None]
        return (u_n, Q_n, ls + jnp.log(jnp.abs(jnp.diag(R)))), None

    (_, _, log_tot), _ = jax.lax.scan(benettin, (u0j, Q0, log0), None, length=n_steps)
    return np.array(log_tot / (n_steps * dt))


# ── Four NODE variants ─────────────────────────────────────────────────────────

# --- Variant 1: Unconstrained A (baseline from neural_ode.py) ---
def init_unconstrained(key, N=64, hidden=128, n_layers=3):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (N, N)) * 0.01
    sizes = [N] + [hidden] * n_layers + [N]
    return {"A": A, "mlp": init_mlp(k2, sizes, scale=0.01)}

def rhs_unconstrained(params, u):
    return u @ params["A"].T + mlp_forward(params["mlp"], u)


# --- Variant 2: Negative-definite A = -(B^T B + eps*I) ---
EPS_NEGDEF = 1e-3  # fixed regularization, not a parameter

def init_negdef(key, N=64, hidden=128, n_layers=3):
    k1, k2 = jax.random.split(key)
    B = jax.random.normal(k1, (N, N)) * 0.1
    sizes = [N] + [hidden] * n_layers + [N]
    return {"B": B, "mlp": init_mlp(k2, sizes, scale=0.01)}

def rhs_negdef(params, u):
    B = params["B"]
    A = -(B.T @ B + EPS_NEGDEF * jnp.eye(B.shape[0]))
    return u @ A.T + mlp_forward(params["mlp"], u)


# --- Variant 3: Diagonal-negative A = -softplus(diag_vec) ---
def init_diag_neg(key, N=64, hidden=128, n_layers=3):
    k1, k2 = jax.random.split(key)
    # Initialize so softplus(d_vec) ≈ 0.1 (small dissipation to start)
    d_vec = jnp.ones(N) * (-2.0)  # softplus(-2) ≈ 0.13
    sizes = [N] + [hidden] * n_layers + [N]
    return {"d_vec": d_vec, "mlp": init_mlp(k2, sizes, scale=0.01)}

def rhs_diag_neg(params, u):
    # A = diag(-softplus(d_vec)) — always negative diagonal
    diag_A = -jax.nn.softplus(params["d_vec"])
    return diag_A * u + mlp_forward(params["mlp"], u)


# --- Variant 4: Physics-informed A init (fixed to true KSE linear operator) ---
def init_physics_informed(key, N=64, hidden=128, n_layers=3, L=22.0):
    k2 = key
    # KSE linear operator: L_k = q_k^2 - q_k^4
    q = jnp.array([2 * jnp.pi * k / L for k in range(N)])
    L_k = q**2 - q**4
    # For physical space, A is diagonal in Fourier space but not physical space.
    # Simplification: use diagonal approximation in physical space with L_k values
    # This is an approximation — exact would require DFT basis change
    A_diag = jnp.real(jnp.fft.ifft(L_k))  # approximate diagonal in physical space
    A = jnp.diag(A_diag)
    sizes = [N] + [hidden] * n_layers + [N]
    return {"A": A, "mlp": init_mlp(k2, sizes, scale=0.01)}

def rhs_physics_informed(params, u):
    return u @ params["A"].T + mlp_forward(params["mlp"], u)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
traj_train = np.load("data/traj_train.npy")
traj_analysis = np.load("data/traj_analysis.npy")
le_true = np.load("data/lyapunov_exponents_full.npy")

solver = KSSolver(L=22.0, N=64, dt=0.25)
u0_lyap = traj_analysis[500].astype(np.float64)

print(f"True KSE: L1={le_true[0]:.4f}, n_pos={int(np.sum(le_true>0))}, "
      f"D_KY={kaplan_yorke(le_true):.2f}")

print("\nPreparing MSE training data (cached)...")
data = prepare_training_data(
    traj_train, solver,
    compute_jacobians=False,
    subsample=2,
    cache_path="data/mse_training_cache.npz"
)
u_d = jnp.array(data["u"], dtype=jnp.float64)
rhs_d = jnp.array(data["rhs"], dtype=jnp.float64)
print(f"  {len(u_d)} training points")


# ── Generic MSE training ───────────────────────────────────────────────────────
def train_mse(params, rhs_fn, n_epochs=600, batch_size=256, lr_init=1e-3, lr_final=1e-5):
    n_data = len(u_d)
    lr_sched = optax.exponential_decay(lr_init, n_epochs, lr_final/lr_init)
    optimizer = optax.adam(lr_sched)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, u_b, rhs_b):
        def loss_fn(p):
            rhs_pred = jax.vmap(lambda u: rhs_fn(p, u))(u_b)
            return jnp.mean((rhs_pred - rhs_b)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    rng = np.random.default_rng(0)
    loss_hist = []
    for epoch in range(n_epochs):
        idx = rng.permutation(n_data)
        el = []
        for s in range(0, n_data - batch_size + 1, batch_size):
            b = idx[s:s+batch_size]
            params, opt_state, l = step(params, opt_state, u_d[b], rhs_d[b])
            el.append(float(l))
        ml = np.mean(el)
        loss_hist.append(ml)
        if (epoch+1) % (n_epochs//5) == 0 or epoch == n_epochs-1:
            print(f"  Epoch {epoch+1:5d}/{n_epochs}  loss={ml:.5f}")
    return params, loss_hist


# ── Run all variants ───────────────────────────────────────────────────────────
variants = {
    "unconstrained": (init_unconstrained, rhs_unconstrained),
    "negdef":        (init_negdef,        rhs_negdef),
    "diag_neg":      (init_diag_neg,      rhs_diag_neg),
    "physics_init":  (init_physics_informed, rhs_physics_informed),
}

results = {}

for name, (init_fn, rhs_fn) in variants.items():
    print(f"\n{'='*60}")
    print(f"Variant: {name}")
    print(f"{'='*60}")
    variant_start = time.time()

    key = jax.random.PRNGKey(hash(name) % 2**32)
    params = init_fn(key, N=64, hidden=128, n_layers=3)

    train_start = time.time()
    params, loss_hist = train_mse(params, rhs_fn, n_epochs=600)
    train_wall_s = time.time() - train_start

    # Rollout stability
    print(f"  Testing rollout (4000 steps)...")
    u_curr = jnp.array(traj_analysis[0], dtype=jnp.float64)
    traj_out = []
    stable = True
    try:
        step_jit = jax.jit(lambda u: rk4_step_generic(rhs_fn, params, u, 0.25))
        for i in range(4000):
            u_curr = step_jit(u_curr)
            if jnp.any(jnp.isnan(u_curr)) or jnp.linalg.norm(u_curr) > 1e6:
                stable = False
                print(f"  Diverged at step {i}")
                break
            if i % 200 == 0:
                traj_out.append(np.array(u_curr))
        if stable:
            print(f"  Stable for 4000 steps, final norm={float(jnp.linalg.norm(u_curr)):.2f}")
    except Exception as e:
        stable = False
        print(f"  Rollout error: {e}")

    # Lyapunov
    print(f"  Computing Lyapunov (20 modes, 500 steps screen)...")
    try:
        le = compute_lyapunov(rhs_fn, params, u0_lyap, n_modes=20, n_steps=500)
        if np.any(np.abs(le) < 0.02):
            print(f"    Near-zero exponents detected -- escalating to 1500 steps...")
            le = compute_lyapunov(rhs_fn, params, u0_lyap, n_modes=20, n_steps=1500)
        l_summary = {
            "L1": float(le[0]),
            "n_pos": int(np.sum(le > 0)),
            "D_KY": kaplan_yorke(le),
            "h_KS": float(np.sum(le[le > 0])),
            "exponents": le
        }
        print(f"  L1={l_summary['L1']:+.4f}, n_pos={l_summary['n_pos']}, "
              f"D_KY={l_summary['D_KY']:.2f}, h_KS={l_summary['h_KS']:.4f}")
    except Exception as e:
        print(f"  Lyapunov failed: {e}")
        l_summary = None

    # Full rollout for diagnostics
    u_curr2 = jnp.array(traj_analysis[0], dtype=jnp.float64)
    traj_full = []
    step_jit2 = jax.jit(lambda u: rk4_step_generic(rhs_fn, params, u, 0.25))
    for _ in range(2000):
        u_curr2 = step_jit2(u_curr2)
        if jnp.any(jnp.isnan(u_curr2)) or jnp.linalg.norm(u_curr2) > 1e6:
            break
        traj_full.append(np.array(u_curr2))
    traj_full = np.array(traj_full) if traj_full else np.zeros((1, 64))
    energy = float(np.mean(np.sum(traj_full**2, axis=1)))

    results[name] = {
        "params": params,
        "loss_history": loss_hist,
        "final_loss": loss_hist[-1],
        "stable": stable,
        "lyapunov": l_summary,
        "energy": energy,
        "traj": traj_full[:500],
    }
    log_event(
        "T4",
        "variant_complete",
        config={
            "variant": name,
            "epochs": 600,
            "batch_size": 256,
            "subsample": 2,
        },
        metrics={
            "final_loss": float(loss_hist[-1]),
            "stable": stable,
            "energy": energy,
            **(l_summary or {}),
        },
        timings={
            "train_wall_s": train_wall_s,
            "total_wall_s": time.time() - variant_start,
        },
    )

with open("data/constrained_a_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: data/constrained_a_results.pkl")

# ── Summary table ──────────────────────────────────────────────────────────────
true_energy = float(np.mean(np.sum(traj_analysis[:2000]**2, axis=1)))
print("\n" + "="*75)
print(f"{'Variant':<18} {'Loss':>8} {'Stable':>7} {'L1':>8} {'n_pos':>6} "
      f"{'D_KY':>7} {'h_KS':>8} {'Energy':>8}")
print("-"*75)
print(f"{'True KSE':<18} {'':>8} {'Yes':>7} {le_true[0]:>+8.4f} "
      f"{int(np.sum(le_true>0)):>6d} {kaplan_yorke(le_true):>7.2f} "
      f"{float(np.sum(le_true[le_true>0])):>8.4f} {true_energy:>8.1f}")
for name, r in results.items():
    s = r["lyapunov"]
    stab = "Yes" if r["stable"] else "No"
    if s:
        print(f"{name:<18} {r['final_loss']:>8.5f} {stab:>7} {s['L1']:>+8.4f} "
              f"{s['n_pos']:>6d} {s['D_KY']:>7.2f} {s['h_KS']:>8.4f} {r['energy']:>8.1f}")
    else:
        print(f"{name:<18} {r['final_loss']:>8.5f} {stab:>7} {'N/A':>8} {'':>6} {'':>7} {'':>8}")
print("="*75)

# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
colors = {'unconstrained': 'C0', 'negdef': 'C1', 'diag_neg': 'C2', 'physics_init': 'C3'}

# Loss curves
ax = axes[0, 0]
for name, r in results.items():
    ax.semilogy(r["loss_history"], color=colors[name], label=name, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training Loss")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Lyapunov spectra
ax = axes[0, 1]
ax.axhline(0, color='k', lw=0.8, ls='--')
for name, r in results.items():
    if r["lyapunov"]:
        le = r["lyapunov"]["exponents"]
        ax.plot(np.arange(1, len(le)+1), le, 'o-', color=colors[name],
                label=name, ms=4, lw=1.5)
ax.set_xlabel("Index")
ax.set_ylabel("Lyapunov exponent")
ax.set_title("Lyapunov Spectrum (20 modes)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Energy comparison
ax = axes[0, 2]
names_r = list(results.keys())
energies = [results[n]["energy"] for n in names_r]
bars = ax.bar(range(len(names_r)), energies,
              color=[colors[n] for n in names_r])
ax.axhline(true_energy, ls='--', color='k', lw=2, label='True KSE')
ax.set_xticks(range(len(names_r)))
ax.set_xticklabels([n.replace('_', '\n') for n in names_r], fontsize=8)
ax.set_ylabel("Mean ||u||^2")
ax.set_title("Rollout Energy")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# n_pos comparison
ax = axes[0, 3]
n_pos_vals = [results[n]["lyapunov"]["n_pos"] if results[n]["lyapunov"] else 0
              for n in names_r]
ax.bar(range(len(names_r)), n_pos_vals, color=[colors[n] for n in names_r])
ax.axhline(int(np.sum(le_true > 0)), ls='--', color='k', lw=2, label='True KSE (3)')
ax.set_xticks(range(len(names_r)))
ax.set_xticklabels([n.replace('_', '\n') for n in names_r], fontsize=8)
ax.set_ylabel("n_pos (# positive LEs)")
ax.set_title("Positive Lyapunov Exponents")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Space-time plots
x = np.linspace(0, 22, 64)
t_vals = np.arange(500) * 0.25
for col, name in enumerate(list(results.keys())[:4]):
    ax = axes[1, col]
    traj = results[name]["traj"]
    n_plot = min(500, len(traj))
    if n_plot > 0:
        vmax = max(np.percentile(np.abs(traj[:n_plot]), 98), 0.1)
        ax.pcolormesh(x, t_vals[:n_plot], traj[:n_plot],
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      shading='auto', rasterized=True)
    stable_str = "stable" if results[name]["stable"] else "DIVERGES"
    ax.set_title(f"{name}\n({stable_str})", fontsize=9)
    ax.set_xlabel("x")
    if col == 0:
        ax.set_ylabel("t")

plt.suptitle("T4: Constrained vs Unconstrained Linear Term A", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT4_constrained_a.png", dpi=120)
log_event(
    "T4",
    "script_complete",
    config={"variants": list(variants.keys())},
    metrics={"n_variants": len(variants)},
)
plt.close()
print("  Saved: figures/figT4_constrained_a.png")

print("\nT4 complete.")
