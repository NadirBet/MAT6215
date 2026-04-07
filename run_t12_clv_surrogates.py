"""
run_t12_clv_surrogates.py - Task T12: CLV Angles for Surrogates
================================================================
Runs the Ginelli CLV algorithm on:
  1. True KSE (ground truth)
  2. NODE-Std-MSE (most stable surrogate)
  3. NODE-Stab-MSE (if stable enough to run)
  4. Latent NODE (POD d=10, best from T5/T6)

For each, computes:
  - CLV angles between adjacent vectors (unstable-unstable, u-s boundary)
  - Distribution of near-zero angles (tangencies — hyperbolicity measure)
  - Mean angle vs CLV index pair

Key diagnostic (Özalp & Magri 2024): near-zero angle distributions indicate
tangencies between stable and unstable manifolds. Surrogates with wrong
tangent geometry produce qualitatively wrong statistics even if MSE is low.

Outputs:
  data/clv_results.pkl
  figures/figT12_clv_angles.png
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
from functools import partial

from ks_solver import KSSolver
from lyapunov import CLVComputer, _physical_discrete_step
from neural_ode import stabilized_node_rhs, standard_node_rhs, init_mlp, mlp_forward
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_CLV    = 8     # number of CLVs (covers unstable + first few stable)
N_STEPS  = 1500  # forward/backward pass steps (need enough for convergence)
N_WARMUP = 200   # discard initial transient from CLV sequence

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
traj_analysis = np.load("data/traj_analysis.npy")
le_true = np.load("data/lyapunov_exponents_full.npy")
solver = KSSolver(L=22.0, N=64, dt=0.25)

u0 = traj_analysis[500].astype(np.float64)   # starting point well on attractor


# ═══════════════════════════════════════════════════════════════════════════════
# NODE "solver" wrapper — wraps RK4 around node RHS to make a CLVComputer-
# compatible interface. Mimics solver.step() interface.
# ═══════════════════════════════════════════════════════════════════════════════

class NODESolverWrapper:
    """
    Wraps a NODE RHS function into a solver-like object for CLVComputer.
    The 'discrete step' is one RK4 step of the NODE.
    dt must match training dt (0.25).
    """
    def __init__(self, rhs_fn, params, N=64, dt=0.25):
        self.rhs_fn = rhs_fn
        self.params = params
        self.N = N
        self.dt = dt

    def step(self, u_phys):
        """One RK4 step in physical space."""
        dt = self.dt
        p = self.params
        rhs = self.rhs_fn
        k1 = rhs(p, u_phys)
        k2 = rhs(p, u_phys + dt/2*k1)
        k3 = rhs(p, u_phys + dt/2*k2)
        k4 = rhs(p, u_phys + dt*k3)
        return u_phys + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def _node_discrete_step(node_solver, u_phys):
    """Standalone function for JAX JVP (used by CLVComputer)."""
    return node_solver.step(u_phys)


class NodeCLVComputer:
    """
    Ginelli CLV algorithm for NODE surrogates.
    Mirrors lyapunov.CLVComputer but uses JVP through NODE RK4 step.
    """
    def __init__(self, node_solver, n_clv=8):
        self.solver = node_solver
        self.n_clv = n_clv
        self._step = jax.jit(lambda u: node_solver.step(u))

    def forward_pass(self, u0, n_steps):
        N = u0.shape[0]
        n_clv = self.n_clv
        step = self._step

        Q = np.eye(N, n_clv)
        u = np.array(u0, dtype=np.float64)
        Q_history = np.zeros((n_steps, N, n_clv))
        R_history = np.zeros((n_steps, n_clv, n_clv))
        log_s = np.zeros(n_clv)

        print(f"  NODE CLV forward: {n_steps} steps...")
        for i in range(n_steps):
            u_j = jnp.array(u)
            Q_raw = np.array(jax.vmap(
                lambda q: jax.jvp(step, (u_j,), (q,))[1],
                in_axes=1, out_axes=1)(jnp.array(Q)))
            u = np.array(step(u_j))
            if np.any(np.isnan(u)) or np.linalg.norm(u) > 1e6:
                print(f"    Diverged at step {i}")
                return None, None, None, None
            Q, R = np.linalg.qr(Q_raw)
            sgn = np.sign(np.diag(R))
            Q = Q * sgn[None, :]; R = R * sgn[:, None]
            Q_history[i] = Q; R_history[i] = R
            log_s += np.log(np.abs(np.diag(R)))
            if (i + 1) % max(n_steps // 4, 1) == 0:
                cur = log_s / ((i + 1) * self.solver.dt)
                print(f"    Forward {i+1}/{n_steps}: L1={cur[0]:.4f}")

        exponents = log_s / (n_steps * self.solver.dt)
        return u, Q_history, R_history, exponents

    def backward_pass(self, Q_history, R_history, seed=42):
        n_steps, N, n_clv = Q_history.shape
        rng = np.random.default_rng(seed)
        C = np.triu(rng.standard_normal((n_clv, n_clv)))
        C = C / np.linalg.norm(C, axis=0, keepdims=True)
        CLVs = np.zeros((n_steps, N, n_clv))
        print(f"  NODE CLV backward: {n_steps} steps...")
        for i in range(n_steps - 1, -1, -1):
            C = np.linalg.solve(R_history[i], C)
            norms = np.linalg.norm(C, axis=0, keepdims=True)
            C = C / (norms + 1e-14)
            CLVs[i] = Q_history[i] @ C
        return CLVs

    def compute_clv_angles(self, CLVs, pairs=None):
        n_steps, N, n_clv = CLVs.shape
        if pairs is None:
            pairs = [(i, i+1) for i in range(n_clv - 1)]
        angles = np.zeros((n_steps, len(pairs)))
        for t in range(n_steps):
            for p_idx, (i, j) in enumerate(pairs):
                vi = CLVs[t, :, i]; vj = CLVs[t, :, j]
                cos_a = np.abs(np.dot(vi, vj)) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-14)
                angles[t, p_idx] = np.degrees(np.arccos(np.clip(cos_a, 0, 1)))
        return angles


# ═══════════════════════════════════════════════════════════════════════════════
# Run CLV for each system
# ═══════════════════════════════════════════════════════════════════════════════
clv_results = {}

# ── 1. True KSE ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("System: True KSE")
print("="*55)
t0 = time.time()
clv_true = CLVComputer(solver, n_clv=N_CLV)
key_true = jax.random.PRNGKey(0)
result_true = clv_true.run(u0, n_steps=N_STEPS, key=key_true)
angles_true = result_true["angles"][N_WARMUP:]   # discard warmup
clv_results["true_kse"] = {
    "angles": angles_true,
    "exponents": result_true["exponents"],
    "runtime": time.time() - t0,
}
print(f"  True KSE: mean angles = {np.mean(angles_true, axis=0).round(1)}")
log_event("T12", "clv_complete", config={"system": "true_kse", "n_clv": N_CLV},
          metrics={"mean_angles": np.mean(angles_true, axis=0).tolist(),
                   "L1": float(result_true["exponents"][0])})


# ── 2. NODE-Std-MSE ────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("System: NODE-Std-MSE")
print("="*55)
try:
    with open("data/node_standard_mse.pkl", "rb") as f:
        node_std = pickle.load(f)
    # params structure: {"nonlinear": mlp_params}  (from init_standard_node)
    params_std = node_std["params"]
    node_solver_std = NODESolverWrapper(standard_node_rhs, params_std, N=64, dt=0.25)
    t0 = time.time()
    clv_comp_std = NodeCLVComputer(node_solver_std, n_clv=N_CLV)
    u_f, Q_h, R_h, exps = clv_comp_std.forward_pass(u0, n_steps=N_STEPS)
    if u_f is not None:
        CLVs_std = clv_comp_std.backward_pass(Q_h, R_h)
        angles_std = clv_comp_std.compute_clv_angles(CLVs_std)[N_WARMUP:]
        clv_results["node_std_mse"] = {
            "angles": angles_std,
            "exponents": exps,
            "runtime": time.time() - t0,
        }
        print(f"  NODE-Std-MSE: mean angles = {np.mean(angles_std, axis=0).round(1)}")
        log_event("T12", "clv_complete", config={"system": "node_std_mse", "n_clv": N_CLV},
                  metrics={"mean_angles": np.mean(angles_std, axis=0).tolist(),
                           "L1": float(exps[0])})
    else:
        print("  NODE-Std-MSE diverged during CLV forward pass.")
        clv_results["node_std_mse"] = None
except Exception as e:
    print(f"  NODE-Std-MSE CLV failed: {e}")
    clv_results["node_std_mse"] = None


# ── 3. NODE-Stab-MSE (use negdef variant from T4 as it's stable) ───────────────
print("\n" + "="*55)
print("System: NODE-Stab-negdef (T4 constrained-A, stable)")
print("="*55)
try:
    with open("data/constrained_a_results.pkl", "rb") as f:
        t4_results = pickle.load(f)
    negdef_params = t4_results["negdef"]["params"]
    EPS = 1e-3

    def rhs_negdef(params, u):
        B = params["B"]
        A = -(B.T @ B + EPS * jnp.eye(B.shape[0]))
        return u @ A.T + mlp_forward(params["mlp"], u)

    node_solver_neg = NODESolverWrapper(rhs_negdef, negdef_params, N=64, dt=0.25)
    t0 = time.time()
    clv_comp_neg = NodeCLVComputer(node_solver_neg, n_clv=N_CLV)
    u_f2, Q_h2, R_h2, exps2 = clv_comp_neg.forward_pass(u0, n_steps=N_STEPS)
    if u_f2 is not None:
        CLVs_neg = clv_comp_neg.backward_pass(Q_h2, R_h2)
        angles_neg = clv_comp_neg.compute_clv_angles(CLVs_neg)[N_WARMUP:]
        clv_results["node_negdef"] = {
            "angles": angles_neg,
            "exponents": exps2,
            "runtime": time.time() - t0,
        }
        print(f"  NODE-negdef: mean angles = {np.mean(angles_neg, axis=0).round(1)}")
        log_event("T12", "clv_complete", config={"system": "node_negdef", "n_clv": N_CLV},
                  metrics={"mean_angles": np.mean(angles_neg, axis=0).tolist(),
                           "L1": float(exps2[0])})
    else:
        print("  NODE-negdef diverged.")
        clv_results["node_negdef"] = None
except Exception as e:
    print(f"  NODE-negdef CLV failed: {e}")
    clv_results["node_negdef"] = None


# ── 4. Latent NODE (if T5 results available, use best d) ──────────────────────
print("\n" + "="*55)
print("System: Latent NODE (POD, best d from T5 or d=10)")
print("="*55)
try:
    from latent_node import (
        fit_pod_autoencoder, pod_encode, pod_decode,
        init_latent_ode, latent_ode_rhs, rk4_step_latent,
        kaplan_yorke as lat_ky
    )
    traj_train = np.load("data/traj_train.npy")

    # Try to load T5 results for best d; fallback to d=10
    try:
        with open("data/latent_dim_sweep.pkl", "rb") as f:
            t5 = pickle.load(f)
        # Best d = closest D_KY to truth
        best_d = min(t5.keys(), key=lambda d: abs(t5[d]["D_KY"] - 5.11)
                     if not np.isnan(t5[d]["D_KY"]) else 999)
        print(f"  Using d={best_d} (best D_KY from T5)")
        best_t5 = t5[best_d]
        # Re-train with same params (we don't store ode_params in T5 pkl by default)
        # so just use the T5 latent dim with a fresh fit
        D_LATENT = best_d
    except Exception:
        D_LATENT = 10
        print(f"  T5 not available, using d={D_LATENT}")

    # Fit POD and train latent ODE
    pod = fit_pod_autoencoder(traj_train, D_LATENT)
    encode_fn = lambda u: pod_encode(pod, u)

    from latent_node import prepare_latent_data_pod, train_latent_ode
    lat_data = prepare_latent_data_pod(traj_train, pod, solver, subsample=2)
    key_ode = jax.random.PRNGKey(42)
    ode_params = init_latent_ode(key_ode, d=D_LATENT, hidden=128, n_layers=3)
    ode_params, _ = train_latent_ode(
        ode_params, lat_data["h"], lat_data["dhdt"],
        n_epochs=300, batch_size=256, key=key_ode)

    # Latent CLV: work entirely in latent space (d-dimensional)
    # Benettin via JVP through latent RK4
    d = D_LATENT
    h0 = np.array(encode_fn(jnp.array(u0)))

    step_lat = jax.jit(lambda h: rk4_step_latent(ode_params, h, 0.25))
    Q0 = jnp.eye(d, dtype=jnp.float64)
    h0j = jnp.array(h0, dtype=jnp.float64)
    log0 = jnp.zeros(d, dtype=jnp.float64)

    n_clv_lat = min(N_CLV, d)
    Q0_lat = jnp.eye(d, n_clv_lat, dtype=jnp.float64)
    Q_hist_lat = np.zeros((N_STEPS, d, n_clv_lat))
    R_hist_lat = np.zeros((N_STEPS, n_clv_lat, n_clv_lat))
    log_lat = np.zeros(n_clv_lat)

    print(f"  Latent CLV forward: d={d}, n_clv={n_clv_lat}, steps={N_STEPS}")
    h_curr = h0j
    Q_curr = Q0_lat
    t0 = time.time()
    for i in range(N_STEPS):
        Q_raw = np.array(jax.vmap(
            lambda q: jax.jvp(step_lat, (h_curr,), (q,))[1],
            in_axes=1, out_axes=1)(Q_curr))
        h_curr = step_lat(h_curr)
        Q_n, R = np.linalg.qr(Q_raw)
        sgn = np.sign(np.diag(R))
        Q_n = Q_n * sgn[None, :]; R = R * sgn[:, None]
        Q_hist_lat[i] = Q_n; R_hist_lat[i] = R
        log_lat += np.log(np.abs(np.diag(R)))
        Q_curr = jnp.array(Q_n)

    exps_lat = log_lat / (N_STEPS * 0.25)

    # Backward pass
    rng = np.random.default_rng(0)
    C = np.triu(rng.standard_normal((n_clv_lat, n_clv_lat)))
    C /= np.linalg.norm(C, axis=0, keepdims=True)
    CLVs_lat = np.zeros((N_STEPS, d, n_clv_lat))
    for i in range(N_STEPS - 1, -1, -1):
        C = np.linalg.solve(R_hist_lat[i], C)
        C /= (np.linalg.norm(C, axis=0, keepdims=True) + 1e-14)
        CLVs_lat[i] = Q_hist_lat[i] @ C

    # Angles in latent space
    pairs_lat = [(j, j+1) for j in range(n_clv_lat - 1)]
    angles_lat = np.zeros((N_STEPS, len(pairs_lat)))
    for t in range(N_STEPS):
        for p_idx, (i_, j_) in enumerate(pairs_lat):
            vi = CLVs_lat[t, :, i_]; vj = CLVs_lat[t, :, j_]
            ca = np.abs(np.dot(vi, vj)) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-14)
            angles_lat[t, p_idx] = np.degrees(np.arccos(np.clip(ca, 0, 1)))
    angles_lat = angles_lat[N_WARMUP:]

    clv_results["latent_node"] = {
        "angles": angles_lat,
        "exponents": exps_lat,
        "d": D_LATENT,
        "runtime": time.time() - t0,
    }
    print(f"  Latent NODE: mean angles = {np.mean(angles_lat, axis=0).round(1)}")
    log_event("T12", "clv_complete",
              config={"system": "latent_node", "d": D_LATENT, "n_clv": n_clv_lat},
              metrics={"mean_angles": np.mean(angles_lat, axis=0).tolist(),
                       "L1": float(exps_lat[0])})

except Exception as e:
    print(f"  Latent NODE CLV failed: {e}")
    import traceback; traceback.print_exc()
    clv_results["latent_node"] = None


# ── Save ───────────────────────────────────────────────────────────────────────
with open("data/clv_results.pkl", "wb") as f:
    pickle.dump(clv_results, f)
print("\nSaved: data/clv_results.pkl")


# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
colors = {"true_kse": "k", "node_std_mse": "C0", "node_negdef": "C1", "latent_node": "C2"}
labels = {"true_kse": "True KSE", "node_std_mse": "NODE-Std-MSE",
          "node_negdef": "NODE-negdef", "latent_node": "Latent NODE"}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Mean angle vs CLV pair index
ax = axes[0, 0]
for name, r in clv_results.items():
    if r is not None:
        mean_a = np.mean(r["angles"], axis=0)
        ax.plot(range(1, len(mean_a)+1), mean_a, 'o-', color=colors[name],
                label=labels[name], lw=2, ms=6)
ax.set_xlabel("CLV pair (i, i+1)")
ax.set_ylabel("Mean angle (deg)")
ax.set_title("Mean CLV Angles by Pair")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.axhline(90, ls='--', color='gray', lw=0.8)

# PDF of angles for pair (0,1) — unstable-unstable
ax = axes[0, 1]
for name, r in clv_results.items():
    if r is not None and r["angles"].shape[1] > 0:
        ax.hist(r["angles"][:, 0], bins=40, density=True,
                alpha=0.5, color=colors[name], label=labels[name])
ax.set_xlabel("CLV angle: CLV1-CLV2 (deg)")
ax.set_ylabel("PDF")
ax.set_title("Angle PDF: Pair (1,2) — unstable-unstable")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Near-zero angle fraction for each pair (hyperbolicity)
ax = axes[0, 2]
NEAR_ZERO_DEG = 10
for name, r in clv_results.items():
    if r is not None:
        frac = [np.mean(r["angles"][:, p] < NEAR_ZERO_DEG)
                for p in range(r["angles"].shape[1])]
        ax.plot(range(1, len(frac)+1), frac, 'o-', color=colors[name],
                label=labels[name], lw=2, ms=6)
ax.set_xlabel("CLV pair index")
ax.set_ylabel(f"Fraction < {NEAR_ZERO_DEG}°")
ax.set_title(f"Near-Zero Angle Fraction (< {NEAR_ZERO_DEG}°)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Time series of angle for true KSE vs NODE
ax = axes[1, 0]
t_vec = np.arange(N_STEPS - N_WARMUP) * 0.25
for name in ["true_kse", "node_std_mse"]:
    if clv_results.get(name) is not None:
        ax.plot(t_vec[:500], clv_results[name]["angles"][:500, 0],
                color=colors[name], alpha=0.7, lw=1, label=labels[name])
ax.set_xlabel("t"); ax.set_ylabel("CLV angle 1-2 (deg)")
ax.set_title("CLV Angle Time Series (pair 1-2)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Lyapunov exponents comparison
ax = axes[1, 1]
for name, r in clv_results.items():
    if r is not None:
        exps = r["exponents"]
        ax.plot(range(1, len(exps)+1), exps, 'o-', color=colors[name],
                label=labels[name], lw=2, ms=5)
true_exps = le_true[:N_CLV]
ax.plot(range(1, len(true_exps)+1), true_exps, 'k^--',
        label='True KSE (full)', lw=2, ms=7)
ax.axhline(0, ls='--', color='gray', lw=0.8)
ax.set_xlabel("Index"); ax.set_ylabel("Lyapunov exponent")
ax.set_title(f"Lyapunov Spectrum (CLV pass, {N_CLV} modes)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Summary bar: mean unstable-stable boundary angle (pair n_pos, n_pos+1)
ax = axes[1, 2]
n_pos_true = int(np.sum(le_true > 0))
pair_boundary = min(n_pos_true - 1, N_CLV - 2)  # pair (n_pos-1, n_pos)
means = []
names_plot = []
for name, r in clv_results.items():
    if r is not None and r["angles"].shape[1] > pair_boundary:
        means.append(np.mean(r["angles"][:, pair_boundary]))
        names_plot.append(labels[name])
ax.bar(range(len(means)), means, color=[colors[list(clv_results.keys())[i]]
       for i in range(len(means))])
ax.set_xticks(range(len(means)))
ax.set_xticklabels(names_plot, fontsize=8, rotation=15)
ax.set_ylabel(f"Mean angle at stable/unstable boundary (°)")
ax.set_title(f"Stable-Unstable Boundary Angle (pair {pair_boundary+1}-{pair_boundary+2})")
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("T12: Covariant Lyapunov Vectors — True KSE vs Surrogates", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT12_clv_angles.png", dpi=120)
plt.close()
print("  Saved: figures/figT12_clv_angles.png")

log_event("T12", "script_complete",
          config={"n_clv": N_CLV, "n_steps": N_STEPS},
          metrics={"systems_computed": [k for k, v in clv_results.items() if v is not None]})
print("\nT12 complete.")
