"""
run_t3_jac_sweep.py - Task T3: JAC Epoch + Lambda Sweep
========================================================
Trains NODE-Stab-JAC at multiple checkpoints and lambda values,
computes Lyapunov metrics after each, and plots dynamical fidelity
vs training effort.

Epoch sweep: 150, 300, 600, 1000  (at lambda=0.01)
Lambda sweep: 0.001, 0.01, 0.05, 0.1  (at 600 epochs)

Outputs:
  data/jac_sweep_epochs.pkl  -- epoch sweep results
  data/jac_sweep_lambda.pkl  -- lambda sweep results
  figures/figT3_jac_sweeps.png
  figures/figT3_jac_diagnostics.png
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
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from ks_solver import KSSolver
from neural_ode import (
    init_stabilized_node, stabilized_node_rhs,
    prepare_training_data, jacobian_matching_loss, mse_loss
)
from diagnostics import (
    spatial_power_spectrum, joint_pdf_derivatives, kl_divergence_pdf,
    temporal_autocorrelation, invariant_measure_stats, wasserstein1_marginals,
)
from experiment_log import log_event

jax.config.update("jax_enable_x64", True)

DT = 0.25
L_DOMAIN = 22.0
LYAP_SCREEN_STEPS = 500
LYAP_FINAL_STEPS = 1500
N_DIAG_WARMUP = 200
N_DIAG_STEPS = 2000
MIN_DIAG_STEPS = 512
DIAG_MAX_LAG = 200
DIAG_START_IDX = 100
TRAIN_BATCH_SIZE = 16
JAC_CACHE_VERSION = 2

# ── Lyapunov via RK4+JVP (same approach as main.py) ───────────────────────────

def rk4_step(rhs_fn, params, u, dt):
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


def compute_node_lyapunov(rhs_fn, params, u0, n_modes=20, n_steps=1500, dt=0.25):
    """Benettin QR via JVP through fixed-step RK4."""
    N = u0.shape[0]
    u0_j = jnp.array(u0, dtype=jnp.float64)
    Q0 = jnp.eye(N, n_modes, dtype=jnp.float64)
    log0 = jnp.zeros(n_modes, dtype=jnp.float64)

    step = jax.jit(lambda u: rk4_step(rhs_fn, params, u, dt))

    def benettin(carry, _):
        u, Q, ls = carry
        Q_raw = jax.vmap(lambda q: jax.jvp(step, (u,), (q,))[1],
                         in_axes=1, out_axes=1)(Q)
        u_n = step(u)
        Q_n, R = jnp.linalg.qr(Q_raw)
        s = jnp.sign(jnp.diag(R))
        Q_n = Q_n * s[None, :]
        R = R * s[:, None]
        ls_n = ls + jnp.log(jnp.abs(jnp.diag(R)))
        return (u_n, Q_n, ls_n), None

    (_, _, log_tot), _ = jax.lax.scan(benettin, (u0_j, Q0, log0), None, length=n_steps)
    return np.array(log_tot / (n_steps * dt))


def lyapunov_summary(le):
    n_pos = int(np.sum(le > 0))
    dky = kaplan_yorke(le)
    h_ks = float(np.sum(le[le > 0]))
    return {"L1": float(le[0]), "n_pos": n_pos, "D_KY": dky, "h_KS": h_ks,
            "exponents": le}


def rollout_rk4(rhs_fn, params, u0, n_steps):
    """Fixed-step RK4 rollout with a simple divergence guard."""
    step = jax.jit(lambda u: rk4_step(rhs_fn, params, u, DT))
    u = jnp.array(u0, dtype=jnp.float64)
    traj = []
    diverged_step = None

    for i in range(n_steps):
        u = step(u)
        if jnp.any(jnp.isnan(u)) or jnp.linalg.norm(u) > 1e6:
            diverged_step = i
            break
        traj.append(np.array(u))

    traj_arr = np.array(traj) if traj else np.zeros((0, u.shape[0]), dtype=np.float64)
    return traj_arr, diverged_step is None, diverged_step


def empty_diagnostics(stable, diverged_step, rollout_steps, diagnostic_steps):
    """Placeholder diagnostics when the rollout is too short to trust."""
    return {
        "stable": stable,
        "diverged_step": diverged_step,
        "rollout_steps": int(rollout_steps),
        "diagnostic_steps": int(diagnostic_steps),
        "diagnostic_complete": diagnostic_steps >= N_DIAG_STEPS,
        "W1": float("nan"),
        "KL": float("nan"),
        "power_rel_l2": float("nan"),
        "autocorr_rel_l2": float("nan"),
        "power_spectrum": None,
        "joint_pdf": None,
        "autocorr": None,
        "stats": None,
    }


def evaluate_rollout_diagnostics(rhs_fn, params, u0, traj_true_ref):
    """
    Roll out a checkpoint model and compute the T3 diagnostics promised in the
    task text: W1, KL, power spectrum, autocorrelation, and invariant stats.
    """
    n_total = N_DIAG_WARMUP + N_DIAG_STEPS
    traj_full, stable, diverged_step = rollout_rk4(rhs_fn, params, u0, n_total)

    if len(traj_full) <= N_DIAG_WARMUP:
        return empty_diagnostics(stable, diverged_step, len(traj_full), 0)

    traj = traj_full[N_DIAG_WARMUP:N_DIAG_WARMUP + N_DIAG_STEPS]
    if len(traj) < MIN_DIAG_STEPS:
        return empty_diagnostics(stable, diverged_step, len(traj_full), len(traj))

    ref = traj_true_ref[:len(traj)]
    q, E = spatial_power_spectrum(traj, L_DOMAIN)
    _, E_ref = spatial_power_spectrum(ref, L_DOMAIN)
    ux_edges, uxx_edges, pdf = joint_pdf_derivatives(traj, L_DOMAIN)
    _, _, pdf_ref = joint_pdf_derivatives(ref, L_DOMAIN)
    max_lag = min(DIAG_MAX_LAG, len(traj) - 1)
    autocorr = temporal_autocorrelation(traj, max_lag=max_lag)
    autocorr_ref = temporal_autocorrelation(ref, max_lag=max_lag)
    stats = invariant_measure_stats(traj, label="NODE-Stab-JAC checkpoint")

    denom_E = np.linalg.norm(E_ref) + 1e-12
    denom_ac = np.linalg.norm(autocorr_ref) + 1e-12

    return {
        "stable": stable,
        "diverged_step": diverged_step,
        "rollout_steps": int(len(traj_full)),
        "diagnostic_steps": int(len(traj)),
        "diagnostic_complete": len(traj) >= N_DIAG_STEPS,
        "W1": float(wasserstein1_marginals(ref, traj)),
        "KL": float(kl_divergence_pdf(pdf_ref, pdf)),
        "power_rel_l2": float(np.linalg.norm(E - E_ref) / denom_E),
        "autocorr_rel_l2": float(np.linalg.norm(autocorr - autocorr_ref) / denom_ac),
        "power_spectrum": (q, E),
        "joint_pdf": (ux_edges, uxx_edges, pdf),
        "autocorr": autocorr,
        "stats": stats,
    }


def diagnostics_log_metrics(diag):
    """Flatten diagnostics into JSON-safe scalar metrics for experiment_log."""
    stats = diag["stats"] or {}
    return {
        "rollout_stable": bool(diag["stable"]),
        "rollout_steps": int(diag["rollout_steps"]),
        "diagnostic_steps": int(diag["diagnostic_steps"]),
        "diagnostic_complete": bool(diag["diagnostic_complete"]),
        "W1": float(diag["W1"]),
        "KL": float(diag["KL"]),
        "power_rel_l2": float(diag["power_rel_l2"]),
        "autocorr_rel_l2": float(diag["autocorr_rel_l2"]),
        "diag_energy": float(stats.get("energy", float("nan"))),
        "diag_rms": float(stats.get("rms", float("nan"))),
        "diag_skewness": float(stats.get("skewness", float("nan"))),
    }


def load_pickle_if_available(path):
    """Best-effort pickle loader for resume support."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def has_rollout_diagnostics(record):
    """Check whether a saved checkpoint/lambda entry is from the refreshed T3 format."""
    return (
        isinstance(record, dict)
        and isinstance(record.get("diagnostics"), dict)
        and "W1" in record["diagnostics"]
        and "KL" in record["diagnostics"]
    )


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
traj_train = np.load("data/traj_train.npy")
traj_analysis = np.load("data/traj_analysis.npy")
le_true = np.load("data/lyapunov_exponents_full.npy")

solver = KSSolver(L=22.0, N=64, dt=0.25)
u0_lyap = traj_analysis[500].astype(np.float64)
u0_diag = traj_analysis[DIAG_START_IDX].astype(np.float64)
traj_true_diag = traj_analysis[
    DIAG_START_IDX + N_DIAG_WARMUP + 1:
    DIAG_START_IDX + N_DIAG_WARMUP + 1 + N_DIAG_STEPS
].astype(np.float64)
true_diag_stats = invariant_measure_stats(traj_true_diag, label="True KSE")

print(f"True KSE: L1={le_true[0]:.4f}, n_pos={int(np.sum(le_true>0))}, "
      f"D_KY={kaplan_yorke(le_true):.2f}, h_KS={float(np.sum(le_true[le_true>0])):.4f}")
print(f"True diagnostic reference: {len(traj_true_diag)} steps, "
      f"energy={true_diag_stats['energy']:.4f}, rms={true_diag_stats['rms']:.4f}")

# ── Prepare JAC training data ──────────────────────────────────────────────────
print("\nPreparing JAC training data (cached)...")
data = prepare_training_data(
    traj_train, solver,
    compute_jacobians=True,
    subsample=10,
    cache_path="data/jac_training_cache_v2.npz"  # corrected Jacobian cache
)
u_data = np.asarray(data["u"], dtype=np.float64)
rhs_data = np.asarray(data["rhs"], dtype=np.float64)
jac_data = np.asarray(data["jacobians"], dtype=np.float64)
print(f"  {len(u_data)} points, Jacobians shape: {jac_data.shape}")


# ── Generic training function ──────────────────────────────────────────────────
def train_jac(params, n_epochs, lam, batch_size=TRAIN_BATCH_SIZE, lr_init=1e-3, lr_final=1e-5,
              seed=42, checkpoint_every=None):
    """Train stabilized NODE with JAC loss. Returns (params, loss_history, checkpoints)."""
    n_data = len(u_data)
    lr_sched = optax.exponential_decay(lr_init, transition_steps=n_epochs,
                                       decay_rate=lr_final/lr_init)
    optimizer = optax.adam(lr_sched)
    opt_state = optimizer.init(params)

    @partial(jax.jit, static_argnums=(3,))
    def train_step(params, opt_state, u_b, rhs_b, jac_b, optimizer, lam_):
        def loss_fn(p):
            return jacobian_matching_loss(stabilized_node_rhs, p, u_b, rhs_b, jac_b, lam_)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss

    # Pre-JIT with static optimizer
    @jax.jit
    def train_step_jit(params, opt_state, u_b, rhs_b, jac_b):
        def loss_fn(p):
            return jacobian_matching_loss(stabilized_node_rhs, p, u_b, rhs_b, jac_b, lam)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss

    rng = np.random.default_rng(seed)
    loss_history = []
    checkpoints = {}
    train_start = time.time()

    for epoch in range(n_epochs):
        idx = rng.permutation(n_data)
        epoch_losses = []
        for s in range(0, n_data - batch_size + 1, batch_size):
            b = idx[s:s+batch_size]
            params, opt_state, loss = train_step_jit(
                params, opt_state, u_data[b], rhs_data[b], jac_data[b]
            )
            epoch_losses.append(float(loss))

        ml = np.mean(epoch_losses)
        loss_history.append(ml)

        if checkpoint_every and (epoch+1) in checkpoint_every:
            ckpt_start = time.time()
            print(f"  Checkpoint epoch {epoch+1}: loss={ml:.2f} -- computing Lyapunov ({LYAP_SCREEN_STEPS} steps, screen)...")
            le = compute_node_lyapunov(stabilized_node_rhs, params, u0_lyap,
                                       n_modes=20, n_steps=LYAP_SCREEN_STEPS, dt=DT)
            # Escalate to 1500 steps if any exponent is near zero (|le| < 0.02)
            # near-zero exponents converge slowly and n_pos can flip
            near_zero = np.any(np.abs(le) < 0.02)
            is_final = (epoch+1) == max(checkpoint_every)
            if near_zero or is_final:
                reason = "final checkpoint" if is_final else "near-zero exponents detected"
                print(f"    Escalating to {LYAP_FINAL_STEPS} steps ({reason})...")
                le = compute_node_lyapunov(stabilized_node_rhs, params, u0_lyap,
                                           n_modes=20, n_steps=LYAP_FINAL_STEPS, dt=DT)
            summary = lyapunov_summary(le)
            print(f"    L1={summary['L1']:+.4f}, n_pos={summary['n_pos']}, "
                  f"D_KY={summary['D_KY']:.2f}, h_KS={summary['h_KS']:.4f}")
            print(f"    Computing rollout diagnostics ({N_DIAG_STEPS} kept steps after "
                  f"{N_DIAG_WARMUP} warmup)...")
            diagnostics = evaluate_rollout_diagnostics(
                stabilized_node_rhs, params, u0_diag, traj_true_diag
            )
            checkpoints[epoch+1] = {
                "params": params,
                "summary": summary,
                "diagnostics": diagnostics,
                "loss": ml,
                "epoch": epoch+1,
                "lam": lam,
            }
            if np.isfinite(diagnostics["W1"]):
                print(f"    W1={diagnostics['W1']:.4f}, KL={diagnostics['KL']:.4f}, "
                      f"power_rel_l2={diagnostics['power_rel_l2']:.4f}, "
                      f"autocorr_rel_l2={diagnostics['autocorr_rel_l2']:.4f}, "
                      f"stable={diagnostics['stable']}")
            else:
                print(f"    Diagnostics incomplete: rollout_steps={diagnostics['rollout_steps']}, "
                      f"diagnostic_steps={diagnostics['diagnostic_steps']}, "
                      f"stable={diagnostics['stable']}")
            log_event(
                "T3",
                "epoch_checkpoint",
                config={
                    "lam": lam,
                    "epoch": epoch + 1,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "seed": seed,
                },
                metrics={
                    "loss": ml,
                    **summary,
                    **diagnostics_log_metrics(diagnostics),
                },
                timings={
                    "checkpoint_wall_s": time.time() - ckpt_start,
                    "elapsed_train_wall_s": time.time() - train_start,
                },
            )

        if (epoch+1) % max(n_epochs//5, 1) == 0 or epoch == n_epochs-1:
            print(f"  Epoch {epoch+1:5d}/{n_epochs}  loss={ml:.2f}")

    return params, loss_history, checkpoints, (time.time() - train_start)


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP 1: Epoch sweep at fixed lambda=0.01
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SWEEP 1: Epoch sweep (lambda=0.01)")
print("="*60)

EPOCH_CHECKPOINTS = [150, 300, 600, 1000]
epoch_sweep_path = "data/jac_sweep_epochs.pkl"
existing_epoch_sweep = load_pickle_if_available(epoch_sweep_path)
ckpts_ep = {}
loss_hist_ep = []

if (
    isinstance(existing_epoch_sweep, dict)
    and existing_epoch_sweep.get("jacobian_cache_version") == JAC_CACHE_VERSION
    and all(
        ep in existing_epoch_sweep.get("checkpoints", {})
        and has_rollout_diagnostics(existing_epoch_sweep["checkpoints"][ep])
        for ep in EPOCH_CHECKPOINTS
    )
):
    ckpts_ep = existing_epoch_sweep["checkpoints"]
    loss_hist_ep = existing_epoch_sweep.get("loss_history", [])
    print("Using refreshed epoch sweep from disk.")
else:
    key_init = jax.random.PRNGKey(7)
    params_ep = init_stabilized_node(key_init, N=64, hidden=128, n_layers=3)

    epoch_sweep_start = time.time()
    params_ep, loss_hist_ep, ckpts_ep, epoch_sweep_train_time = train_jac(
        params_ep, n_epochs=max(EPOCH_CHECKPOINTS), lam=0.01, batch_size=TRAIN_BATCH_SIZE,
        checkpoint_every=set(EPOCH_CHECKPOINTS)
    )

    # Compile epoch sweep results
    epoch_sweep = {"checkpoints": ckpts_ep, "loss_history": loss_hist_ep,
                   "lambda": 0.01, "epochs": EPOCH_CHECKPOINTS,
                   "jacobian_cache_version": JAC_CACHE_VERSION,
                   "diagnostic_config": {
                       "warmup_steps": N_DIAG_WARMUP,
                       "diagnostic_steps": N_DIAG_STEPS,
                       "min_required_steps": MIN_DIAG_STEPS,
                       "autocorr_max_lag": DIAG_MAX_LAG,
                       "reference_start_idx": DIAG_START_IDX,
                   }}
    with open(epoch_sweep_path, "wb") as f:
        pickle.dump(epoch_sweep, f)
    print("Saved: data/jac_sweep_epochs.pkl")
    log_event(
        "T3",
        "epoch_sweep_complete",
        config={
            "epochs": EPOCH_CHECKPOINTS,
            "lambda": 0.01,
            "batch_size": TRAIN_BATCH_SIZE,
            "subsample": 10,
            "n_train_points": int(len(u_data)),
        },
        metrics={
            "final_loss": float(loss_hist_ep[-1]),
            "checkpoints_completed": [ep for ep in EPOCH_CHECKPOINTS if ep in ckpts_ep],
        },
        timings={
            "train_wall_s": epoch_sweep_train_time,
            "total_wall_s": time.time() - epoch_sweep_start,
        },
    )

# Print table
print("\n" + "="*65)
print(f"{'Epoch':>7} {'Loss':>10} {'L1':>8} {'n_pos':>6} {'D_KY':>7} {'h_KS':>8}")
print("-"*65)
true_summary = {"L1": le_true[0], "n_pos": int(np.sum(le_true>0)),
                "D_KY": kaplan_yorke(le_true), "h_KS": float(np.sum(le_true[le_true>0]))}
print(f"{'TRUE':>7} {'':>10} {true_summary['L1']:>+8.4f} {true_summary['n_pos']:>6d} "
      f"{true_summary['D_KY']:>7.2f} {true_summary['h_KS']:>8.4f}")
for ep in EPOCH_CHECKPOINTS:
    if ep in ckpts_ep:
        c = ckpts_ep[ep]
        s = c["summary"]
        print(f"{ep:>7d} {c['loss']:>10.2f} {s['L1']:>+8.4f} {s['n_pos']:>6d} "
              f"{s['D_KY']:>7.2f} {s['h_KS']:>8.4f}")
print("="*65)

print("\n" + "="*95)
print(f"{'Epoch':>7} {'Stable':>7} {'W1':>8} {'KL':>8} {'PowErr':>9} {'ACErr':>9} {'Energy':>9}")
print("-"*95)
print(f"{'TRUE':>7} {'Yes':>7} {0.0:>8.4f} {0.0:>8.4f} {0.0:>9.4f} {0.0:>9.4f} "
      f"{true_diag_stats['energy']:>9.4f}")
for ep in EPOCH_CHECKPOINTS:
    if ep in ckpts_ep:
        d_r = ckpts_ep[ep]["diagnostics"]
        stats = d_r["stats"] or {}
        print(f"{ep:>7d} {str(d_r['stable']):>7} {d_r['W1']:>8.4f} {d_r['KL']:>8.4f} "
              f"{d_r['power_rel_l2']:>9.4f} {d_r['autocorr_rel_l2']:>9.4f} "
              f"{float(stats.get('energy', float('nan'))):>9.4f}")
print("="*95)


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP 2: Lambda sweep at 600 epochs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SWEEP 2: Lambda sweep (600 epochs each)")
print("="*60)

LAMBDAS = [0.001, 0.01, 0.05, 0.1]
lambda_sweep_path = "data/jac_sweep_lambda.pkl"
existing_lambda_sweep = load_pickle_if_available(lambda_sweep_path)
if isinstance(existing_lambda_sweep, dict):
    lambda_sweep = {
        lam: record for lam, record in existing_lambda_sweep.items()
        if lam in LAMBDAS and has_rollout_diagnostics(record)
    } if existing_lambda_sweep.get("jacobian_cache_version") == JAC_CACHE_VERSION else {}
    if lambda_sweep:
        print(f"Resuming lambda sweep with completed lambdas: {sorted(lambda_sweep.keys())}")
else:
    lambda_sweep = {}

for lam in LAMBDAS:
    if lam in lambda_sweep:
        print(f"\n  Lambda={lam}: already complete, skipping.")
        continue
    lam_start = time.time()
    print(f"\n  Lambda={lam}:")
    key_lam = jax.random.PRNGKey(8 + int(lam * 1000))
    params_lam = init_stabilized_node(key_lam, N=64, hidden=128, n_layers=3)
    params_lam, loss_hist_lam, ckpts_lam, train_wall_s = train_jac(
        params_lam, n_epochs=600, lam=lam, batch_size=TRAIN_BATCH_SIZE,
        checkpoint_every={600}  # 500-step Lyapunov (it's the only checkpoint so is_final=True -> 1500 steps)
    )
    final = ckpts_lam.get(600)
    if final:
        s = final["summary"]
        lambda_sweep[lam] = {
            "params": params_lam,
            "loss_history": loss_hist_lam,
            "summary": s,
            "diagnostics": final["diagnostics"],
            "final_loss": loss_hist_lam[-1],
        }
        print(f"  lam={lam}: L1={s['L1']:+.4f}, n_pos={s['n_pos']}, "
              f"D_KY={s['D_KY']:.2f}, h_KS={s['h_KS']:.4f}")
        log_event(
            "T3",
            "lambda_run_complete",
            config={
                "lam": lam,
                "epochs": 600,
                "batch_size": TRAIN_BATCH_SIZE,
                "subsample": 10,
                "n_train_points": int(len(u_data)),
            },
            metrics={
                "final_loss": float(loss_hist_lam[-1]),
                **s,
                **diagnostics_log_metrics(final["diagnostics"]),
            },
            timings={
                "train_wall_s": train_wall_s,
                "total_wall_s": time.time() - lam_start,
            },
        )
        with open(lambda_sweep_path, "wb") as f:
            pickle.dump({
                **lambda_sweep,
                "jacobian_cache_version": JAC_CACHE_VERSION,
            }, f)
        print(f"  Saved partial: {lambda_sweep_path}")

with open(lambda_sweep_path, "wb") as f:
    pickle.dump({
        **lambda_sweep,
        "jacobian_cache_version": JAC_CACHE_VERSION,
    }, f)
print("\nSaved: data/jac_sweep_lambda.pkl")

# Print lambda table
print("\n" + "="*65)
print(f"{'Lambda':>8} {'Loss':>10} {'L1':>8} {'n_pos':>6} {'D_KY':>7} {'h_KS':>8}")
print("-"*65)
print(f"{'TRUE':>8} {'':>10} {true_summary['L1']:>+8.4f} {true_summary['n_pos']:>6d} "
      f"{true_summary['D_KY']:>7.2f} {true_summary['h_KS']:>8.4f}")
for lam in LAMBDAS:
    if lam in lambda_sweep:
        d_r = lambda_sweep[lam]
        s = d_r["summary"]
        print(f"{lam:>8.3f} {d_r['final_loss']:>10.2f} {s['L1']:>+8.4f} {s['n_pos']:>6d} "
              f"{s['D_KY']:>7.2f} {s['h_KS']:>8.4f}")
print("="*65)

print("\n" + "="*98)
print(f"{'Lambda':>8} {'Stable':>7} {'W1':>8} {'KL':>8} {'PowErr':>9} {'ACErr':>9} {'Energy':>9}")
print("-"*98)
print(f"{'TRUE':>8} {'Yes':>7} {0.0:>8.4f} {0.0:>8.4f} {0.0:>9.4f} {0.0:>9.4f} "
      f"{true_diag_stats['energy']:>9.4f}")
for lam in LAMBDAS:
    if lam in lambda_sweep:
        d_r = lambda_sweep[lam]["diagnostics"]
        stats = d_r["stats"] or {}
        print(f"{lam:>8.3f} {str(d_r['stable']):>7} {d_r['W1']:>8.4f} {d_r['KL']:>8.4f} "
              f"{d_r['power_rel_l2']:>9.4f} {d_r['autocorr_rel_l2']:>9.4f} "
              f"{float(stats.get('energy', float('nan'))):>9.4f}")
print("="*98)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# --- Epoch sweep figures ---
ax = axes[0, 0]
ax.semilogy(loss_hist_ep, color='C0', lw=1.5)
for ep in EPOCH_CHECKPOINTS:
    if ep in ckpts_ep:
        ax.axvline(ep-1, ls='--', color='gray', lw=0.8)
        ax.text(ep, ax.get_ylim()[0]*2, str(ep), fontsize=7, ha='center')
ax.set_xlabel("Epoch")
ax.set_ylabel("JAC Loss")
ax.set_title("Epoch Sweep: Training Loss (lam=0.01)")
ax.grid(True, alpha=0.3)

metrics = ['L1', 'D_KY', 'h_KS']
ylabels = ['L1 (leading LE)', 'D_KY (KY dim)', 'h_KS (KS entropy)']
true_vals = [true_summary['L1'], true_summary['D_KY'], true_summary['h_KS']]

for col, (metric, ylabel, true_val) in enumerate(zip(metrics, ylabels, true_vals)):
    ax = axes[0, col+0] if col == 0 else axes[0, col]
    # Epoch sweep plot
    epochs_done = [ep for ep in EPOCH_CHECKPOINTS if ep in ckpts_ep]
    vals = [ckpts_ep[ep]['summary'][metric] for ep in epochs_done]
    ax2 = axes[1, col]
    ax2.plot(epochs_done, vals, 'o-', color='C0', lw=2, ms=7, label='JAC (lam=0.01)')
    ax2.axhline(true_val, ls='--', color='k', lw=1.5, label='True KSE')
    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"Epoch Sweep: {ylabel}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

# Lambda sweep
for col, (metric, ylabel) in enumerate(zip(metrics[:2], ylabels[:2])):
    ax = axes[0, col+1]
    lams_done = [l for l in LAMBDAS if l in lambda_sweep]
    vals = [lambda_sweep[l]['summary'][metric] for l in lams_done]
    ax.semilogx(lams_done, vals, 's-', color='C1', lw=2, ms=8)
    ax.axhline(true_vals[col], ls='--', color='k', lw=1.5)
    ax.set_xlabel("Lambda (JAC weight)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Lambda Sweep: {ylabel} (600 epochs)")
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.92, "-- True KSE", transform=ax.transAxes, fontsize=8)

plt.suptitle("T3: JAC Training Sweeps — Dynamical Fidelity", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("figures/figT3_jac_sweeps.png", dpi=120)
plt.close()
print("  Saved: figures/figT3_jac_sweeps.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

epoch_diag_metrics = [
    ("W1", "W1 vs true", 0.0),
    ("KL", "KL joint PDF", 0.0),
    ("power_rel_l2", "Power rel L2", 0.0),
]

for col, (metric, ylabel, true_val) in enumerate(epoch_diag_metrics):
    ax = axes[0, col]
    epochs_done = [ep for ep in EPOCH_CHECKPOINTS if ep in ckpts_ep]
    vals = [ckpts_ep[ep]["diagnostics"][metric] for ep in epochs_done]
    ax.plot(epochs_done, vals, 'o-', color='C2', lw=2, ms=7)
    ax.axhline(true_val, ls='--', color='k', lw=1.5)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Epoch Sweep: {ylabel}")
    ax.grid(True, alpha=0.3)

lambda_diag_metrics = [
    ("W1", "W1 vs true", 0.0),
    ("KL", "KL joint PDF", 0.0),
    ("autocorr_rel_l2", "Autocorr rel L2", 0.0),
]

for col, (metric, ylabel, true_val) in enumerate(lambda_diag_metrics):
    ax = axes[1, col]
    lams_done = [lam for lam in LAMBDAS if lam in lambda_sweep]
    vals = [lambda_sweep[lam]["diagnostics"][metric] for lam in lams_done]
    ax.semilogx(lams_done, vals, 's-', color='C3', lw=2, ms=8)
    ax.axhline(true_val, ls='--', color='k', lw=1.5)
    ax.set_xlabel("Lambda (JAC weight)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Lambda Sweep: {ylabel}")
    ax.grid(True, alpha=0.3)

plt.suptitle("T3: JAC Training Sweeps - Rollout Diagnostics", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("figures/figT3_jac_diagnostics.png", dpi=120)
plt.close()
print("  Saved: figures/figT3_jac_diagnostics.png")

print("\nT3 complete.")
log_event(
    "T3",
    "script_complete",
    config={
        "epoch_checkpoints": EPOCH_CHECKPOINTS,
        "lambdas": LAMBDAS,
    },
    metrics={
        "epoch_sweep_saved": True,
        "lambda_sweep_saved": True,
    },
)
