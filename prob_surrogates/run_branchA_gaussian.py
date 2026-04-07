"""
run_branchA_gaussian.py — Branch A: Diagonal Gaussian Transition Model
=======================================================================
Learns p(u_{t+Δt} | u_t) = N(μ_θ(u_t), diag(σ_θ(u_t)²))
in full physical space (R^64), trained by negative log-likelihood.

This is the foundational probabilistic model.  All subsequent branches
(E: Laplace, C: SDE) build on or compare to this one.

Architecture:
  Single MLP: 64 → 256 → 256 → 256 → 128
  Split output into (μ, log_σ) ∈ R^64 each.
  log_σ is clamped to [-6, 2] to prevent collapse/explosion.

Training:
  Loss = NLL = mean over batch of [log σ_i + (u'_i - μ_i)²/(2σ_i²)]
  Optional Jacobian regularization: + λ * ||J||_F (same spirit as parent project)
  Optimizer: Adam with exponential LR decay 1e-3 → 1e-4

Outputs:
  prob_surrogates/data/branchA_results.pkl
  prob_surrogates/figures/figA_*.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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

from prob_diagnostics import (
    nll_gaussian, crps_gaussian, calibration_curve, calibration_error,
    coverage, ensemble_rmse_spread, spread_skill_ratio,
    wasserstein1_empirical, print_prob_summary,
    rank_histogram, ensemble_energy_score,
)

jax.config.update("jax_enable_x64", True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_STATE   = 64
HIDDEN    = 256
N_LAYERS  = 3
N_EPOCHS  = 600
BATCH     = 512
LR_INIT   = 1e-3
LR_FINAL  = 1e-4
JAC_LAMBDA = 0.0    # set > 0 to enable Jacobian regularization (e.g. 1e-3)
N_ENSEMBLE = 50     # members for rollout evaluation
N_ROLLOUT  = 2000   # steps for long-time rollout
DT         = 0.25

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR  = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("Branch A: Diagonal Gaussian Transition Model")
print(f"  Architecture: {N_STATE} → {HIDDEN}x{N_LAYERS} → {2*N_STATE}")
print(f"  Epochs: {N_EPOCHS}, batch: {BATCH}, λ_Jac: {JAC_LAMBDA}")


# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────

def init_gauss_net(key, n=N_STATE, hidden=HIDDEN, n_layers=N_LAYERS):
    """MLP outputting (μ, log_σ) concatenated → shape (2n,)."""
    sizes = [n] + [hidden] * n_layers + [2 * n]
    params = []
    for i in range(len(sizes) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        W = jax.random.normal(k1, (sizes[i], sizes[i+1])) * np.sqrt(2.0 / sizes[i])
        b = jnp.zeros(sizes[i+1])
        params.append({"W": W, "b": b})
    return params


def gauss_forward(params, u):
    """Forward pass. Returns (mean, log_std) each of shape (n,)."""
    x = u
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])
    out = x @ params[-1]["W"] + params[-1]["b"]
    mean    = out[:N_STATE]
    log_std = jnp.clip(out[N_STATE:], -6.0, 2.0)
    return mean, log_std


@jax.jit
def nll_loss_batch(params, u_t_batch, u_next_batch):
    """Mean NLL over a batch."""
    def single(u_t, u_next):
        mean, log_std = gauss_forward(params, u_t)
        std = jnp.exp(log_std)
        return jnp.mean(0.5 * ((u_next - mean) / std) ** 2 + log_std
                        + 0.5 * jnp.log(2 * jnp.pi))
    return jnp.mean(jax.vmap(single)(u_t_batch, u_next_batch))


@jax.jit
def nll_jac_loss_batch(params, u_t_batch, u_next_batch, lam=JAC_LAMBDA):
    """NLL + Frobenius Jacobian regularization on the mean head."""
    nll = nll_loss_batch(params, u_t_batch, u_next_batch)
    if lam == 0.0:
        return nll
    def mean_fn(u):
        m, _ = gauss_forward(params, u)
        return m
    J = jax.vmap(jax.jacobian(mean_fn))(u_t_batch)   # (B, n, n)
    jac_reg = jnp.mean(jnp.sum(J ** 2, axis=(1, 2)))
    return nll + lam * jac_reg


@jax.jit
def update(params, opt_state, u_t_batch, u_next_batch):
    loss, grads = jax.value_and_grad(nll_jac_loss_batch)(
        params, u_t_batch, u_next_batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# ── Sample a single next state ─────────────────────────────────────────────────
def sample_next(params, u_t, key):
    mean, log_std = gauss_forward(params, u_t)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, shape=mean.shape, dtype=jnp.float64)
    return mean + std * eps


# ── Ensemble rollout ────────────────────────────────────────────────────────────
def ensemble_rollout(params, u0, n_steps, n_members, key):
    """
    Roll out n_members stochastic trajectories from u0.
    Returns array of shape (n_members, n_steps, N_STATE).
    Samples collapse to deterministic mean if σ → 0.
    """
    keys = jax.random.split(key, n_members)

    def single_rollout(rng):
        u = jnp.array(u0, dtype=jnp.float64)
        traj = []
        for _ in range(n_steps):
            rng, k = jax.random.split(rng)
            u = sample_next(params, u, k)
            if jnp.any(jnp.isnan(u)) or jnp.linalg.norm(u) > 1e6:
                u = jnp.zeros_like(u)   # mark diverged member as zeros
            traj.append(u)
        return jnp.stack(traj)           # (n_steps, N_STATE)

    # vmap over members is clean but requires fixed-length rollout; use Python loop
    # for safety (divergence detection)
    all_trajs = []
    for m in range(n_members):
        traj = single_rollout(keys[m])
        all_trajs.append(np.array(traj))
    return np.stack(all_trajs)           # (n_members, n_steps, N_STATE)


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")
traj_train = np.load(os.path.join(DATA_DIR, "traj_train.npy")).astype(np.float64)
traj_test  = np.load(os.path.join(DATA_DIR, "traj_analysis.npy")).astype(np.float64)

# Consecutive pairs
U_t    = traj_train[:-1]    # (N-1, 64) conditioning states
U_next = traj_train[1:]     # (N-1, 64) next states

# Validation pairs (first 2000 steps of analysis trajectory)
U_val_t    = traj_test[:1999]
U_val_next = traj_test[1:2000]

# Test trajectory for rollout evaluation
u0_test = traj_test[100].copy()
true_traj_test = traj_test[100:100 + N_ROLLOUT].copy()

n_train = len(U_t)
print(f"  Train pairs: {n_train}, val pairs: {len(U_val_t)}")
print(f"  State energy: {float(np.mean(np.sum(traj_train**2, axis=1))):.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nTraining Branch A ({N_EPOCHS} epochs)...")
key = jax.random.PRNGKey(42)
params = init_gauss_net(key)

schedule = optax.exponential_decay(
    LR_INIT, transition_steps=n_train // BATCH * N_EPOCHS // 4,
    decay_rate=LR_FINAL / LR_INIT)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(params)

train_losses = []
val_losses   = []
t0 = time.time()

for epoch in range(N_EPOCHS):
    perm = np.random.permutation(n_train)
    epoch_loss = 0.0
    n_batches  = 0
    for i in range(0, n_train - BATCH, BATCH):
        idx = perm[i:i + BATCH]
        params, opt_state, loss = update(
            params, opt_state,
            jnp.array(U_t[idx]), jnp.array(U_next[idx]))
        epoch_loss += float(loss)
        n_batches  += 1
    avg = epoch_loss / max(n_batches, 1)
    train_losses.append(avg)

    if (epoch + 1) % 50 == 0:
        # Validation NLL (no grad)
        val_loss = float(nll_loss_batch(
            params,
            jnp.array(U_val_t[:BATCH]),
            jnp.array(U_val_next[:BATCH])))
        val_losses.append((epoch + 1, val_loss))
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:>4}/{N_EPOCHS}: train_NLL={avg:.5f}  "
              f"val_NLL={val_loss:.5f}  ({elapsed:.0f}s)")

print(f"Training complete in {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# One-step evaluation on validation set
# ──────────────────────────────────────────────────────────────────────────────
print("\nOne-step evaluation on validation set...")

pred_means_val = np.array(jax.vmap(lambda u: gauss_forward(params, u)[0])(
    jnp.array(U_val_t)))
pred_logs_val  = np.array(jax.vmap(lambda u: gauss_forward(params, u)[1])(
    jnp.array(U_val_t)))
pred_stds_val  = np.exp(pred_logs_val)
residuals_val  = U_val_next - pred_means_val

nll_val    = nll_gaussian(pred_means_val, pred_stds_val, U_val_next)
crps_val   = crps_gaussian(pred_means_val, pred_stds_val, U_val_next)
cal_err    = calibration_error(pred_stds_val, residuals_val)
mean_std   = float(np.mean(pred_stds_val))
rmse_det   = float(np.sqrt(np.mean(residuals_val**2)))

print(f"  NLL      = {nll_val:.5f}")
print(f"  CRPS     = {crps_val:.5f}")
print(f"  Cal.err  = {cal_err:.5f}  (0=perfect)")
print(f"  Mean σ   = {mean_std:.5f}")
print(f"  Det RMSE = {rmse_det:.5f}  (mean prediction error)")


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble rollout evaluation
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nEnsemble rollout: {N_ENSEMBLE} members × {N_ROLLOUT} steps...")
key_roll = jax.random.PRNGKey(123)
t0 = time.time()
ensemble = ensemble_rollout(params, u0_test, N_ROLLOUT, N_ENSEMBLE, key_roll)
print(f"  Rollout done in {time.time()-t0:.1f}s")

# Check for diverged members (marked as zeros after norm > 1e6)
n_diverged = int(np.sum(np.all(ensemble[:, -1, :] == 0, axis=-1)))
print(f"  Diverged members: {n_diverged}/{N_ENSEMBLE}")

# Trim to valid length
valid_T = N_ROLLOUT
rmse_t, spread_t = ensemble_rmse_spread(ensemble, true_traj_test[:valid_T])
ssr = spread_skill_ratio(ensemble, true_traj_test[:valid_T])
cov_dict = coverage(ensemble, true_traj_test[:valid_T], levels=[0.5, 0.9, 0.95])

print(f"  Spread/skill ratio : {ssr:.3f}  (ideal=1.0)")
print(f"  Coverage 50%  : {cov_dict[0.5]:.3f}  (ideal=0.50)")
print(f"  Coverage 90%  : {cov_dict[0.9]:.3f}  (ideal=0.90)")
print(f"  Coverage 95%  : {cov_dict[0.95]:.3f}  (ideal=0.95)")

# Long-time: compare invariant statistics
ensemble_flat = ensemble[:, 500:, :].reshape(-1, N_STATE)
swd = wasserstein1_empirical(ensemble_flat, true_traj_test[500:], n_proj=100)
print(f"  Sliced W1 distance (long-time): {swd:.5f}")

# Energy evolution
ensemble_energy = np.mean(np.sum(ensemble**2, axis=-1), axis=0)   # (T,)
true_energy     = np.sum(true_traj_test**2, axis=-1)               # (T,)
print(f"  True mean energy  : {float(np.mean(true_energy[500:])):.2f}")
print(f"  Ensemble mean energy: {float(np.mean(ensemble_energy[500:])):.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty analysis: σ(u) varies along attractor?
# ──────────────────────────────────────────────────────────────────────────────
print("\nAnalyzing state-dependent uncertainty σ(u) along attractor...")
# Evaluate σ at attractor points from true trajectory
n_eval = min(2000, len(traj_test))
pred_stds_attractor = np.array(jax.vmap(
    lambda u: jnp.exp(gauss_forward(params, u)[1]))(
    jnp.array(traj_test[:n_eval])))    # (n_eval, 64)

mean_sigma_by_wavenumber = np.mean(pred_stds_attractor, axis=0)  # (64,)
std_sigma_by_wavenumber  = np.std(pred_stds_attractor, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────────
results = {
    "params": params,
    "train_losses": train_losses,
    "val_losses": val_losses,
    # One-step metrics
    "nll_val": nll_val,
    "crps_val": crps_val,
    "cal_error": cal_err,
    "mean_sigma": mean_std,
    "det_rmse": rmse_det,
    # Ensemble metrics
    "spread_skill_ratio": ssr,
    "coverage": cov_dict,
    "swd_longtime": swd,
    "n_diverged": n_diverged,
    # Ensemble trajectory (for Branch E comparison)
    "ensemble_traj": ensemble,
    "true_traj_test": true_traj_test,
    # Uncertainty profile
    "mean_sigma_by_wavenum": mean_sigma_by_wavenumber,
    "std_sigma_by_wavenum": std_sigma_by_wavenumber,
    # Calibration arrays for plotting
    "pred_stds_val": pred_stds_val,
    "residuals_val": residuals_val,
}

out_path = os.path.join(OUT_DIR, "branchA_results.pkl")
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
tau_L = 22.0   # Lyapunov time
t_axis = np.arange(N_ROLLOUT) * DT

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Training curve
ax = axes[0, 0]
ax.semilogy(train_losses, 'C0', lw=1.5, label='train NLL')
if val_losses:
    ep_v, l_v = zip(*val_losses)
    ax.semilogy(ep_v, l_v, 'C1o--', ms=5, label='val NLL')
ax.set_xlabel("Epoch"); ax.set_ylabel("NLL"); ax.set_title("Training Curve")
ax.legend(); ax.grid(True, alpha=0.3)

# 2. Calibration curve
ax = axes[0, 1]
exp_cov, obs_cov = calibration_curve(pred_stds_val.ravel(), residuals_val.ravel())
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')
ax.plot(exp_cov, obs_cov, 'C0-o', ms=4, lw=1.5, label=f'Branch A (err={cal_err:.3f})')
ax.set_xlabel("Expected coverage"); ax.set_ylabel("Observed coverage")
ax.set_title("Calibration Reliability Diagram")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 3. Spread vs RMSE over time
ax = axes[0, 2]
ax.plot(t_axis / tau_L, rmse_t,   'C0', lw=1.5, label='Ensemble RMSE')
ax.plot(t_axis / tau_L, spread_t, 'C1--', lw=1.5, label='Ensemble spread')
ax.set_xlabel("Time / τ_L"); ax.set_ylabel("Error / spread")
ax.set_title(f"Spread vs RMSE (ratio={ssr:.2f})")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0, min(10, t_axis[-1] / tau_L))

# 4. Rank histogram
ax = axes[1, 0]
counts, edges = rank_histogram(
    ensemble[:, :200, :4], true_traj_test[:200, :4])
centers = 0.5 * (edges[:-1] + edges[1:])
ax.bar(centers, counts, width=np.diff(edges)[0], color='C0', alpha=0.7)
flat_level = 1.0 / len(counts)
ax.axhline(flat_level, ls='--', color='k', lw=1, label='ideal flat')
ax.set_xlabel("Rank"); ax.set_ylabel("Frequency")
ax.set_title("Rank Histogram (first 4 dims, 200 steps)")
ax.legend(); ax.grid(True, alpha=0.3)

# 5. Predicted σ profile across state dimensions
ax = axes[1, 1]
ax.plot(range(N_STATE), mean_sigma_by_wavenumber, 'C0', lw=1.5, label='mean σ(u)')
ax.fill_between(range(N_STATE),
    mean_sigma_by_wavenumber - std_sigma_by_wavenumber,
    mean_sigma_by_wavenumber + std_sigma_by_wavenumber,
    alpha=0.3, color='C0', label='±1 std over attractor')
ax.set_xlabel("State dimension (wavenumber)"); ax.set_ylabel("Predicted σ")
ax.set_title("State-dependent Uncertainty Profile")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 6. Energy evolution
ax = axes[1, 2]
ens_energy_q25 = np.percentile(np.sum(ensemble**2, axis=-1), 25, axis=0)
ens_energy_q75 = np.percentile(np.sum(ensemble**2, axis=-1), 75, axis=0)
ax.plot(t_axis / tau_L, true_energy, 'k', lw=1.5, label='True KSE')
ax.plot(t_axis / tau_L, ensemble_energy, 'C0', lw=1.5, label='Ensemble mean')
ax.fill_between(t_axis / tau_L, ens_energy_q25, ens_energy_q75,
                alpha=0.3, color='C0', label='IQR')
ax.set_xlabel("Time / τ_L"); ax.set_ylabel(r"$\|u\|^2$")
ax.set_title("Energy Evolution")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle("Branch A: Diagonal Gaussian Transition Model", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figA_summary.png"), dpi=120)
plt.close()
print("  Saved: figA_summary.png")

# Spacetime plot of ensemble mean vs truth
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
T_show = min(200, N_ROLLOUT)
x = np.linspace(0, 22, N_STATE)
t_show = np.arange(T_show) * DT

ens_mean_traj = np.mean(ensemble, axis=0)
ens_std_traj  = np.std(ensemble, axis=0)

vmax = float(np.percentile(np.abs(true_traj_test[:T_show]), 98))
axes[0].pcolormesh(x, t_show, true_traj_test[:T_show],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
axes[0].set_title("True KSE"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

axes[1].pcolormesh(x, t_show, ens_mean_traj[:T_show],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
axes[1].set_title("Ensemble Mean"); axes[1].set_xlabel("x")

im = axes[2].pcolormesh(x, t_show, ens_std_traj[:T_show],
                        cmap='hot_r', shading='auto')
plt.colorbar(im, ax=axes[2])
axes[2].set_title("Ensemble Std (uncertainty)"); axes[2].set_xlabel("x")

plt.suptitle("Branch A: Spacetime — True vs Ensemble Mean vs Uncertainty",
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figA_spacetime.png"), dpi=120)
plt.close()
print("  Saved: figA_spacetime.png")

print("\nBranch A complete.")
print(f"  NLL={nll_val:.4f}  CRPS={crps_val:.4f}  "
      f"Cal.err={cal_err:.4f}  SSR={ssr:.3f}  SWD={swd:.4f}")
