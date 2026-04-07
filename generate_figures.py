"""generate_figures.py -- Generate all analysis figures."""
import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from ks_solver import KSSolver
from neural_ode import standard_node_rhs, stabilized_node_rhs, rollout_node
from lyapunov import lyapunov_summary
from diagnostics import (spatial_power_spectrum, joint_pdf_derivatives,
                          temporal_autocorrelation, marginal_pdf,
                          invariant_measure_stats, wasserstein1_marginals,
                          kl_divergence_pdf)

os.makedirs("figures", exist_ok=True)
solver = KSSolver(L=22.0, N=64, dt=0.25)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# Load trajectories and Lyapunov data
# ------------------------------------------------------------------
traj_true = np.load("data/traj_analysis.npy")[:4000]
lyap_true = np.load("data/lyapunov_exponents.npy")
lsum_true = load_pkl("data/lyapunov_summary.pkl")

# Load NODE models
node_std  = load_pkl("data/node_standard_mse.pkl")
node_stab = load_pkl("data/node_stabilized_mse.pkl")
params_std  = jax.tree_util.tree_map(lambda x: jnp.array(x), node_std["params"])
params_stab = jax.tree_util.tree_map(lambda x: jnp.array(x), node_stab["params"])

# Rollout from attractor IC
u0 = jnp.array(traj_true[0])

print("Rolling out Standard NODE (4000 steps)...")
traj_std = np.array(rollout_node(standard_node_rhs, params_std, u0, 4000, dt=0.25))
std_ok = not np.any(np.isnan(traj_std)) and np.max(np.abs(traj_std)) < 200
print(f"  OK={std_ok}, range=[{traj_std.min():.2f},{traj_std.max():.2f}]")

print("Rolling out Stabilized NODE (4000 steps)...")
try:
    traj_stab = np.array(rollout_node(stabilized_node_rhs, params_stab, u0, 4000, dt=0.25))
    stab_ok = not np.any(np.isnan(traj_stab)) and np.max(np.abs(traj_stab)) < 200
except Exception:
    stab_ok = False
    traj_stab = None
print(f"  OK={stab_ok}")

# Build trajectory dict
trajs = {"True KSE": traj_true}
if std_ok:
    trajs["NODE-Std-MSE"] = traj_std
if stab_ok:
    trajs["NODE-Stab-MSE"] = traj_stab

# SINDy
if os.path.exists("data/sindy_model.pkl"):
    sindy = load_pkl("data/sindy_model.pkl")
    try:
        dt_s = getattr(sindy, "dt", 0.25)
        ts = sindy.integrate(np.array(u0), n_steps=4000, dt=dt_s)
        if ts is not None and not np.any(np.isnan(ts)) and np.max(np.abs(ts)) < 200:
            trajs["SINDy"] = ts
            print("SINDy rollout OK")
        else:
            print("SINDy diverged")
    except Exception as e:
        print(f"SINDy error: {e}")

# Lyapunov data for all systems
lyap_exps = {"True KSE": lyap_true}
lyap_sums = {"True KSE": lsum_true}
for name in ["NODE-Std-MSE", "NODE-Stab-MSE", "NODE-Stab-JAC"]:
    key = name.replace("-", "_")
    epath = f"data/lyap_{key}.npy"
    spath = f"data/lyap_summary_{key}.pkl"
    if os.path.exists(epath) and os.path.exists(spath):
        lyap_exps[name] = np.load(epath)
        lyap_sums[name] = load_pkl(spath)
        print(f"Loaded Lyapunov for {name}: L1={lyap_exps[name][0]:.4f}")

# ------------------------------------------------------------------
# Diagnostics (only on non-diverging long trajs)
# ------------------------------------------------------------------
print("\nComputing diagnostics...")
diag = {}
for name, traj in trajs.items():
    q, E = spatial_power_spectrum(traj, L=22.0)
    ux_e, uxx_e, jpdf = joint_pdf_derivatives(traj, L=22.0)
    ac = temporal_autocorrelation(traj, max_lag=200)
    stats = invariant_measure_stats(traj, label=name)
    centers, mpdf = marginal_pdf(traj)
    diag[name] = dict(q=q, E=E, ux_e=ux_e, uxx_e=uxx_e, jpdf=jpdf,
                      ac=ac, stats=stats, centers=centers, mpdf=mpdf)
    print(f"  {name}: energy={stats['energy']:.4f}, rms={stats['rms']:.4f}")

ref = traj_true
for name in list(trajs.keys())[1:]:
    w1 = wasserstein1_marginals(ref, trajs[name])
    kl = kl_divergence_pdf(diag["True KSE"]["jpdf"], diag[name]["jpdf"])
    diag[name]["W1"] = w1
    diag[name]["KL"] = kl
    print(f"  {name}: W1={w1:.4f}, KL={kl:.4f}")

# ------------------------------------------------------------------
# Color / style scheme
# ------------------------------------------------------------------
colors = {
    "True KSE":      "#1f77b4",
    "NODE-Std-MSE":  "#ff7f0e",
    "NODE-Stab-MSE": "#2ca02c",
    "NODE-Stab-JAC": "#d62728",
    "SINDy":         "#9467bd",
}
lsmap = {
    "True KSE":      "-",
    "NODE-Std-MSE":  "--",
    "NODE-Stab-MSE": "-.",
    "NODE-Stab-JAC": ":",
    "SINDy":         (0, (3, 1, 1, 1)),
}

# ------------------------------------------------------------------
# FIG 1: Space-time plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(len(trajs), 1, figsize=(12, 2.8 * len(trajs)))
if len(trajs) == 1:
    axes = [axes]
x_grid = np.linspace(0, 22, 64, endpoint=False)
t_grid = np.arange(500) * 0.25
for ax, (name, traj) in zip(axes, trajs.items()):
    n = min(500, traj.shape[0])
    im = ax.pcolormesh(x_grid, t_grid[:n], traj[:n],
                       cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
    ax.set_ylabel("t")
    ax.set_title(name, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85)
axes[-1].set_xlabel("x")
plt.suptitle("KSE Space-Time: True vs Surrogates", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig1_spacetime.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig1_spacetime.png")

# ------------------------------------------------------------------
# FIG 2: Power spectrum
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for name, d in diag.items():
    ax.semilogy(d["q"][1:], d["E"][1:], label=name,
                color=colors.get(name, "gray"), ls=lsmap.get(name, "-"), lw=2)
ax.set_xlabel("Wavenumber q")
ax.set_ylabel("E(q)")
ax.set_title("Time-Averaged Spatial Power Spectrum (Linot 2022)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig2_power_spectrum.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig2_power_spectrum.png")

# ------------------------------------------------------------------
# FIG 3: Joint PDF (u_x, u_xx) — Linot 2022 Fig 8
# ------------------------------------------------------------------
n_sys = len(diag)
fig, axes = plt.subplots(1, n_sys, figsize=(4.5 * n_sys, 4))
if n_sys == 1:
    axes = [axes]
for ax, (name, d) in zip(axes, diag.items()):
    im = ax.pcolormesh(d["ux_e"], d["uxx_e"], d["jpdf"].T,
                       cmap="hot_r", shading="auto")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("u_x")
    ax.set_ylabel("u_xx")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-8, 8)
    plt.colorbar(im, ax=ax, shrink=0.85)
plt.suptitle("Joint PDF (u_x, u_xx) -- Linot 2022 Key Diagnostic", y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig("figures/fig3_joint_pdf.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig3_joint_pdf.png")

# ------------------------------------------------------------------
# FIG 4: Marginal PDF + Autocorrelation
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for name, d in diag.items():
    ax1.plot(d["centers"], d["mpdf"], label=name,
             color=colors.get(name, "gray"), ls=lsmap.get(name, "-"), lw=2)
ax1.set_xlabel("u")
ax1.set_ylabel("p(u)")
ax1.set_title("Marginal PDF of u")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

lags = np.arange(200) * 0.25
for name, d in diag.items():
    ax2.plot(lags, d["ac"], label=name,
             color=colors.get(name, "gray"), ls=lsmap.get(name, "-"), lw=2)
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel("Lag (time units)")
ax2.set_ylabel("C(tau)")
ax2.set_title("Temporal Autocorrelation")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig4_pdf_autocorr.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig4_pdf_autocorr.png")

# ------------------------------------------------------------------
# FIG 5: Lyapunov spectrum — KEY FIGURE (Park 2024 / Ozalp 2024)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for name, exp in lyap_exps.items():
    res = lyap_sums[name]
    lbl = (f"{name}  "
           f"(KY={res['kaplan_yorke_dim']:.1f}, "
           f"L1={res['lambda_1']:.4f})")
    ax.plot(np.arange(1, len(exp) + 1), exp,
            label=lbl,
            color=colors.get(name, "gray"),
            ls=lsmap.get(name, "-"),
            marker="o", markersize=4, lw=2)
ax.axhline(0, color="k", lw=1.2, ls="--")
ax.set_xlabel("Index i")
ax.set_ylabel("Lyapunov exponent L_i")
ax.set_title("Lyapunov Spectrum: True KSE vs Surrogates\n"
             "(MSE-trained NODEs recover wrong Lyapunov structure)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0.5, 20.5)
plt.tight_layout()
plt.savefig("figures/fig5_lyapunov_spectrum.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig5_lyapunov_spectrum.png")

# ------------------------------------------------------------------
# FIG 6: Summary bar chart
# ------------------------------------------------------------------
names_b = list(lyap_sums.keys())
ky_vals  = [lyap_sums[n]["kaplan_yorke_dim"] for n in names_b]
hks_vals = [lyap_sums[n]["ks_entropy"] for n in names_b]
l1_vals  = [lyap_sums[n]["lambda_1"] for n in names_b]
bc       = [colors.get(n, "gray") for n in names_b]
xticks   = [n.replace("NODE-", "").replace("Stab-", "S-") for n in names_b]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
true_vals = [lsum_true["kaplan_yorke_dim"],
             lsum_true["ks_entropy"],
             lsum_true["lambda_1"]]
for ax, vals, lbl, tv in zip(
    axes,
    [ky_vals, hks_vals, l1_vals],
    ["KY Dimension", "KS Entropy (h_KS)", "Leading LE (L1)"],
    true_vals
):
    ax.bar(range(len(names_b)), vals, color=bc, alpha=0.85, edgecolor="k", lw=0.5)
    ax.axhline(tv, color="k", lw=2, ls="--", label="True KSE")
    ax.set_xticks(range(len(names_b)))
    ax.set_xticklabels(xticks, rotation=30, ha="right", fontsize=8)
    ax.set_title(lbl)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
plt.suptitle("Dynamical Invariants: Trajectory Accuracy != Dynamical Fidelity",
             fontsize=10)
plt.tight_layout()
plt.savefig("figures/fig6_summary_bars.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig6_summary_bars.png")

# ------------------------------------------------------------------
# FIG 7: Training loss curves
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for name, pkl in [("NODE-Std-MSE",  "data/node_standard_mse.pkl"),
                  ("NODE-Stab-MSE", "data/node_stabilized_mse.pkl"),
                  ("NODE-Stab-JAC", "data/node_stabilized_jac.pkl")]:
    if os.path.exists(pkl):
        r = load_pkl(pkl)
        ax.semilogy(r["loss_history"], label=name,
                    color=colors.get(name, "gray"),
                    ls=lsmap.get(name, "-"), lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Neural ODE Training Convergence")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig7_training_loss.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig7_training_loss.png")

# ------------------------------------------------------------------
# FIG 8: Wasserstein + KL table as figure
# ------------------------------------------------------------------
rows = []
for name in list(trajs.keys())[1:]:
    w1 = diag[name].get("W1", float("nan"))
    kl = diag[name].get("KL", float("nan"))
    rows.append([name, f"{w1:.4f}", f"{kl:.4f}"])

fig, ax = plt.subplots(figsize=(7, 2 + 0.5 * len(rows)))
ax.axis("off")
table = ax.table(
    cellText=rows,
    colLabels=["System", "W1 (vs True KSE)", "KL (joint PDF)"],
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)
plt.title("Wasserstein Distance and KL Divergence from True KSE", pad=20)
plt.tight_layout()
plt.savefig("figures/fig8_distance_table.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig8_distance_table.png")

print("\nAll figures generated.")
