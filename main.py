"""
main.py -- Full Pipeline Orchestrator for KSE Surrogate Dynamical Fidelity Study
==================================================================================
Runs the complete analysis pipeline:
1. Load pre-generated trajectories and Lyapunov data
2. Roll out all surrogates on test ICs
3. Compute full diagnostics on each system
4. Compute Lyapunov spectra for each surrogate
5. Compute CLV angles (true KSE only, due to cost)
6. Generate all figures
7. Save results table

All data cached to disk -- safe to re-run from any checkpoint.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial

jax.config.update("jax_enable_x64", True)

from ks_solver import KSSolver
from lyapunov import (compute_lyapunov_spectrum_jit, lyapunov_summary,
                       CLVComputer, kaplan_yorke_dimension, ks_entropy)
from diagnostics import (spatial_power_spectrum, joint_pdf_derivatives,
                          temporal_autocorrelation, marginal_pdf,
                          invariant_measure_stats, wasserstein1_marginals,
                          kl_divergence_pdf, ensemble_error)
from neural_ode import (standard_node_rhs, stabilized_node_rhs,
                         rollout_node, node_lyapunov_rhs)
from sindy import SINDyModel

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ===========================================================================
# Helpers
# ===========================================================================

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def rollout_sindy(sindy_model, u0_phys, n_steps):
    """Roll out SINDy model from physical-space IC."""
    try:
        traj = sindy_model.integrate(u0_phys, n_steps=n_steps,
                                      dt=sindy_model.dt if hasattr(sindy_model, 'dt') else 0.25)
        if traj is None or np.any(np.isnan(traj)):
            return None
        return traj
    except Exception as e:
        print(f"  SINDy rollout failed: {e}")
        return None


def rollout_node_model(result, u0_phys, n_steps):
    """Roll out NODE model from physical-space IC."""
    node_type = result["metadata"]["node_type"]
    rhs_fn = stabilized_node_rhs if node_type == "stabilized" else standard_node_rhs
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), result["params"])
    try:
        traj = np.array(rollout_node(rhs_fn, params, jnp.array(u0_phys),
                                      n_steps=n_steps, dt=0.25))
        if np.any(np.isnan(traj)) or np.any(np.abs(traj) > 1e4):
            return None
        return traj
    except Exception as e:
        print(f"  NODE rollout failed: {e}")
        return None


# ===========================================================================
# Lyapunov spectrum for surrogate (NODE)
# ===========================================================================

def compute_node_lyapunov(result, u0_phys, n_steps=2000, n_modes=20):
    """Compute Lyapunov spectrum for a NODE surrogate using a fake solver wrapper."""
    node_type = result["metadata"]["node_type"]
    rhs_fn = stabilized_node_rhs if node_type == "stabilized" else standard_node_rhs
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), result["params"])

    # Wrap NODE as a solver-like object with a step() that uses diffrax integration
    from neural_ode import integrate_node
    from functools import partial

    class FakeSolver:
        dt = 0.25
        def step(self, u_phys):
            # u_phys is physical-space array here
            return jnp.array(integrate_node(rhs_fn, params, u_phys, 0.0, self.dt, dt0=self.dt/4))

    # But compute_lyapunov_spectrum_jit expects physical-space input/output
    # Use _physical_discrete_step directly
    from lyapunov import _physical_discrete_step
    fake_solver = FakeSolver()

    # Override _physical_discrete_step to use NODE step
    def node_phys_step(u_phys):
        return integrate_node(rhs_fn, params, jnp.array(u_phys), 0.0, 0.25, dt0=0.25/4)

    # Inline the JIT scan for NODE
    N = u0_phys.shape[0]
    Q0 = jnp.eye(N, n_modes)
    u0_jax = jnp.array(u0_phys, dtype=jnp.float64)
    log0 = jnp.zeros(n_modes)

    def step_fn(carry, _):
        u, Q, log_sum = carry
        Q_raw = jax.vmap(
            lambda q: jax.jvp(node_phys_step, (u,), (q,))[1],
            in_axes=1, out_axes=1
        )(Q)
        u_next = node_phys_step(u)
        Q_next, R = jnp.linalg.qr(Q_raw)
        signs = jnp.sign(jnp.diag(R))
        Q_next = Q_next * signs[None, :]
        R = R * signs[:, None]
        log_next = log_sum + jnp.log(jnp.abs(jnp.diag(R)))
        return (u_next, Q_next, log_next), jnp.diag(R)

    print(f"  Computing NODE Lyapunov spectrum (JIT compile + {n_steps} steps)...")
    (u_final, Q_final, log_total), _ = jax.lax.scan(
        step_fn, (u0_jax, Q0, log0), None, length=n_steps
    )
    exponents = np.array(log_total / (n_steps * 0.25))
    return exponents


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    solver = KSSolver(L=22.0, N=64, dt=0.25)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading data...")
    traj_analysis = np.load("data/traj_analysis.npy")
    traj_train = np.load("data/traj_train.npy")
    lyap_true = np.load("data/lyapunov_exponents.npy")
    with open("data/lyapunov_summary.pkl", "rb") as f:
        lyap_summary_true = pickle.load(f)

    u0_test = jnp.array(traj_analysis[0])   # physical space IC for rollouts
    print(f"  Analysis traj: {traj_analysis.shape}")
    print(f"  True L1={lyap_true[0]:.4f}, KY={lyap_summary_true['kaplan_yorke_dim']:.2f}")

    # -----------------------------------------------------------------------
    # 2. Load models
    # -----------------------------------------------------------------------
    print("\nLoading models...")
    models = {}
    model_files = {
        "NODE-Std-MSE":  "data/node_standard_mse.pkl",
        "NODE-Stab-MSE": "data/node_stabilized_mse.pkl",
        "NODE-Stab-JAC": "data/node_stabilized_jac.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = load_model(path)
            print(f"  Loaded {name}: best_loss={models[name]['metadata']['best_loss']:.6f}")
        else:
            print(f"  Missing: {path}")

    sindy_loaded = None
    if os.path.exists("data/sindy_model.pkl"):
        with open("data/sindy_model.pkl", "rb") as f:
            sindy_loaded = pickle.load(f)
        print("  Loaded SINDy model")

    # -----------------------------------------------------------------------
    # 3. Generate trajectories for all surrogates
    # -----------------------------------------------------------------------
    print("\nGenerating surrogate trajectories (n=4000 steps)...")
    n_traj = 4000
    trajs = {"True KSE": traj_analysis[:n_traj]}

    for name, result in models.items():
        print(f"  Rolling out {name}...")
        traj = rollout_node_model(result, u0_test, n_traj)
        if traj is not None:
            trajs[name] = traj
            print(f"    OK: range=[{traj.min():.2f}, {traj.max():.2f}]")
        else:
            print(f"    FAILED (diverged or NaN)")

    if sindy_loaded is not None:
        print("  Rolling out SINDy...")
        traj_sindy = rollout_sindy(sindy_loaded, np.array(u0_test), n_traj)
        if traj_sindy is not None:
            trajs["SINDy"] = traj_sindy
            print(f"    OK: range=[{traj_sindy.min():.2f}, {traj_sindy.max():.2f}]")
        else:
            print("    FAILED")

    # -----------------------------------------------------------------------
    # 4. Diagnostics on all systems
    # -----------------------------------------------------------------------
    print("\nComputing diagnostics...")
    diag_results = {}
    for name, traj in trajs.items():
        print(f"  {name}:")
        q, E = spatial_power_spectrum(traj, L=22.0)
        ux_e, uxx_e, jpdf = joint_pdf_derivatives(traj, L=22.0)
        autocorr = temporal_autocorrelation(traj, max_lag=200)
        stats = invariant_measure_stats(traj, label=name)
        centers, mpdf = marginal_pdf(traj)
        diag_results[name] = {
            "power_spectrum": (q, E),
            "joint_pdf": (ux_e, uxx_e, jpdf),
            "autocorr": autocorr,
            "stats": stats,
            "marginal_pdf": (centers, mpdf),
        }
        print(f"    energy={stats['energy']:.4f}, rms={stats['rms']:.4f}, "
              f"skew={stats['skewness']:.3f}")

    # Wasserstein and KL vs true
    ref_traj = trajs["True KSE"]
    for name in list(trajs.keys())[1:]:
        w1 = wasserstein1_marginals(ref_traj, trajs[name])
        _, _, jpdf_ref = diag_results["True KSE"]["joint_pdf"]
        _, _, jpdf_approx = diag_results[name]["joint_pdf"]
        kl = kl_divergence_pdf(jpdf_ref, jpdf_approx)
        diag_results[name]["W1"] = w1
        diag_results[name]["KL"] = kl
        print(f"  {name}: W1={w1:.4f}, KL={kl:.4f}")

    with open("data/diagnostics.pkl", "wb") as f:
        pickle.dump(diag_results, f)

    # -----------------------------------------------------------------------
    # 5. Lyapunov spectra for NODE surrogates
    # -----------------------------------------------------------------------
    print("\nComputing Lyapunov spectra for surrogates...")
    lyap_results = {"True KSE": {"exponents": lyap_true, **lyap_summary_true}}

    for name, result in models.items():
        cache_path = f"data/lyap_{name.replace(' ', '_').replace('-', '_')}.npy"
        if os.path.exists(cache_path):
            exp = np.load(cache_path)
            print(f"  {name}: loaded from cache, L1={exp[0]:.4f}")
        else:
            print(f"  {name}: computing...")
            exp = compute_node_lyapunov(result, np.array(u0_test),
                                         n_steps=2000, n_modes=20)
            np.save(cache_path, exp)
            print(f"  {name}: L1={exp[0]:.4f}")
        summary = lyapunov_summary(exp, label=name)
        lyap_results[name] = summary

    with open("data/lyapunov_all.pkl", "wb") as f:
        pickle.dump(lyap_results, f)

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'System':<22} {'L1':>8} {'TL':>8} {'n_pos':>6} {'D_KY':>8} {'h_KS':>8}")
    print("=" * 70)
    for name, res in lyap_results.items():
        print(f"{name:<22} {res['lambda_1']:>8.4f} {res['lyapunov_time']:>8.2f} "
              f"{res['n_positive']:>6d} {res['kaplan_yorke_dim']:>8.2f} "
              f"{res['ks_entropy']:>8.4f}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 6. Generate figures
    # -----------------------------------------------------------------------
    print("\nGenerating figures...")
    generate_all_figures(trajs, diag_results, lyap_results, solver)

    print("\nPipeline complete. All results in data/ and figures/")


# ===========================================================================
# Figure generation
# ===========================================================================

def generate_all_figures(trajs, diag_results, lyap_results, solver):
    colors = {
        "True KSE":     "#1f77b4",
        "NODE-Std-MSE": "#ff7f0e",
        "NODE-Stab-MSE":"#2ca02c",
        "NODE-Stab-JAC":"#d62728",
        "SINDy":        "#9467bd",
    }
    ls_map = {
        "True KSE": "-",
        "NODE-Std-MSE": "--",
        "NODE-Stab-MSE": "-.",
        "NODE-Stab-JAC": ":",
        "SINDy": (0, (3, 1, 1, 1)),
    }

    # --- Fig 1: Space-time plots ---
    fig, axes = plt.subplots(len(trajs), 1, figsize=(12, 2.5 * len(trajs)))
    if len(trajs) == 1:
        axes = [axes]
    t_show = 500
    x = np.linspace(0, 22, 64, endpoint=False)
    t = np.arange(t_show) * 0.25
    for ax, (name, traj) in zip(axes, trajs.items()):
        n = min(t_show, traj.shape[0])
        im = ax.pcolormesh(x, t[:n], traj[:n], cmap="RdBu_r",
                           vmin=-3, vmax=3, shading="auto")
        ax.set_ylabel("t")
        ax.set_title(name)
        plt.colorbar(im, ax=ax)
    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig("figures/fig1_spacetime.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved fig1_spacetime.png")

    # --- Fig 2: Power spectrum (Linot 2022 style) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, res in diag_results.items():
        q, E = res["power_spectrum"]
        ax.semilogy(q[1:], E[1:], label=name,
                    color=colors.get(name, "gray"),
                    ls=ls_map.get(name, "-"), lw=1.8)
    ax.set_xlabel("Wavenumber q")
    ax.set_ylabel("E(q)")
    ax.set_title("Spatial Power Spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig2_power_spectrum.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved fig2_power_spectrum.png")

    # --- Fig 3: Joint PDF of (u_x, u_xx) --- Linot 2022 Fig 8 ---
    n_sys = len(diag_results)
    fig, axes = plt.subplots(1, n_sys, figsize=(4 * n_sys, 4))
    if n_sys == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, diag_results.items()):
        ux_e, uxx_e, pdf = res["joint_pdf"]
        ax.pcolormesh(ux_e, uxx_e, pdf.T, cmap="hot_r", shading="auto")
        ax.set_xlabel("u_x")
        ax.set_ylabel("u_xx")
        ax.set_title(name)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-8, 8)
    plt.suptitle("Joint PDF of Derivatives (Linot 2022 Fig 8)", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/fig3_joint_pdf.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved fig3_joint_pdf.png")

    # --- Fig 4: Marginal PDF ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in diag_results.items():
        centers, mpdf = res["marginal_pdf"]
        ax.plot(centers, mpdf, label=name,
                color=colors.get(name, "gray"),
                ls=ls_map.get(name, "-"), lw=1.8)
    ax.set_xlabel("u")
    ax.set_ylabel("p(u)")
    ax.set_title("Marginal PDF of u")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig4_marginal_pdf.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved fig4_marginal_pdf.png")

    # --- Fig 5: Temporal autocorrelation ---
    fig, ax = plt.subplots(figsize=(7, 4))
    lags = np.arange(200) * 0.25
    for name, res in diag_results.items():
        ac = res["autocorr"]
        ax.plot(lags, ac, label=name,
                color=colors.get(name, "gray"),
                ls=ls_map.get(name, "-"), lw=1.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Lag (time units)")
    ax.set_ylabel("C(tau)")
    ax.set_title("Temporal Autocorrelation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig5_autocorr.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved fig5_autocorr.png")

    # --- Fig 6: Lyapunov spectrum comparison (Park 2024 / Ozalp 2024 style) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in lyap_results.items():
        exp = res["exponents"]
        ax.plot(np.arange(1, len(exp) + 1), exp,
                label=f"{name} (KY={res['kaplan_yorke_dim']:.1f})",
                color=colors.get(name, "gray"),
                ls=ls_map.get(name, "-"),
                marker="o", markersize=4, lw=1.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Index i")
    ax.set_ylabel("Lyapunov exponent L_i")
    ax.set_title("Lyapunov Spectrum Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 21)
    plt.tight_layout()
    plt.savefig("figures/fig6_lyapunov_spectrum.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved fig6_lyapunov_spectrum.png")

    # --- Fig 7: Training loss curves (if available) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, result in [(n, r) for n, r in [
        ("NODE-Std-MSE", "data/node_standard_mse.pkl"),
        ("NODE-Stab-MSE", "data/node_stabilized_mse.pkl"),
        ("NODE-Stab-JAC", "data/node_stabilized_jac.pkl"),
    ] if os.path.exists(r)]:
        with open(result, "rb") as f:
            res = pickle.load(f)
        loss = res["loss_history"]
        ax.semilogy(loss, label=name, color=colors.get(name, "gray"),
                    ls=ls_map.get(name, "-"), lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig7_training_loss.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved fig7_training_loss.png")

    # --- Fig 8: Summary comparison bar chart ---
    names_all = list(lyap_results.keys())
    metrics = ["kaplan_yorke_dim", "ks_entropy", "lambda_1"]
    metric_labels = ["KY Dim", "KS Entropy", "L1"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    true_vals = {m: lyap_results["True KSE"][m] for m in metrics}

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        vals = [lyap_results[n][metric] for n in names_all]
        bar_colors = [colors.get(n, "gray") for n in names_all]
        bars = ax.bar(range(len(names_all)), vals, color=bar_colors, alpha=0.85)
        ax.axhline(true_vals[metric], color="k", lw=1.5, ls="--", label="True KSE")
        ax.set_xticks(range(len(names_all)))
        ax.set_xticklabels([n.replace("NODE-", "").replace("Stab-", "S-")
                            for n in names_all], rotation=30, ha="right", fontsize=8)
        ax.set_title(mlabel)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Dynamical Invariants: True KSE vs Surrogates", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/fig8_summary_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved fig8_summary_comparison.png")

    print("All figures saved to figures/")


if __name__ == "__main__":
    main()
