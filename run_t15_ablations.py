"""
run_t15_ablations.py - Task T15: Ablation Table
================================================
Collects all completed results from prior tasks into a single comprehensive
ablation table. No new training — pure aggregation.

Table rows: every trained variant across T3, T4, T5, T6, T8, T14
Table columns: MSE/Loss | Stable | Energy | L1 | n_pos | D_KY | h_KS | notes

Also generates:
  - Summary figure comparing all variants on D_KY and h_KS
  - Pareto plot: training loss vs D_KY fidelity

Outputs:
  data/ablation_table.pkl
  figures/figT15_ablations.png
"""

import sys
sys.path = [p for p in sys.path if 'MAT6215' not in p]
sys.path.insert(0, '.')

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_log import log_event

# ── True KSE reference ────────────────────────────────────────────────────────
le_true = np.load("data/lyapunov_exponents_full.npy")


def kaplan_yorke(le):
    cs = np.cumsum(le)
    k = np.where(cs < 0)[0]
    if len(k) == 0:
        return float(len(le))
    k = k[0]
    return float(k) + (cs[k-1] if k > 0 else 0.0) / abs(le[k])


dky_true = kaplan_yorke(le_true)
h_ks_true = float(np.sum(le_true[le_true > 0]))
l1_true = float(le_true[0])

rows = []

def add_row(name, group, loss, stable, energy, L1, n_pos, D_KY, h_KS, notes=""):
    rows.append({
        "name": name,
        "group": group,
        "loss": loss,
        "stable": stable,
        "energy": energy,
        "L1": L1,
        "n_pos": n_pos,
        "D_KY": D_KY,
        "h_KS": h_KS,
        "notes": notes,
        # Derived fidelity scores (lower is better)
        "dky_err": abs(D_KY - dky_true) / dky_true if not np.isnan(D_KY) else float('nan'),
        "h_ks_err": abs(h_KS - h_ks_true) / (h_ks_true + 1e-10) if not np.isnan(h_KS) else float('nan'),
    })


# ── True KSE (reference) ──────────────────────────────────────────────────────
add_row("True KSE", "reference", loss=0.0, stable=True,
        energy=None, L1=l1_true, n_pos=int(np.sum(le_true>0)),
        D_KY=dky_true, h_KS=h_ks_true, notes="Ground truth")


# ── Original trained models (from STATUS.md) ─────────────────────────────────
# NODE-Std-MSE (undertrained, from saved artifact)
try:
    with open("data/lyap_summary_NODE_Std_MSE_full.pkl", "rb") as f:
        s = pickle.load(f)
    add_row("NODE-Std-MSE (orig)", "NODE-full", loss=0.0511, stable=True,
            energy=None, L1=s.get("L1",float('nan')), n_pos=s.get("n_pos",0),
            D_KY=s.get("D_KY",float('nan')), h_KS=s.get("h_KS",float('nan')),
            notes="100ep, hidden=64 (undertrained)")
except Exception:
    add_row("NODE-Std-MSE (orig)", "NODE-full", loss=0.0511, stable=True,
            energy=None, L1=0.0035, n_pos=31, D_KY=62.96, h_KS=0.0085,
            notes="100ep, hidden=64 (undertrained; from STATUS.md)")

# NODE-Stab-JAC (from artifact)
try:
    with open("data/lyap_summary_NODE_Stab_JAC_full.pkl", "rb") as f:
        s = pickle.load(f)
    add_row("NODE-Stab-JAC (orig)", "NODE-full", loss=730.4, stable=False,
            energy=None, L1=s.get("L1",float('nan')), n_pos=s.get("n_pos",0),
            D_KY=s.get("D_KY",float('nan')), h_KS=s.get("h_KS",float('nan')),
            notes="400ep, JAC lambda=0.01")
except Exception:
    add_row("NODE-Stab-JAC (orig)", "NODE-full", loss=730.4, stable=False,
            energy=None, L1=0.6397, n_pos=25, D_KY=52.36, h_KS=1.187,
            notes="400ep, JAC lambda=0.01 (from STATUS.md)")


# ── T3: JAC Epoch Sweep ───────────────────────────────────────────────────────
try:
    with open("data/jac_sweep_epochs.pkl", "rb") as f:
        t3e = pickle.load(f)
    for ep, ckpt in (t3e.get("checkpoints") or {}).items():
        if isinstance(ckpt, dict) and "summary" in ckpt:
            s = ckpt["summary"]
            add_row(f"NODE-JAC ep={ep}", "T3-epoch",
                    loss=ckpt.get("loss", float('nan')), stable=False,
                    energy=None, L1=s["L1"], n_pos=s["n_pos"],
                    D_KY=s["D_KY"], h_KS=s["h_KS"],
                    notes=f"JAC lam=0.01, ep={ep}")
except Exception as e:
    print(f"  T3 epoch data unavailable: {e}")

# T3: Lambda Sweep
try:
    with open("data/jac_sweep_lambda.pkl", "rb") as f:
        t3l = pickle.load(f)
    for lam, d_r in t3l.items():
        if isinstance(d_r, dict) and "summary" in d_r:
            s = d_r["summary"]
            add_row(f"NODE-JAC lam={lam}", "T3-lambda",
                    loss=d_r.get("final_loss", float('nan')), stable=False,
                    energy=None, L1=s["L1"], n_pos=s["n_pos"],
                    D_KY=s["D_KY"], h_KS=s["h_KS"],
                    notes=f"JAC lam={lam}, 600ep")
except Exception as e:
    print(f"  T3 lambda data unavailable: {e}")


# ── T4: Constrained A ─────────────────────────────────────────────────────────
try:
    with open("data/constrained_a_results.pkl", "rb") as f:
        t4 = pickle.load(f)
    for vname, vr in t4.items():
        if isinstance(vr, dict):
            lp = vr.get("lyapunov") or {}
            add_row(f"NODE-A-{vname}", "T4-constrained",
                    loss=vr.get("final_loss", float('nan')),
                    stable=vr.get("stable", False),
                    energy=vr.get("energy", float('nan')),
                    L1=lp.get("L1", float('nan')) if lp else float('nan'),
                    n_pos=lp.get("n_pos", 0) if lp else 0,
                    D_KY=lp.get("D_KY", float('nan')) if lp else float('nan'),
                    h_KS=lp.get("h_KS", float('nan')) if lp else float('nan'),
                    notes=f"MSE, 600ep, A={vname}")
except Exception as e:
    print(f"  T4 data unavailable: {e}")


# ── T5: Latent Dim Sweep ──────────────────────────────────────────────────────
try:
    with open("data/latent_dim_sweep.pkl", "rb") as f:
        t5 = pickle.load(f)
    for d, dr in t5.items():
        add_row(f"Latent-NODE-d={d}", "T5-latent-dim",
                loss=dr.get("final_ode_loss", float('nan')),
                stable=dr.get("stable", False),
                energy=dr.get("energy", float('nan')),
                L1=float(dr["lyapunov"][0]) if dr.get("lyapunov") is not None else float('nan'),
                n_pos=dr.get("n_pos", 0),
                D_KY=dr.get("D_KY", float('nan')),
                h_KS=dr.get("h_KS", float('nan')),
                notes=f"POD d={d}, 500ep")
except Exception as e:
    print(f"  T5 data unavailable: {e}")


# ── T6: Tau Sweep ─────────────────────────────────────────────────────────────
try:
    with open("data/tau_sweep_results.pkl", "rb") as f:
        t6 = pickle.load(f)
    for stride, r in t6.items():
        add_row(f"Lat-ODE-tau={r['tau']:.2f}", "T6-tau",
                loss=r.get("ode_loss", float('nan')),
                stable=r.get("ode_stable", False),
                energy=r.get("ode_energy", float('nan')),
                L1=r.get("ode_L1", float('nan')),
                n_pos=r.get("ode_n_pos", 0),
                D_KY=r.get("ode_D_KY", float('nan')),
                h_KS=r.get("ode_h_KS", float('nan')),
                notes=f"tau={r['tau']:.2f} (ODE)")
        add_row(f"Lat-Map-tau={r['tau']:.2f}", "T6-tau",
                loss=r.get("map_loss", float('nan')),
                stable=r.get("map_stable", False),
                energy=r.get("map_energy", float('nan')),
                L1=r.get("map_L1", float('nan')),
                n_pos=r.get("map_n_pos", 0),
                D_KY=r.get("map_D_KY", float('nan')),
                h_KS=r.get("map_h_KS", float('nan')),
                notes=f"tau={r['tau']:.2f} (Map)")
except Exception as e:
    print(f"  T6 data unavailable: {e}")


# ── T8: SINDy Threshold Sweep ──────────────────────────────────────────────────
try:
    with open("data/sindy_sweep_results.pkl", "rb") as f:
        t8 = pickle.load(f)
    for thresh, r in t8.get("threshold_sweep", {}).items():
        add_row(f"SINDy-thresh={thresh}", "T8-sindy",
                loss=float('nan'), stable=r.get("stable", False),
                energy=r.get("energy", float('nan')),
                L1=r.get("L1", float('nan')),
                n_pos=r.get("n_pos", 0),
                D_KY=r.get("D_KY", float('nan')),
                h_KS=r.get("h_KS", float('nan')),
                notes=f"thresh={thresh}, deg=2, r=8")
    # SINDy PI from original model (stored as known stable)
    add_row("SINDy-PI (orig)", "T8-sindy",
            loss=float('nan'), stable=True,
            energy=None,  # ~3x true energy
            L1=float('nan'), n_pos=0, D_KY=float('nan'), h_KS=float('nan'),
            notes="Physics-informed SINDy, 10 POD modes (orig)")
except Exception as e:
    print(f"  T8 data unavailable: {e}")


# ── T14: Multi-seed (best params) ─────────────────────────────────────────────
try:
    with open("data/multiseed_results.pkl", "rb") as f:
        t14 = pickle.load(f)
    for seed_run in t14.get("node_std_mse", []):
        add_row(f"NODE-Std-MSE-s{seed_run['seed']}", "T14-multiseed",
                loss=seed_run["final_loss"], stable=seed_run["stable"],
                energy=seed_run["energy"],
                L1=seed_run["L1"], n_pos=seed_run["n_pos"],
                D_KY=seed_run["D_KY"], h_KS=seed_run["h_KS"],
                notes=f"seed={seed_run['seed']}, 600ep, h=128")
    for seed_run in t14.get("latent_node_d10", []):
        add_row(f"Lat-NODE-d10-s{seed_run['seed']}", "T14-multiseed",
                loss=seed_run["final_loss"], stable=seed_run["stable"],
                energy=seed_run["energy"],
                L1=seed_run["L1"], n_pos=seed_run["n_pos"],
                D_KY=seed_run["D_KY"], h_KS=seed_run["h_KS"],
                notes=f"seed={seed_run['seed']}, d=10, 500ep")
except Exception as e:
    print(f"  T14 data unavailable: {e}")


# ── Print full table ───────────────────────────────────────────────────────────
def fmt(v, fmt_str=":.4f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  N/A "
    return format(v, fmt_str[1:])

print("\n" + "="*120)
print(f"{'Name':<30} {'Group':<18} {'Loss':>8} {'Stable':>7} "
      f"{'L1':>8} {'n_pos':>6} {'D_KY':>7} {'h_KS':>8} {'DKY_err%':>9}")
print("-"*120)
for r in rows:
    stable_s = ("Yes" if r["stable"] else "No") if r["stable"] is not None else "?"
    dky_pct = f"{100*r['dky_err']:>8.1f}%" if not np.isnan(r['dky_err']) else "    N/A "
    print(f"{r['name']:<30} {r['group']:<18} {fmt(r['loss']):>8} {stable_s:>7} "
          f"{fmt(r['L1'], ':+.4f'):>8} {r['n_pos']:>6d} "
          f"{fmt(r['D_KY']):>7} {fmt(r['h_KS']):>8} {dky_pct:>9}")
print("="*120)
print(f"\nTotal rows: {len(rows)}")


# ── Save ───────────────────────────────────────────────────────────────────────
with open("data/ablation_table.pkl", "wb") as f:
    pickle.dump({"rows": rows, "true_kse": {"L1": l1_true, "D_KY": dky_true, "h_KS": h_ks_true}}, f)
print("\nSaved: data/ablation_table.pkl")


# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
group_colors = {
    "reference": "k",
    "NODE-full": "C0",
    "T3-epoch": "C1", "T3-lambda": "C1",
    "T4-constrained": "C2",
    "T5-latent-dim": "C3",
    "T6-tau": "C4",
    "T8-sindy": "C5",
    "T14-multiseed": "C6",
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# D_KY vs Loss (Pareto plot)
ax = axes[0]
for r in rows[1:]:   # skip reference
    if not np.isnan(r["D_KY"]) and not np.isnan(r["loss"]):
        clr = group_colors.get(r["group"], "gray")
        ax.scatter(r["loss"], r["D_KY"], color=clr, s=50, alpha=0.7)
ax.axhline(dky_true, ls='--', color='k', lw=1.5, label=f'True D_KY={dky_true:.2f}')
ax.set_xlabel("Training Loss (MSE/JAC)")
ax.set_ylabel("D_KY (Kaplan-Yorke Dim)")
ax.set_title("Pareto: Loss vs Dynamical Fidelity (D_KY)")
ax.grid(True, alpha=0.3)
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=v, label=k) for k, v in group_colors.items()
              if k != "reference"]
ax.legend(handles=legend_els, fontsize=7, ncol=2)

# D_KY comparison bars (key variants only)
ax = axes[1]
key_rows = [r for r in rows if r["group"] in
            ("reference", "NODE-full", "T4-constrained", "T5-latent-dim", "T8-sindy")]
key_rows = [r for r in key_rows if not np.isnan(r["D_KY"])][:20]
names = [r["name"] for r in key_rows]
dkys = [r["D_KY"] for r in key_rows]
colors = [group_colors.get(r["group"], "gray") for r in key_rows]
bars = ax.barh(range(len(names)), dkys, color=colors)
ax.axvline(dky_true, ls='--', color='k', lw=1.5, label=f'True ({dky_true:.2f})')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("D_KY")
ax.set_title("D_KY by Variant (key groups)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='x')

# h_KS comparison
ax = axes[2]
h_rows = [r for r in rows if not np.isnan(r["h_KS"])][:20]
names_h = [r["name"] for r in h_rows]
h_ks_vals = [r["h_KS"] for r in h_rows]
colors_h = [group_colors.get(r["group"], "gray") for r in h_rows]
ax.barh(range(len(names_h)), h_ks_vals, color=colors_h)
ax.axvline(h_ks_true, ls='--', color='k', lw=1.5, label=f'True ({h_ks_true:.4f})')
ax.set_yticks(range(len(names_h)))
ax.set_yticklabels(names_h, fontsize=7)
ax.set_xlabel("h_KS (KS entropy)")
ax.set_title("KS Entropy by Variant")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='x')

plt.suptitle("T15: Ablation Table — All Variants, All Tasks", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figT15_ablations.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: figures/figT15_ablations.png")

log_event("T15", "script_complete",
          config={"n_rows": len(rows)},
          metrics={"groups": list(set(r["group"] for r in rows))})
print("\nT15 complete.")
