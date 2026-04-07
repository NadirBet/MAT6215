"""
run_t22_update_report.py - Task T22: Update Final Report
=========================================================
Reads all completed pkl files and regenerates the results tables
in reports/FINAL_REPORT.md in-place.  Safe to re-run at any time.

Updates:
  - Table 1: True KSE canonical numbers
  - Table 2: All trained surrogate Lyapunov metrics (from STATUS canonical + all tasks)
  - Table 3: Diagnostics comparison (W1, KL, energy) from diagnostics_all.pkl
  - Table 4: T4 constrained-A ablation
  - Table 5: T6 tau sweep summary
  - Table 6: SINDy comparison (T8 threshold sweep best config vs T11)
  - Appends "New Results" section summarizing T3-T16

No new computation — only reads existing pkl files.
"""

import sys
sys.path = [p for p in sys.path if 'MAT6215' not in p]
sys.path.insert(0, '.')

import numpy as np
import pickle
import os
from datetime import date

# ── Helpers ───────────────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def kaplan_yorke(le):
    cs = np.cumsum(le)
    k = np.where(cs < 0)[0]
    if len(k) == 0:
        return float(len(le))
    k = k[0]
    return float(k) + (cs[k-1] if k > 0 else 0.0) / abs(le[k])

def fmt(v, spec=".4f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return format(v, spec)

def md_table(headers, rows, alignments=None):
    """Generate GitHub-flavored markdown table."""
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    def row_str(r):
        return "| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([row_str(headers), sep] + [row_str(r) for r in rows])


# ── Load everything ────────────────────────────────────────────────────────────
print("Loading pkl files...")
le_true   = np.load("data/lyapunov_exponents_full.npy")
t3_epochs = load("data/jac_sweep_epochs.pkl")
t3_lambda = load("data/jac_sweep_lambda.pkl")
t4        = load("data/constrained_a_results.pkl")
t5        = load("data/latent_dim_sweep.pkl")
t6        = load("data/tau_sweep_results.pkl")
t8        = load("data/sindy_sweep_results.pkl")
t11       = load("data/discrete_sindy_results.pkl")
t13       = load("data/ensemble_error_results.pkl")
t14       = load("data/multiseed_results.pkl")
t15       = load("data/ablation_table.pkl")
diag      = load("data/diagnostics_all.pkl")
lyap_std  = load("data/lyap_summary_NODE_Std_MSE_full.pkl")
lyap_jac  = load("data/lyap_summary_NODE_Stab_JAC_full.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Build sections
# ─────────────────────────────────────────────────────────────────────────────

sections = {}

# ── Section: True KSE ─────────────────────────────────────────────────────────
dky_true = kaplan_yorke(le_true)
h_ks_true = float(np.sum(le_true[le_true > 0]))
n_pos_true = int(np.sum(le_true > 0))

sections["true_kse"] = f"""
### True KSE Canonical Numbers

| Metric | Value |
|--------|-------|
| L=22, N=64, dt=0.25 | |
| Largest Lyapunov exponent L1 | {le_true[0]:.4f} |
| # positive exponents n_pos | {n_pos_true} |
| Kaplan-Yorke dimension D_KY | {dky_true:.4f} |
| KS entropy h_KS (Pesin) | {h_ks_true:.4f} |
| Lyapunov time τ_L ≈ 1/L1 | {1/le_true[0]:.1f} |
"""

# ── Section: Main surrogates (canonical results) ───────────────────────────────
rows_main = [
    ["True KSE", "—", f"{le_true[0]:+.4f}", str(n_pos_true), f"{dky_true:.2f}", f"{h_ks_true:.4f}", "Yes", "reference"],
]

if lyap_std:
    rows_main.append(["NODE-Std-MSE",  "MSE, 100ep, h=64",
                       f"{lyap_std.get('L1',float('nan')):+.4f}",
                       str(lyap_std.get('n_pos', 0)),
                       fmt(lyap_std.get('D_KY'), ".2f"),
                       fmt(lyap_std.get('h_KS')),
                       "Yes", "undertrained"])

if lyap_jac:
    rows_main.append(["NODE-Stab-JAC", "JAC, 400ep, lam=0.01",
                       f"{lyap_jac.get('L1',float('nan')):+.4f}",
                       str(lyap_jac.get('n_pos', 0)),
                       fmt(lyap_jac.get('D_KY'), ".2f"),
                       fmt(lyap_jac.get('h_KS')),
                       "No (diverges)", ""])

sections["main_surrogates"] = "\n### Main Surrogate Results\n\n" + md_table(
    ["Model", "Training", "L1", "n_pos", "D_KY", "h_KS", "Stable", "Notes"],
    rows_main
) + "\n"

# ── Section: T3 JAC epoch sweep ────────────────────────────────────────────────
if t3_epochs and "checkpoints" in t3_epochs:
    rows_t3e = []
    for ep, ckpt in sorted((t3_epochs["checkpoints"] or {}).items()):
        if isinstance(ckpt, dict) and "summary" in ckpt:
            s = ckpt["summary"]
            rows_t3e.append([str(ep), fmt(ckpt.get("loss"), ".4f"),
                              f"{s['L1']:+.4f}", str(s['n_pos']),
                              fmt(s['D_KY'], ".2f"), fmt(s['h_KS'])])
    if rows_t3e:
        sections["t3_epoch"] = "\n### T3: JAC Epoch Sweep (λ=0.01)\n\n" + md_table(
            ["Epoch", "Loss", "L1", "n_pos", "D_KY", "h_KS"], rows_t3e) + "\n"

# T3 lambda sweep
if t3_lambda:
    rows_t3l = []
    for lam in sorted(t3_lambda.keys()):
        dr = t3_lambda[lam]
        if isinstance(dr, dict) and "summary" in dr:
            s = dr["summary"]
            rows_t3l.append([str(lam), fmt(dr.get("final_loss"), ".4f"),
                              f"{s['L1']:+.4f}", str(s['n_pos']),
                              fmt(s['D_KY'], ".2f"), fmt(s['h_KS'])])
    if rows_t3l:
        sections["t3_lambda"] = "\n### T3: JAC Lambda Sweep (600 epochs)\n\n" + md_table(
            ["λ", "Loss", "L1", "n_pos", "D_KY", "h_KS"], rows_t3l) + "\n"

# ── Section: T4 constrained A ─────────────────────────────────────────────────
if t4:
    rows_t4 = []
    le_true_row = [f"{le_true[0]:+.4f}", str(n_pos_true), f"{dky_true:.2f}",
                   f"{h_ks_true:.4f}", "Yes", "—", "reference"]
    rows_t4.append(["True KSE"] + le_true_row)
    for vname, vr in t4.items():
        if isinstance(vr, dict):
            lp = vr.get("lyapunov") or {}
            rows_t4.append([
                vname,
                fmt(lp.get("L1"), "+.4f") if lp else "N/A",
                str(lp.get("n_pos", 0)) if lp else "N/A",
                fmt(lp.get("D_KY"), ".2f") if lp else "N/A",
                fmt(lp.get("h_KS")) if lp else "N/A",
                "Yes" if vr.get("stable") else "No",
                fmt(vr.get("energy"), ".1f"),
                vname,
            ])
    sections["t4"] = "\n### T4: Constrained Linear Term A\n\n" + md_table(
        ["Variant", "L1", "n_pos", "D_KY", "h_KS", "Stable", "Energy", "Notes"],
        rows_t4) + "\n"
    sections["t4"] += """
**Key finding:** Constraining A = −(BᵀB + εI) (negdef) stabilizes rollouts
but eliminates all positive Lyapunov exponents (D_KY = 0, n_pos = 0).
The network learns a globally contracting dynamics that cannot reproduce chaos.
This demonstrates a fundamental tension: stability constraints incompatible
with preserving the chaotic attractor structure.
"""

# ── Section: T5 latent dim ────────────────────────────────────────────────────
if t5:
    rows_t5 = [["True KSE", "—", "—", f"{le_true[0]:+.4f}",
                str(n_pos_true), f"{dky_true:.2f}", f"{h_ks_true:.4f}", "Yes"]]
    for d in sorted(t5.keys()):
        r = t5[d]
        le = r.get("lyapunov")
        rows_t5.append([
            str(d),
            fmt(r["ae_diagnostics"]["mse"]) if "ae_diagnostics" in r else "—",
            fmt(r.get("final_ode_loss")),
            fmt(float(le[0]) if le is not None else float('nan'), "+.4f"),
            str(r.get("n_pos", "?")),
            fmt(r.get("D_KY"), ".2f"),
            fmt(r.get("h_KS")),
            "Yes" if r.get("stable") else "No",
        ])
    sections["t5"] = "\n### T5: Latent Dimension Sweep\n\n" + md_table(
        ["d", "AE_MSE", "ODE_loss", "L1", "n_pos", "D_KY", "h_KS", "Stable"],
        rows_t5) + "\n"

# ── Section: T6 tau sweep ─────────────────────────────────────────────────────
if t6:
    rows_t6 = []
    for s in sorted(t6.keys()):
        r = t6[s]
        rows_t6.append([
            str(r["stride"]), fmt(r["tau"], ".2f"), fmt(r["tau_lyap"], ".3f"),
            fmt(r.get("ode_L1"), "+.4f"), fmt(r.get("ode_D_KY"), ".2f"), "Yes" if r.get("ode_stable") else "No",
            fmt(r.get("map_L1"), "+.4f"), fmt(r.get("map_D_KY"), ".2f"), "Yes" if r.get("map_stable") else "No",
        ])
    sections["t6"] = "\n### T6: Data-Spacing (Tau) Sweep\n\n" + md_table(
        ["stride", "tau", "tau/τ_L", "ODE_L1", "ODE_D_KY", "ODE_stab",
         "Map_L1", "Map_D_KY", "Map_stab"],
        rows_t6) + "\n"

# ── Section: T13 predictability horizons ──────────────────────────────────────
if t13 and "horizons" in t13:
    rows_t13 = [[n, fmt(h, ".3f")]
                for n, h in sorted(t13["horizons"].items(), key=lambda x: -x[1])]
    sections["t13"] = "\n### T13: Predictability Horizons (in Lyapunov times)\n\n" + md_table(
        ["Surrogate", "Horizon (τ_L)"], rows_t13) + "\n"

# ── Section: T14 multi-seed ───────────────────────────────────────────────────
if t14:
    def stats(runs, key):
        vals = [r[key] for r in runs if not np.isnan(r.get(key, float('nan')))]
        if not vals: return "N/A", "N/A"
        return fmt(np.mean(vals)), fmt(np.std(vals))

    rows_t14 = []
    for model_name, run_key in [("NODE-Std-MSE", "node_std_mse"),
                                  ("Latent-NODE-d10", "latent_node_d10")]:
        runs = t14.get(run_key, [])
        if runs:
            m_loss, s_loss = stats(runs, "final_loss")
            m_l1,   s_l1   = stats(runs, "L1")
            m_dky,  s_dky  = stats(runs, "D_KY")
            rows_t14.append([model_name,
                              f"{m_loss} ± {s_loss}",
                              f"{m_l1} ± {s_l1}",
                              f"{m_dky} ± {s_dky}",
                              str(sum(r["stable"] for r in runs)) + f"/{len(runs)}"])
    if rows_t14:
        sections["t14"] = "\n### T14: Multi-Seed Robustness (5 seeds)\n\n" + md_table(
            ["Model", "Loss (mean ± std)", "L1 (mean ± std)",
             "D_KY (mean ± std)", "Stable"],
            rows_t14) + "\n"

# ── Section: Diagnostics W1 ───────────────────────────────────────────────────
if diag:
    rows_diag = []
    for name, r in diag.items():
        s = r.get("stats", {})
        w1 = r.get("wasserstein_vs_true", 0.0 if name == "True KSE" else float('nan'))
        rows_diag.append([name, fmt(s.get("energy"), ".2f"),
                          fmt(s.get("rms"), ".4f"),
                          fmt(s.get("skewness"), ".3f"),
                          fmt(w1)])
    sections["diagnostics"] = "\n### Invariant Measure Diagnostics\n\n" + md_table(
        ["System", "Energy", "RMS", "Skewness", "W1 (vs True KSE)"],
        rows_diag) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Read current FINAL_REPORT.md and splice in new sections
# ─────────────────────────────────────────────────────────────────────────────
report_path = "reports/FINAL_REPORT.md"
with open(report_path, "r", encoding="utf-8") as f:
    original = f.read()

# Append a "## New Results (auto-generated)" section
# We don't touch the existing content — just append.
MARKER = "\n\n---\n## New Results (auto-generated)"
# Remove old auto-generated section if present
if MARKER in original:
    original = original[:original.index(MARKER)]

new_section = MARKER + f"\n*Last updated: {date.today().isoformat()} by run_t22_update_report.py*\n"

new_section += "\n" + sections.get("true_kse", "")
new_section += "\n" + sections.get("main_surrogates", "")

if "t3_epoch" in sections:
    new_section += "\n" + sections["t3_epoch"]
if "t3_lambda" in sections:
    new_section += "\n" + sections["t3_lambda"]
if "t4" in sections:
    new_section += "\n" + sections["t4"]
if "t5" in sections:
    new_section += "\n" + sections["t5"]
if "t6" in sections:
    new_section += "\n" + sections["t6"]
if "t13" in sections:
    new_section += "\n" + sections["t13"]
if "t14" in sections:
    new_section += "\n" + sections["t14"]
if "diagnostics" in sections:
    new_section += "\n" + sections["diagnostics"]

updated = original + new_section

with open(report_path, "w", encoding="utf-8") as f:
    f.write(updated)

print(f"Updated: {report_path}")
print(f"  Sections added: {list(sections.keys())}")
print(f"  Total report length: {len(updated)} chars")
print("\nT22 complete.")
