# Linot 2022 Stabilized NODE Reproduction

Paper: **Stabilized Neural Ordinary Differential Equations for Long-Time Forecasting of Dynamical Systems**  
Reference PDF: [2203.15706v2_stab_lin_operator.pdf](/c:/Users/nadir/Desktop/MAT6215/2203.15706v2_stab_lin_operator.pdf)

## Scope

This folder is for the **KSE-only** reproduction path from Linot, Zeng, and
Graham (2022).

In scope:
- three KSE models:
  - standard nonlinear NODE
  - fixed-linear NODE
  - CNN-stabilized NODE
- the core KSE diagnostics from the paper:
  - spacetime rollout comparison
  - ensemble short-time error
  - energy spectrum
  - joint PDF of `(u_x, u_xx)`
  - CNN noise robustness
- an explicit **extension** section for Lyapunov diagnostics

Out of scope for now:
- the viscous Burgers section
- ROM / Galerkin material
- CLV analysis

## Important Correction

This reproduction does **not** use the earlier Hurwitz-constrained
`A = S - D` story.

For the KSE experiment in this paper, the relevant models are:

1. `nonlinear`
   - `f(u) = MLP(u)`
2. `fixed_linear`
   - `f(u) = A_true u + MLP(u)`
   - where `A_true` is the true KSE linear operator applied spectrally
3. `cnn`
   - `f(u) = conv_periodic(u, w) + MLP(u)`
   - where `w` is a learned 5-tap periodic filter

So the key architectural comparison is:
- no explicit stabilizing linear structure
- exact fixed linear structure
- learned local convolutional linear structure

## Data

The workflow reuses the existing cached `L=22, N=64, dt=0.25` KS data when
available:

- [ks_l22_n64_dt025_train.npy](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data/ks_l22_n64_dt025_train.npy)
- [ks_l22_n64_dt025_test.npy](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data/ks_l22_n64_dt025_test.npy)

Local copies for this folder are written to:

- [linot2022_ks_train.npy](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/data/linot2022_ks_train.npy)
- [linot2022_ks_test.npy](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/data/linot2022_ks_test.npy)
- [linot2022_ks_meta.json](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/data/linot2022_ks_meta.json)

If those caches do not exist, the workflow can generate them with the shared
[ks_solver.py](/c:/Users/nadir/Desktop/MAT6215/ks_solver.py).

## Deviations From Paper

The first implementation pass intentionally reuses the cached `linot2021_ks`
trajectory rather than regenerating the paper's full `10^5` time-unit KSE
dataset.

Current practical setup:
- cached split: `20000` train snapshots and `8000` test snapshots
- effective time span: `5000` training time units and `2000` test time units
- training budget: `3000` epochs rather than the paper's much larger PyTorch run

This is enough for a credible first reproduction and keeps the workflow
compatible with the existing KS cache. Any final report should state this
explicitly.

## Workflow

The staged runner is:

- [run_reproduction.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/run_reproduction.py)

Main stages:

1. `data`
   - load or generate the `L=22` KSE dataset
2. `train`
   - train the three NODE variants with one-step integrated `L1` loss
3. `diag`
   - generate the paper-style diagnostics
4. `dyn`
   - run the Lyapunov-spectrum extension
5. `summary`
   - regenerate the local markdown/json summary

## Files

- [models.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/models.py)
  - model definitions and one-step integration wrappers
- [data.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/data.py)
  - dataset loading, caching, and noise injection helpers
- [train.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/train.py)
  - one-step integrated `L1` training loop
- [diagnostics.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/diagnostics.py)
  - Figure 6-9 style diagnostics
- [dynamics_extension.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/dynamics_extension.py)
  - Lyapunov spectrum, `n_pos`, `D_KY`, `h_KS`
- [run_reproduction.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2022_stab/run_reproduction.py)
  - staged orchestrator with cache-aware reruns

## Planned Outputs

Paper-faithful outputs:

- `fig6_spacetime.png`
- `fig7_ensemble_error.png`
- `fig8_joint_pdf.png`
- `fig9_noise_robustness.png`

Bonus rollout-derived diagnostic:

- `fig_energy_spectrum.png`

Extension outputs:

- `fig_ext_lyapunov_spectrum.png`
- `fig_ext_metrics_bars.png`
- `dynamics_extension_summary.json`

## Project Positioning

This folder is the bridge between the two existing KS reproductions:

- [linot2021_ks](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks)
  - latent reduced-manifold learning
- [park2024_ks](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks)
  - Jacobian matching and Lyapunov fidelity

The new question here is:

**If a stabilized NODE matches the paper's statistical diagnostics, does it
also preserve the correct Lyapunov structure?**
