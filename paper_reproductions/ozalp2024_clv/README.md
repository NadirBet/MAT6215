# Ozalp & Magri 2024 CLV Reproduction

Paper: **Data-driven analysis and forecasting of chaotic systems with covariant Lyapunov vectors**  
Reference PDF: [2410.00480v1_clvs.pdf](/c:/Users/nadir/Desktop/MAT6215/2410.00480v1_clvs.pdf)

## Scope

This folder is for the **KSE-only** part of the Ozalp and Magri (2024) paper,
with two explicit presets:

- `paper`
  - paper-faithful KSE setting
  - `L = 2*pi*10 ~= 62.83`
  - `N = 128`
- `project`
  - project-aligned setting for cross-paper comparisons
  - `L = 22`
  - `N = 64`

The main reproduced components are:

1. convolutional autoencoder
2. echo state network in latent space
3. Lyapunov spectrum comparison
4. CLV angle-distribution comparison

## Presets

### `paper`

Use this when the goal is direct article fidelity.

Default target values from the paper:
- reference `lambda_1 ~= 0.085`
- reference `D_KY ~= 15.018`
- latent size around `24`
- large ESN reservoir

### `project`

Use this when the goal is direct comparison against the existing Linot 2021 and
Park 2024 reproductions in this repo.

Project-aligned target values:
- paper reports for `L=22`:
  - reference `lambda_1 ~= 0.046`
  - reference `D_KY ~= 6.007`
- this preset reuses the cached `linot2021_ks` `L=22, N=64, dt=0.25` data

## Files

- [data.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/data.py)
  - preset definitions, dataset loading, normalization, and cached splits
- [cae.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/cae.py)
  - periodic convolutional autoencoder
- [esn.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/esn.py)
  - ESN initialization, teacher forcing, ridge fit, closed-loop map, Jacobian
- [train_cae.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/train_cae.py)
  - CAE training loop and latent-dimension sweep helper
- [train_esn.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/train_esn.py)
  - ESN fitting and validation search
- [reference_clv.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/reference_clv.py)
  - reference KSE Lyapunov spectrum and CLV-angle cache
- [latent_clv.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/latent_clv.py)
  - ESN latent-space Lyapunov spectrum and CLV-angle cache
- [angle_diagnostics.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/angle_diagnostics.py)
  - figure helpers for spectra and angle distributions
- [project_extension.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/project_extension.py)
  - extension: apply the same CLV-angle workflow to the Linot 2021 latent NODE
- [run_reproduction.py](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/ozalp2024_clv/run_reproduction.py)
  - staged orchestrator with cached reruns

## Deviations To Document

The preset system makes the deviations explicit.

For `project`:
- reuse cached `linot2021_ks` `L=22` data
- smaller latent dimension and smaller reservoir than the main paper setting

For `paper`:
- the workflow is designed to support the paper's `L ~= 62.83` setting, but
  the exact runtime choice still depends on local compute and cache availability

## Output Targets

Paper-facing outputs:
- `fig4_reconstruction_mse.png`
- `fig7_lyapunov_spectrum.png`
- `fig8_clv_angle_distributions.png`
- `RESULTS.md`

Extension outputs:
- `ext_lyapunov_comparison.png`
- `ext_clv_angles_comparison.png`

## Project Positioning

This paper is the bridge between:

- latent reduced models from [linot2021_ks](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks)
- tangent-space fidelity from [park2024_ks](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks)

The key new question is:

**Can a latent surrogate preserve not only trajectories and statistics, but
also the CLV-angle geometry of the attractor?**
