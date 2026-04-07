# Reproduction Plan

## Goal

Reproduce the main Kuramoto-Sivashinsky experiments from
[2109.00060_reduced_manifold.pdf](c:/Users/nadir/Desktop/MAT6215/2109.00060_reduced_manifold.pdf),
starting with `L = 22` and preserving the option to extend to `L = 44` and `L = 66`.

## What The Paper Uses

- periodic KS equation
- `64` grid-point state representation
- domain sizes `L = 22, 44, 66`
- reduced manifold dimensions interpreted near:
  - `8, 18, 28`
- autoencoder / hybrid PCA correction for representation
- latent Neural ODE for reduced dynamics
- comparisons against:
  - full physical-space NODE
  - full Fourier-space NODE
  - discrete reduced map

## Exact Figure / Table Targets

- `Figure 2`: reconstruction MSE vs latent dimension
- `Table I`: architectures for Sections III A-C
- `Figure 3`: trajectory comparisons
- `Figure 4`: high-wavenumber blowup in full-space models
- `Figures 5-9`: temporal spacing `tau` study
- `Table II`: architectures for Section III D
- `Figures 10-12`: latent-dimension sweet-spot study

## Build Order

### Phase 1: Figure 2 baseline for `L = 22`

Purpose:

- reproduce the basic representation-quality curve
- use x-axis label `d`, not a normalized or shifted manifold label

Minimal deliverables:

- saved reconstruction-MSE data
- figure with `d` on the x-axis

### Phase 2: Exact representation model for `L = 22`

- implement the paper's hybrid PCA/NN decoder idea
- compare simple PCA baseline with the hybrid correction
- identify the elbow near the expected low-dimensional manifold size

### Phase 3: Figure 3 / 4 dynamics comparison

- latent-space NODE
- full physical-space NODE
- full Fourier-space NODE
- reproduce trajectory comparison and diagnose high-wavenumber artifacts

### Phase 4: `tau` study

- train latent NODE and discrete map at different data spacings
- reproduce:
  - short-time tracking error
  - autocorrelation
  - joint PDF of `(u_x, u_xx)`
  - relative PDF error vs `tau / tau_L`

### Phase 5: latent-dimension study

- nonlinear encoder version
- reproduce the sweet-spot behavior in latent dimension

### Phase 6: extend beyond `L = 22`

- repeat for `L = 44`
- repeat for `L = 66`

## Practical Notes

- `Figure 2` is the cheapest entry point and should be completed first
- the `tau` and full-dynamics comparisons are more compute-heavy but still much lighter than the Park/JAC paper
- we should keep this folder separate from older latent-NODE code because the existing code is Linot-inspired, not paper-faithful
