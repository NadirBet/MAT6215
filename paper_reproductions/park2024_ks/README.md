# Park 2024 KS Reproduction

This folder is a clean restart for a faithful reproduction of the
Kuramoto-Sivashinsky experiment in:

- [2411.06311v2.pdf](c:/Users/nadir/Desktop/MAT6215/2411.06311v2.pdf)

It is intentionally separate from the older project code because the existing
pipeline drifted away from the paper setup in several important ways.

## Paper Target

For the KS experiment, the paper and downloaded `stacNODE` repository use:

- a **modified Kuramoto-Sivashinsky equation**
- **127 interior nodes**
- **L = 128**
- **dx = 1**
- **second-order finite differences**
- **Dirichlet + Neumann boundary conditions**
- a **Neural ODE vector field** integrated into a one-step map
- **MLP** architecture for KS
- **GELU** activation for the MLP
- **AdamW**
- `dt = 0.25`
- `epochs = 3000`
- train/test size `[3000, 3000]`
- Jacobian weight `lambda = 1`

The two immediate Park targets in this folder are:

1. **Figure 8**
   True KS solution, MSE Neural ODE solution, JAC Neural ODE solution
2. **Table 5, KS row**
   The first 15 Lyapunov exponents for the true system, MSE model, and JAC model

## Important Difference From The Old Code

The old `T3` experiment trained Jacobian matching on the **continuous-time RHS**
of a periodic spectral KS system in 64 dimensions.

This folder instead follows the Park paper more closely:

- **finite-difference modified KS**
- **127D physical-space state**
- Jacobian loss on the **one-step map** `F(x)` and `dF(x)`, not on the RHS
- plain **MLP/GELU** Neural ODE path for the KS experiment
- repo-aligned **IMEX** stepping for the true KS simulator

## What Tables 7, 8, 9 Are

These are **appendix hyperparameter search tables**, not the main KS result table.

- **Table 7**
  Search over batch size, weight decay, depth, and width for **MLP** Neural ODEs
  trained with **MSE**
- **Table 8**
  Search over batch size, weight decay, depth, and width for **MLP_skip / ResNet**
  Neural ODEs trained with **MSE**
- **Table 9**
  Search over depth and width for **MLP** and **MLP_skip / ResNet**
  trained with **Jacobian matching**

What they report:

- test loss
- one-step relative error averaged over 8000 samples

They are useful for later tuning, but they are not the first reproduction target
for the KS experiment.

## Current Contents

- `config.py`
  Park-specific experiment configuration
- `modified_ks_fd.py`
  Modified KS finite-difference solver with clamped boundary conditions
- `model.py`
  MLP flow-map model with GELU activation
- `train.py`
  Map-level MSE and Jacobian-matching training
- `lyapunov_map.py`
  QR / Benettin Lyapunov computation for differentiable maps
- `run_park2024_ks.py`
  End-to-end script for Figure 8 and the KS row of Table 5

## Progress And Checkpoints

The Park run now writes resumable caches and live progress files so long stages
do not look silent:

- `data/park2024_ks_progress.json`
  Current stage plus the latest progress metrics
- `data/jac_train_cache.npy`, `data/jac_train_cache_meta.json`
  Incremental Jacobian cache for the training set
- `data/jac_test_cache.npy`, `data/jac_test_cache_meta.json`
  Incremental Jacobian cache for the test set
- `data/park2024_ks_mse_history.json`
  Live MSE training snapshots, then final full history
- `data/park2024_ks_jac_history.json`
  Live JAC training snapshots, then final full history
- `data/park2024_ks_partial.pkl`
  Partial results after major stages

## Robustness Notes

This local JAX path now matches the downloaded `stacNODE` code on the major KS
numerics:

- `L = 128`, `n_inner = 127`, `dx = 1`, `dt = 0.25`, `c = 0.4`
- repo-style finite-difference boundary closure for the fourth derivative
- repo-style IMEX update for the true KS simulator
- Neural ODE vector field learned in interior coordinates

It is still a **local reimplementation**, not a byte-for-byte port of the
PyTorch code, so very small numeric differences are still possible.

## KS Table 5 Target Values

From the paper:

- `Lambda_true`
  `[0.3036, 0.2733, 0.2592, 0.2257, 0.2050, 0.1888, 0.1649, 0.1496, 0.1288, 0.1128, 0.0992, 0.0776, 0.0646, 0.0492, 0.0342]`
- `Lambda_mse`
  `[0.1652, 0.1647, 0.1540, 0.1524, 0.1443, 0.1411, 0.1336, 0.1262, 0.1236, 0.1143, 0.1141, 0.1091, 0.1045, 0.0971, 0.0985]`
- `Lambda_jac`
  `[0.2904, 0.2622, 0.2293, 0.1990, 0.1701, 0.1584, 0.1320, 0.1071, 0.0912, 0.0724, 0.0591, 0.0442, 0.0306, 0.0157, 0.0023]`

The goal is not just to print exponents, but to make a visually and numerically
credible comparison against these targets.
