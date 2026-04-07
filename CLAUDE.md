# MAT6215 — KSE Surrogate Dynamical Fidelity Project

## Project Goal
Take the Kuramoto-Sivashinsky equation as a canonical chaotic PDE.
Simulate it, compute its full dynamical fingerprint (Lyapunov spectrum, KY dimension,
KS entropy, CLV angles, invariant statistics, spatial power spectrum).
Train two surrogates on trajectory data — Neural ODE and SINDy.
Run the same diagnostics on both surrogates.
Show that trajectory accuracy and dynamical fidelity are different things,
and that different learning paradigms fail in different ways.

## Key Papers
- 2203.15706v2.pdf — Linot et al. 2022 Stabilized Neural ODEs (extend to Lyapunov)
- 2411.06311v2.pdf — Park et al. 2024 Statistical accuracy, Jacobian matching
- 2410.00480.pdf   — Özalp & Magri 2024 CLV in latent spaces (closest existing work)
- 2109.00060.pdf   — Linot & Graham 2022 Neural ODE ROM of KSE
- 0706.0510.pdf    — Ginelli 2007 CLV algorithm
- 1212.3961.pdf    — Ginelli 2013 CLV review
- 1806.07366.pdf   — Chen 2018 Neural ODEs
- 1509.03580.pdf   — Brunton 2016 SINDy
- 1606.05340.pdf   — Poole 2016 order-to-chaos in deep networks

## Stack
Canonical saved artifacts were produced with Python 3.13, JAX 0.8.2, diffrax 0.7.2, optax 0.2.8, matplotlib 3.10.
Current accelerated environment is WSL2 Ubuntu with Python 3.12 and JAX 0.9.2 CUDA support in `/home/nadir/venvs/mat6215`.

## File Structure
```
MAT6215/
├── CLAUDE.md               # This file — project brain
├── STATUS.md               # Build progress tracker
├── ks_solver.py            # ETD-RK4 pseudospectral KSE solver
├── lyapunov.py             # Benettin QR + Ginelli CLV algorithm
├── diagnostics.py          # KY dim, KS entropy, power spectrum, PDFs, CLV angles
├── neural_ode.py           # MLP + diffrax Dopri5 Neural ODE
├── sindy.py                # Galerkin projection + STLSQ sparse regression
├── train.py                # Training loops for Neural ODE and SINDy
├── main.py                 # Orchestrates full pipeline
├── figures/                # All generated figures
├── reports/
│   ├── report_linot2022_stabilized.md
│   ├── report_park2024.md
│   ├── report_ozalp2024.md
│   └── FINAL_REPORT.md     # Full report with UML, architecture, results
└── data/                   # Saved trajectory data (npy files)
```

## KSE Physics
PDE: u_t = -u*u_x - u_xx - u_xxxx
Domain: [0, L], L=22, periodic boundary conditions
Discretization: N=64 Fourier modes
Key parameters: Lyapunov time τ_L ≈ 22, ~12 positive Lyapunov exponents

## Data Simulation Strategy
Method: Pseudospectral ETD-RK4 in Fourier space
- Linear operator diagonal in Fourier: L_k = q_k^2 - q_k^4, q_k = 2πk/L
- Nonlinear term: N_hat_k = -i*q_k/2 * FFT(u^2)  [pseudospectral]
- ETD-RK4 integrates linear part exactly, no stiffness problem
- dt = 0.25, 2/3 dealiasing
Trajectories:
1. Warmup: t=0 to 500 (discard, let transient die)
2. Training trajectory: t=500 to 25500 (100,000 steps → 25,000 time units)
3. Analysis trajectory: t=0 to 5000 (separate run for Lyapunov/CLV/diagnostics)
4. Ensemble: 100 short trajectories from random ICs for error curves

## Lyapunov Algorithm (Benettin + Ginelli)
Benettin forward pass:
- Evolve state u(t) with ETD-RK4
- Simultaneously evolve N perturbation vectors Q(t) via linearized flow
- Every K=1 steps: QR decompose Q → store R diagonal for exponents
- Lyapunov exponents: λ_i = mean(log|R_ii|) / dt

Ginelli CLV backward pass:
- Store Q matrices from forward pass
- Initialize C = random upper triangular
- Backward pass: C_{n-1} = R_n^{-1} * C_n, then normalize columns
- CLV_i(t) = Q(t) * C_i(t)
- CLV angles: angle between CLV_i and CLV_j = arccos(|CLV_i · CLV_j|)

## Neural ODE Architecture
- MLP: 64 → 256 → 256 → 256 → 64
- Activation: tanh
- Two modes: MSE loss only, MSE + Jacobian matching (λ=0.01)
- Integration: diffrax Dopri5, dt0=0.25
- Optimizer: optax Adam, lr=1e-3 → 1e-4

## SINDy Architecture
- Project KSE state onto first 8 Galerkin modes (eigenvectors of learned linear term)
- Build polynomial library: [1, a_i, a_i*a_j] → ~45 terms for 8 modes
- STLSQ: threshold=0.1, max_iter=10
- Identified ODE in reduced coordinates

## Diagnostics Computed (true system + both surrogates)
1. Full Lyapunov spectrum (all 64 exponents)
2. Kaplan-Yorke dimension: D_KY = k + sum(λ_i, i=1..k) / |λ_{k+1}|
3. KS entropy (Pesin formula): h_KS = sum of positive λ_i
4. Spatial power spectrum: E(q) = |û_q|^2 averaged over time
5. Joint PDF of (u_x, u_xx) — key diagnostic from Linot 2022
6. CLV angles between adjacent vectors — hyperbolicity measure from Özalp 2024
7. Wasserstein distance between invariant measures — from Park 2024
8. Ensemble-averaged prediction error vs Lyapunov time

## If Resuming After Usage Reset
1. Read this file and STATUS.md
2. Check which files exist in the folder
3. Continue from the first pending item in STATUS.md
4. Data files in data/ can be reused — no need to regenerate
5. Preferred runtime for new heavy experiments is WSL2 Ubuntu GPU:
   `source /home/nadir/venvs/mat6215/bin/activate`
   `cd /mnt/c/Users/nadir/Desktop/MAT6215`
