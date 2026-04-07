# MAT6215 Project Status
Last updated: 2026-04-05 (post-audit of T3/T5/T6 + TASKS.md sync)

---

## Canonical Numbers (from saved `.npy` / `.pkl` artifacts)

### True KSE (`lyapunov_exponents_full.npy` - 64 modes)
| Metric | Value |
|--------|-------|
| L1 (largest LE) | +0.0381 |
| n_pos (positive LEs) | 3 |
| D_KY (Kaplan-Yorke) | 5.11 |
| h_KS (KS entropy, Pesin) | 0.0487 |

### NODE-Std-MSE (`node_standard_mse.pkl` + `lyap_NODE_Std_MSE_full.npy`)
| Metric | Value |
|--------|-------|
| Architecture | 64 -> 64 -> 64 (hidden=64, n_layers=2) |
| Training | 100 epochs, best loss=0.0511 |
| L1 (64-mode Lyapunov) | +0.0035 |
| n_pos | 31 |
| D_KY | 62.96 |
| h_KS | 0.0085 |
| Rollout | Stable (low energy) |
| Note | Undertrained: small arch, only 100 epochs |

### NODE-Stab-MSE (`node_stabilized_mse.pkl` + `lyap_NODE_Stab_MSE.npy` - 20 modes only)
| Metric | Value |
|--------|-------|
| Architecture | 64 -> 128 -> 128 -> 128 -> 64, stabilized (`A*u + F(u)`) |
| Training | 600 epochs, best loss=0.0526 |
| L1 (20-mode Lyapunov) | +0.0583 |
| n_pos | >=20 (all 20 computed modes positive) |
| D_KY | lower bound only (20-mode spectrum) |
| h_KS | lower bound only (20-mode spectrum) |
| Rollout | Diverges |
| Note | Full 64-mode spectrum not computed |

### NODE-Stab-JAC (`node_stabilized_jac.pkl` + `lyap_NODE_Stab_JAC_full.npy`)
| Metric | Value |
|--------|-------|
| Architecture | 64 -> 128 -> 128 -> 128 -> 64, stabilized (`A*u + F(u)`) |
| Training | 400 epochs, JAC loss=730.4 |
| L1 (64-mode Lyapunov) | +0.6397 |
| n_pos | 25 |
| D_KY | 52.36 |
| h_KS | 1.1870 |
| Rollout | Diverges |
| Note | 400 epochs appears insufficient; lambda=0.01 underweights the Jacobian term |

### SINDy PI (`sindy_model.pkl` + `traj_sindy_pi.npy`)
| Metric | Value |
|--------|-------|
| Type | Physics-informed: fixed KSE linear operator + STLSQ on nonlinear residual |
| POD modes | 10 |
| Rollout | Stable, but energy is inflated |
| Lyapunov | Not computed for the saved canonical artifact |

---

## Verification Update - 2026-04-05

The project summary was checked against the current code and saved artifacts.
Three important findings came out of that audit.

### 1. T3 was mischaracterized in the older notes

`run_t3_jac_sweep.py` and the saved files:
- `data/jac_sweep_epochs.pkl`
- `data/jac_sweep_lambda.pkl`

already contain Lyapunov summaries at checkpoints and lambda runs.
So T3 is **not** "loss only" in the current codebase.

What is still missing:
- `W1`
- `KL`
- power spectrum
- autocorrelation
- other richer non-Lyapunov diagnostics per checkpoint / per lambda

### 2. Latent/discrete Lyapunov utilities were repaired

`latent_node.py` was updated so that the latent and discrete-map Benettin QR paths now:
- share one implementation
- use warmup before accumulation
- use safer QR sign handling
- sort Lyapunov exponents in descending order before reporting
- compute `D_KY` from sorted exponents

This was necessary because latent spectra can be near-neutral and finite-time fragile,
and the older T5/T6 summaries could overstate confidence in `D_KY`.

### 3. T5/T6 artifacts are now stale

Because the Lyapunov utilities changed, the following artifacts should be regenerated
before they are quoted in the final report:

- `data/latent_dim_sweep.pkl`
- `figures/figT5_latent_dim_sweep.png`
- `data/tau_sweep_results.pkl`
- `figures/figT6_tau_sweep.png`

Interpretation until rerun:
- old T5/T6 qualitative trends are still useful
- old T5/T6 `D_KY` values should be treated as provisional

---

## Completed Infrastructure

### Core code
- [x] `ks_solver.py` - ETD-RK4, Hermitian symmetry enforcement, warmup, integrate
- [x] `lyapunov.py` - Benettin QR via JVP through discrete step, Ginelli CLV
- [x] `neural_ode.py` - Standard/Stabilized NODE, MSE/JAC loss, rollout via diffrax
- [x] `latent_node.py` - latent ODE, discrete latent map, repaired latent/discrete Lyapunov utilities
- [x] `sindy.py` - POD + polynomial library + STLSQ + physics-informed variant
- [x] `diagnostics.py` - power spectrum, joint PDF, W1, KL, autocorr
- [x] `train.py` - training loops with Adam + LR schedule
- [x] `generate_figures.py` - figure generation
- [x] `main.py` - full pipeline orchestrator
- [x] `TASKS.md` - task tracker synced to current audit
- [x] WSL2 Ubuntu GPU environment - JAX CUDA working in `/home/nadir/venvs/mat6215`

### Cached data
- [x] `traj_train.npy` - 20000 steps x 64 modes
- [x] `traj_analysis.npy` - 8000 steps x 64 modes
- [x] `lyapunov_exponents_full.npy` - true KSE 64-mode spectrum

### Trained canonical models
- [x] `node_standard_mse.pkl`
- [x] `node_stabilized_mse.pkl`
- [x] `node_stabilized_jac.pkl`
- [x] `sindy_model.pkl`

### Canonical Lyapunov summaries
- [x] True KSE (64-mode)
- [x] NODE-Std-MSE (64-mode)
- [x] NODE-Stab-MSE (20-mode only)
- [x] NODE-Stab-JAC (64-mode)

### Main report figures already on disk
- [x] `fig1_spacetime.png`
- [x] `fig2_power_spectrum.png`
- [x] `fig3_joint_pdf.png`
- [x] `fig4_pdf_autocorr.png`
- [x] `fig5_lyapunov_spectrum.png`
- [x] `fig5_lyapunov_spectrum_full.png`
- [x] `fig6_summary_bars.png`
- [x] `fig7_training_loss.png`
- [x] `fig8_distance_table.png`
- [x] `fig_sindy_pi_spacetime.png`

### Reports already written
- [x] `reports/report_linot2022_stabilized.md`
- [x] `reports/report_park2024.md`
- [x] `reports/report_ozalp2024.md`
- [x] `reports/FINAL_REPORT.md`

---

## Task Progress Snapshot

| Task | Status | Notes |
|------|--------|-------|
| T1 Reconcile results | DONE | STATUS + FINAL_REPORT were reconciled to canonical saved artifacts |
| T2 Latent NODE | DONE | Implemented in `latent_node.py` and executed via `run_t2_latent_node.py` |
| T3 JAC sweep | DONE | Re-run on 2026-04-06 with corrected `(T,N,N)` Jacobian cache; still dynamically wrong and rollout-unstable |
| T4 Constrained A | DONE | `constrained_a_results.pkl`; stable constrained variants kill chaos |
| T5 Latent dim sweep | RERUN NEEDED | Code was repaired on 2026-04-05; old artifact is stale |
| T6 Tau sweep | RERUN NEEDED | Code was repaired on 2026-04-05; old artifact is stale |
| T7 Discrete latent map | DONE | Implemented as part of the latent NODE pass |
| T8 SINDy threshold sweep | DONE | CPU sweep completed; stable regimes found |
| T9 SINDy library expansion | DONE | Degree/mode sweep completed inside T8 |
| T10 SINDy derivatives | DONE | Exact RHS, finite-difference, and spectral targets compared |
| T11 Discrete SINDy | SCRIPT READY | Not run yet |
| T12 CLV surrogates | SCRIPT READY | Not run yet |
| T13 Ensemble errors | SCRIPT READY | Not run yet |
| T14 Multi-seed | SCRIPT READY | Not run yet |
| T15 Ablations | SCRIPT READY | Not run yet |
| T16 Trajectory supervision | SCRIPT READY | Not run yet |
| T17 Solver tolerance | PENDING | No script yet |
| T18 Irregular times | PENDING | No script yet |
| T19 Jacobian geometry | SCRIPT READY | Not run yet |
| T20 L=44 | SCRIPT READY | Not run yet |
| T21 Parameterized study | PENDING | No script yet |
| T22 Final report clean | SCRIPT READY | Needs rerun after refreshed artifacts |
| DIAG Full diagnostics | SCRIPT READY | Not run yet |

---

## Known Issues / Current Limitations

1. `NODE-Std-MSE` is undertrained relative to the larger stabilized variants.
2. The T3 CPU sweep suggests JAC undertraining is not the only issue:
   150-1000 epochs and lambda sweep 0.001-0.1 remain dynamically poor.
3. Stabilized NODE: the simple constrained-`A` variants tested so far stabilize
   rollout but overdamp the system and kill chaos.
4. `NODE-Stab-MSE` only has a 20-mode Lyapunov computation; a full 64-mode
   comparison is still missing.
5. `diagnostics.pkl` only covers True KSE, NODE-Std-MSE, and SINDy.
6. CLV angles for surrogate models have not been computed yet.
7. Ensemble short-time error curves have not been computed yet.
8. Multi-seed robustness has not been computed yet.
9. T5/T6 latent-model summaries must be refreshed after the Lyapunov utility repair.
10. The latent nonlinear AE path is still weak compared with the POD latent path.

---

## Next Steps

Immediate next actions:
1. Re-run `run_t5_latent_dim_sweep.py`
2. Re-run `run_t6_tau_sweep.py`
3. Decide whether T3 should be extended with `W1`, `KL`, power spectrum, and autocorrelation
4. Build the benchmark/methods/fidelity tables for the final literature report
5. Run the highest-value unexecuted scripts:
   - `run_t12_clv_surrogates.py`
   - `run_t13_ensemble_errors.py`
   - `run_t14_multiseed.py`
   - `run_t19_jacobian_geometry.py`

---

## How To Resume

1. Read `CLAUDE.md`, `STATUS.md`, and `TASKS.md`
2. Treat T5/T6 saved outputs as stale until regenerated
3. All base KSE trajectories in `data/` can be reused; no need to regenerate them
4. Preferred runtime for new heavy experiments: WSL2 Ubuntu + GPU
   Activate with: `source /home/nadir/venvs/mat6215/bin/activate`
   Project path in WSL: `/mnt/c/Users/nadir/Desktop/MAT6215`
5. Windows Python is still available for legacy CPU-only reruns:
   `C:/Users/nadir/AppData/Local/Programs/Python/Python313/python.exe`
