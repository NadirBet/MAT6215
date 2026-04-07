# MAT6215 Task List
Last updated: 2026-04-05 (post-audit of T3/T5/T6 + report-first task ordering)

Legend: [x] done | [ ] pending | [~] in progress

---

## Verification Update - 2026-04-05

The codebase and saved artifacts were audited against the current project summary.
This section records what was actually fixed in code, why it was fixed, and which
saved results are now stale and must be regenerated.

### Fixes Applied

1. **T3 was verified first, then extended on 2026-04-05.**
   `run_t3_jac_sweep.py` already computes and saves Lyapunov summaries at the
   epoch checkpoints and lambda runs. The earlier concern that T3 only saved
   loss curves was incorrect for the current script and current `.pkl` files.
   The missing per-checkpoint diagnostics promised in the task text were then
   added to the script: `W1`, `KL`, power spectrum, autocorrelation, and compact
   rollout-statistics summaries now save with each checkpoint / lambda run.

2. **Latent/discrete Lyapunov utilities were repaired in `latent_node.py`.**
   The latent and discrete-map Benettin QR paths now use:
   - one shared `_benettin_qr(...)` implementation
   - optional warmup before accumulation
   - safer QR sign handling
   - explicit descending-order sorting of Lyapunov exponents
   - one shared `compute_discrete_map_lyapunov(...)`
   - a `kaplan_yorke(...)` implementation that sorts exponents before computing `D_KY`

3. **T5 (`run_t5_latent_dim_sweep.py`) was updated to use the repaired latent Lyapunov path.**
   Why fixed:
   - latent spectra can be near-neutral and finite-time fragile
   - finite-time summaries could overstate confidence in `D_KY`
   - `D_KY` values near the latent dimension can be numerical artifacts rather than
     trustworthy geometry
   What changed:
   - longer follow-up runs are triggered when spectra are near-neutral or suspicious
   - warning flags are written when `D_KY` should be treated cautiously

4. **T6 (`run_t6_tau_sweep.py`) was updated to use the repaired latent/discrete Lyapunov path.**
   Why fixed:
   - ODE and MAP should use the same Benettin/QR conventions
   - warmup and exponent sorting are needed before trusting `D_KY`
   - the audit showed the per-`tau` scaling itself is not the bug; a linear discrete-map
     test recovers the exact exponents
   What changed:
   - both ODE and MAP now use the same stabilized Lyapunov logic
   - suspicious spectra trigger longer accumulation runs
   - warning flags are stored instead of silently trusting borderline `D_KY`

### Artifacts Now Stale

- `data/jac_sweep_epochs.pkl`
- `data/jac_sweep_lambda.pkl`
- `figures/figT3_jac_sweeps.png`
- `figures/figT3_jac_diagnostics.png`
- `data/latent_dim_sweep.pkl`
- `figures/figT5_latent_dim_sweep.png`
- `data/tau_sweep_results.pkl`
- `figures/figT6_tau_sweep.png`

These files were produced before the Lyapunov utility repair and should be
regenerated before using them in the final report.

### Immediate Consequences

- Treat the old T5/T6 `D_KY` values as provisional.
- T3 code now matches the intended deliverables, but the saved T3 artifacts are
  stale until the sweep is rerun with the new diagnostics path.

---

## Priority 1 - Must Do

### T1 - Reconcile all results [x]
**Done.** `STATUS.md` and `FINAL_REPORT.md` were updated with canonical numbers
from artifacts.

Key finding:
- `NODE-Std-MSE` was undertrained (100 epochs, hidden=64) versus the previously
  reported 600-epoch interpretation.

Deliverable:
- keep one canonical run per model
- keep saved `.pkl` / `.npy` / figures / report tables consistent

### T2 - Implement Linot-style reduced-manifold NODE [x]
**Done.** `latent_node.py` and `run_t2_latent_node.py` were written and executed.

Current result snapshot:
- POD + latent NODE gives energy close to true KS
- nonlinear AE path is still weak and needs tuning
- discrete latent map (T7) was implemented in the same pass

Artifacts:
- `data/latent_node_results.pkl`
- `figures/figT2_*.png`

### T3 - Finish JAC matching study properly [x]
**Done on 2026-04-06 (with corrected Jacobian cache).**

Artifacts refreshed:
- `data/jac_sweep_epochs.pkl`
- `data/jac_sweep_lambda.pkl`
- `data/jac_training_cache_v2.npz`
- `figures/figT3_jac_sweeps.png`
- `figures/figT3_jac_diagnostics.png`
- timing records to `data/experiment_log.jsonl`

Key result:
- more JAC training and lambda tuning alone do not fix the stabilized NODE
- across the corrected runs, rollout diagnostics remain catastrophic:
  all tested models are unstable long before the 2000-step diagnostic window completes
- epoch sweep at `lambda = 0.01`: `L1 = 0.0850`, `n_pos = 20`, `D_KY = 20.00`,
  `h_KS = 0.4996`, `W1 = 11599.49`, `KL = 19.95`
- lambda sweep at 600 epochs:
  `lambda=0.001` gives the best leading exponent (`L1 = 0.0506`) but still has
  `n_pos = 20`, `D_KY = 20.00`, and unstable rollout diagnostics
  `lambda=0.01` gives `L1 = 0.1974`, `n_pos = 19`, `D_KY = 20.00`
  `lambda=0.05` and `0.1` remain similarly wrong and unstable

Important fix that changed the run:
- the original cached `jac_training_cache.npz` stored the wrong tensor shape
  `(T, N, N, N)` because the helper differentiated the solver Jacobian again
- `neural_ode.py` was fixed so JAC training now uses the true Jacobian shape
  `(T, N, N)`
- a fresh cache was generated as `data/jac_training_cache_v2.npz`

Now implemented in code:
- epoch sweep: 150, 300, 600, 1000
- lambda sweep: 0.001, 0.01, 0.05, 0.1
- saved metrics: `L1`, `n_pos`, `D_KY`, `h_KS`
- `W1`
- `KL`
- power spectrum
- autocorrelation
- compact rollout statistics

### T4 - Fix stabilized NODE (constrain A) [x]
**Done.** `run_t4_constrained_a.py` executed and wrote:
- `data/constrained_a_results.pkl`
- `figures/figT4_constrained_a.png`

Key result:
- unconstrained `A` diverges
- negative-definite and diagonal-negative parameterizations stabilize rollout but
  kill chaos (`D_KY = 0`)
- the simple physics-initialized variant fails badly

Compared variants:
- unconstrained
- `A = -(B^T B + eps I)`
- diagonal-negative
- simple physics-initialized linear term

---

## Priority 2 - Strong Upgrades

### T5 - Latent dimension sweep [~]
**Needs rerun after Lyapunov utility repair.**

Old run:
- `run_t5_latent_dim_sweep.py` completed on CPU and wrote:
  - `data/latent_dim_sweep.pkl`
  - `figures/figT5_latent_dim_sweep.png`
  - timing records to `data/experiment_log.jsonl`

What still looks meaningful from the old run:
- reconstruction improves strongly with latent dimension
- rollout energy can blow up for larger latent dimensions

Why rerun is required:
- the latent Lyapunov pipeline now uses warmup, sorted exponents, safer QR handling,
  and warning flags for near-neutral spectra
- old `D_KY` values should be treated as provisional

Sweep:
- `d = 4, 6, 8, 10, 12, 16`

For each `d`:
- reconstruction error
- rollout energy and stability
- latent Lyapunov spectrum
- `D_KY`, `h_KS`, `n_pos`
- runtime

Refresh outputs:
- `data/latent_dim_sweep.pkl`
- `figures/figT5_latent_dim_sweep.png`

### T6 - Data-spacing (tau) sweep from Linot [~]
**Needs rerun after Lyapunov utility repair.**

Old run:
- `run_t6_tau_sweep.py` completed on CPU and wrote:
  - `data/tau_sweep_results.pkl`
  - `figures/figT6_tau_sweep.png`
  - timing records to `data/experiment_log.jsonl`

What still looks meaningful from the old run:
- the discrete map appears strongly dissipative across all strides
- changing tau does not obviously rescue latent dynamics

Why rerun is required:
- both ODE and MAP now use the same repaired Lyapunov path
- warmup and exponent sorting are now enforced
- warning flags are stored for suspicious spectra
- the audit showed the `tau` scaling itself was not the bug

Sweep:
- strides `1, 2, 4, 8, 16, 32`
- tau `0.25, 0.5, 1.0, 2.0, 4.0, 8.0`

For each tau:
- latent ODE training + rollout
- discrete map training + rollout
- Lyapunov metrics
- stability and energy

Refresh outputs:
- `data/tau_sweep_results.pkl`
- `figures/figT6_tau_sweep.png`

### T7 - Discrete-time latent map [x]
**Done (as part of T2).** `discrete_map_step`, `train_discrete_map`, and
`rollout_discrete_map` were implemented in `latent_node.py`.

Goal:
- compare continuous latent ODE vs discrete latent map in the same reduced space

---

## Priority 3 - SINDy Second Pass

### T8 - SINDy threshold/sparsity sweep [x]
**Done.** `run_t8_sindy_sweep.py` completed on CPU and wrote:
- `data/sindy_sweep_results.pkl`
- `figures/figT8_sindy_sweep.png`
- timing records to `data/experiment_log.jsonl`

Key result:
- there is a usable SINDy regime
- one strong stable regime gives `D_KY` close to the true system

Covered:
- threshold sweep
- library / mode-count sweep
- derivative-target sweep

### T9 - Expand SINDy library [x]
**Done inside T8.**

Covered:
- polynomial degree 2 vs 3
- moderate mode counts vs larger libraries
- stability / energy / dynamical fidelity tradeoffs

### T10 - Improve SINDy derivative handling [x]
**Done inside T8.**

Compared:
- exact RHS
- finite-difference derivatives
- spectral derivative target

Key result:
- exact RHS remains the cleanest path

### T11 - Discrete-time SINDy [ ]
Sparse regression without explicit derivative estimation.

Goal:
- compare continuous-time vs discrete-time SINDy on stability and long-time fidelity

Status:
- script written: `run_t11_discrete_sindy.py`
- not run yet

---

## Priority 4 - Deeper Diagnostics

### T12 - CLV angles for surrogates [ ]
Run Ginelli CLV on surrogate models.

Goal:
- compare unstable-unstable angles
- compare unstable-stable tangencies
- compare near-zero angle distributions

Status:
- script written: `run_t12_clv_surrogates.py`
- not run yet

### T13 - Ensemble short-time error curves [ ]
Many ICs from the attractor. Ensemble-averaged forecast error curves.

Goal:
- plot forecast error vs time in Lyapunov time units
- compare shadowing horizon across surrogates

Status:
- script written: `run_t13_ensemble_errors.py`
- not run yet

### T14 - Multi-seed robustness [ ]
Retrain each model for 3+ seeds.

Goal:
- report mean +/- std for final loss, `L1`, `D_KY`, `h_KS`, energy, `W1`, `KL`

Status:
- script written: `run_t14_multiseed.py`
- not run yet

### T15 - Ablation tables [ ]
Compact ablation:
- width
- depth
- JAC weight
- constrained `A`
- latent dimension
- SINDy threshold/library

Goal:
- show which changes move Lyapunov fidelity, not just MSE

Status:
- script written: `run_t15_ablations.py`
- not run yet

---

## Priority 5 - Faithful to Original NODE Paper

### T16 - Trajectory-supervision training [ ]
Loss on observed states after integration, not just vector field supervision.

Goal:
- compare vector-field supervision vs trajectory supervision

Status:
- script written: `run_t16_traj_supervision.py`
- not run yet

### T17 - Solver tolerance sweep [ ]
Vary forward `rtol` / `atol`.

Measure:
- function evaluations
- runtime
- rollout error
- Lyapunov metrics
- invariant-measure diagnostics

Status:
- not written yet

### T18 - Irregular observation times [ ]
Train latent NODE on irregular observation times.

Goal:
- compare against discrete map or fixed-step surrogate

Status:
- not written yet

---

## Priority 6 - Geometric Diagnostics

### T19 - Local Jacobian singular values [ ]
Measure local singular values of surrogate Jacobians along trajectories.

Goal:
- compare stretching / contraction between true KS and surrogates
- check whether surrogates create too many weakly unstable directions
- if latent autoencoder is used, inspect whether latent coordinates disentangle the attractor

Status:
- script written: `run_t19_jacobian_geometry.py`
- not run yet

---

## Priority 7 - Scope Expansion

### T20 - Experiments at L=44 [ ]
Repeat main experiments at `L = 44`.

Goal:
- check whether the MSE-vs-dynamical-fidelity gap worsens at larger system size

Status:
- script written: `run_t20_l44.py`
- not run yet

### T21 - Parameterized study [ ]
Treat `L` as a parameter.

Goal:
- latent model transfer across nearby `L`
- SINDy structure change across `L`
- JAC importance vs complexity

Status:
- not written yet

---

## Final Polish

### T22 - Clean report to match code [ ]
Update all tables after final reruns.

Required:
- every number from saved artifacts
- add "methods actually run" table
- add "limitations" subsection
- keep canonical headline figure: true vs surrogate Lyapunov spectra

Status:
- script written: `run_t22_update_report.py`
- not run yet after the latest audit/fixes

---

## Report-First Next Steps (2026-04-05)

These are ordered to support the final literature report structure rather than
just to complete isolated scripts.

### R1 - Refresh stale latent artifacts [highest priority]
- rerun `run_t5_latent_dim_sweep.py`
- rerun `run_t6_tau_sweep.py`
- update any report text that still quotes old T5/T6 `D_KY` values

### R2 - Build the benchmark and methods tables
- add a **True KS reference table**:
  `L`, `N`, `dt`, solver, warmup, trajectory lengths, `lambda_1`, `n_pos`, `D_KY`, `h_KS`
- add a **Linot-style methods/setup table**:
  model family, state space, architecture, activation, solver, loss, epochs,
  batch size, learning-rate schedule, linear-term parameterization, latent dimension
- if latent SDE is kept in scope, add a **Park-style latent SDE setup table**
  only after the model exists in code

### R3 - Build the core paper-reproduction tables from existing artifacts
- **Park-style fidelity table**:
  True KSE vs NODE-Std-MSE vs NODE-Stab-JAC vs SINDy
  with `lambda_1`, `n_pos`, `D_KY`, `h_KS`, `W1`, `KL`, rollout stability
- **Linot stabilization table** from `data/constrained_a_results.pkl`
- **Ozalp-style latent-dimension table** from refreshed T5 outputs
- **Tau / data-spacing table** from refreshed T6 outputs

### R4 - Close the highest-value missing diagnostics
- run `run_t12_clv_surrogates.py`
- run `run_t13_ensemble_errors.py`
- run `run_t14_multiseed.py`
- run `run_t19_jacobian_geometry.py`

### R5 - Close the strongest missing baselines
- run `run_t11_discrete_sindy.py`
- run `run_t16_traj_supervision.py`
- run `run_t20_l44.py`

### R6 - Decide whether to extend or narrow T3
- Option A: extend T3 to actually compute `W1`, `KL`, power spectrum, autocorr
  per checkpoint/per lambda
- Option B: keep T3 as a Lyapunov-only sweep and rewrite report/task text accordingly

### R7 - Write missing scripts only if still needed after R1-R6
- T17 solver-tolerance sweep
- T18 irregular-times latent NODE
- T21 parameterized study in `L`
- latent SDE baseline script, if Park latent-SDE comparison is still desired

### R8 - Final reporting pass
- run `run_diagnostics_all.py`
- run `run_t15_ablations.py`
- run `run_t22_update_report.py`
- update `STATUS.md` and `FINAL_REPORT.md`
- freeze the canonical artifact set used in the final literature report

---

## Status Tracker

| Task | Status | Notes |
|------|--------|-------|
| T1 Reconcile results | DONE | `STATUS.md` + `FINAL_REPORT.md` updated; canonical numbers from artifacts |
| T2 Latent NODE | DONE | `latent_node.py` + `run_t2_latent_node.py` implemented and executed |
| T3 JAC sweep | DONE | Re-run on 2026-04-06 with corrected `(T,N,N)` Jacobian cache; still dynamically wrong and rollout-unstable |
| T4 Constrained A | DONE | `run_t4_constrained_a.py` run; negdef stable but `D_KY=0` kills chaos |
| T5 Latent dim sweep | RERUN NEEDED | Code fixed on 2026-04-05; old artifact is stale with provisional Lyapunov summaries |
| T6 Tau sweep | RERUN NEEDED | Code fixed on 2026-04-05; old artifact is stale with provisional Lyapunov summaries |
| T7 Discrete latent map | DONE | Implemented as part of T2 in `latent_node.py` |
| T8 SINDy threshold sweep | DONE | CPU sweep completed; stable regimes found |
| T9 SINDy library expansion | DONE | Library/mode sweep completed inside `run_t8_sindy_sweep.py` |
| T10 SINDy derivatives | DONE | Derivative-method comparison completed inside `run_t8_sindy_sweep.py` |
| T11 Discrete SINDy | SCRIPT READY | `run_t11_discrete_sindy.py` written |
| T12 CLV surrogates | SCRIPT READY | `run_t12_clv_surrogates.py` written; deps: `constrained_a_results.pkl` |
| T13 Ensemble errors | SCRIPT READY | `run_t13_ensemble_errors.py` written; deps: `constrained_a_results.pkl` |
| T14 Multi-seed | SCRIPT READY | `run_t14_multiseed.py` written |
| T15 Ablations | SCRIPT READY | `run_t15_ablations.py` written |
| T16 Trajectory supervision | SCRIPT READY | `run_t16_traj_supervision.py` written |
| T17 Solver tolerance | PENDING | No script yet |
| T18 Irregular times | PENDING | No script yet |
| T19 Jacobian geometry | SCRIPT READY | `run_t19_jacobian_geometry.py` written |
| T20 L=44 | SCRIPT READY | `run_t20_l44.py` written |
| T21 Parameterized study | PENDING | No script yet |
| T22 Final report clean | SCRIPT READY | `run_t22_update_report.py` written; needs rerun after refreshed artifacts |
| DIAG Full diagnostics | SCRIPT READY | `run_diagnostics_all.py` written; deps: `constrained_a_results.pkl` |
