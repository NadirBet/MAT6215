# Codex Handoff

Last updated: 2026-04-07

This file is the fast-start context for the next Codex session.
It is meant to answer:

1. what has already been done
2. what is running right now
3. what is scaffolded but not launched
4. what is required on a new machine to continue cleanly

## Repo State

- GitHub repo: `https://github.com/NadirBet/MAT6215`
- Current pushed branch: `main`
- Initial pushed commit: `2863ae3` (`Initial project snapshot`)
- This repo intentionally excludes large caches and generated outputs via `.gitignore`

Important consequence:
- cloning from GitHub gives you the code and notes
- it does **not** give you the large `data/`, `figures/`, and reproduction caches/results

## What Was Completed

### 1. Linot 2021 (`paper_reproductions/linot2021_ks`)

Status: **done for now**

What is complete:
- Figure 2 reproduction across `L=22,44,66`
- core Figure 3/4 story for `L=22`
- a Figure-5-style latent trajectory/error plot

Important result:
- latent-space dynamics are clearly better than the physical/Fourier baselines in the `L=22` core test

Most important local files:
- `paper_reproductions/linot2021_ks/final_figure2/`
- `paper_reproductions/linot2021_ks/results/figure3_l22_core_summary.json`
- `paper_reproductions/linot2021_ks/results/figure3_l22_core.pkl`
- `paper_reproductions/linot2021_ks/figures/figure3_l22_core.png`
- `paper_reproductions/linot2021_ks/figures/figure4_l22_core.png`
- `paper_reproductions/linot2021_ks/figures/figure5_like_l22_latent.png`

Important caveat:
- the latent NODE parameters were **not** archived in the saved `figure3_l22_core.pkl`
- only histories and trajectories were saved
- so later CLV/Lyapunov extensions cannot reuse a latent checkpoint automatically unless a new latent checkpoint is saved in a future rerun

### 2. Park 2024 (`paper_reproductions/park2024_ks`)

Status: **main reproduction done; paper-faithful rerun still running locally**

What was already completed before the current rerun:
- base Park reproduction
- MSE/JAC training paths
- autonomous-rollout diagnostics
- multiple MSE probes and reruns

Main completed result before the current live run:
- JAC clearly reproduces the Lyapunov structure much better than MSE

Most important completed files:
- `paper_reproductions/park2024_ks/data/park2024_ks_results.pkl`
- `paper_reproductions/park2024_ks/data/park2024_ks_progress.json`
- `paper_reproductions/park2024_ks/figures/park2024_figure8_ks.png`
- `paper_reproductions/park2024_ks/figures/park2024_ks_loss.png`

Important interpretation:
- the original autonomous evaluation is a stronger test than the released Park KS script
- a paper-faithful rerun was therefore launched to evaluate:
  - one-step Figure 8 predictions on true test states
  - Lyapunov exponents along the true trajectory

### 3. Linot 2022 stabilized NODEs (`paper_reproductions/linot2022_stab`)

Status: **scaffolded and compile-checked**

What exists:
- `README.md`
- `data.py`
- `models.py`
- `train.py`
- `diagnostics.py`
- `dynamics_extension.py`
- `run_reproduction.py`

Important implementation notes:
- uses the cached `linot2021_ks` data for the project-aligned path
- batched eval was added to avoid giant one-shot validation passes
- Lyapunov extension loop was converted to `jax.lax.scan`
- ensemble rollouts in diagnostics are vmapped
- energy spectrum is treated as a bonus diagnostic, not a paper-faithful KSE figure

Current state:
- code is ready
- no heavy Linot 2022 training has been launched yet

### 4. Ozalp & Magri 2024 (`paper_reproductions/ozalp2024_clv`)

Status: **scaffolded and compile-checked**

Goal:
- support **two versions**
  - `paper`: paper-faithful KSE setting (`L ~= 62.83`, `N=128`)
  - `project`: project-aligned setting (`L=22`, `N=64`)

What exists:
- `README.md`
- `data.py`
- `cae.py`
- `esn.py`
- `train_cae.py`
- `train_esn.py`
- `reference_clv.py`
- `latent_clv.py`
- `angle_diagnostics.py`
- `project_extension.py`
- `run_reproduction.py`

Important implementation notes:
- both presets are supported from the start
- project preset reuses the cached `linot2021_ks` `L=22` data
- ESN CLV stage avoids materializing a giant dense reservoir Jacobian
- project extension is honest:
  - it attempts to compare against the Linot 2021 latent NODE
  - but it skips cleanly if no archived latent parameters exist

Current state:
- code is ready
- no heavy Ozalp training/CLV run has been launched yet

## What Is Running Right Now

### Park paper-faithful rerun

This is running **only on the local machine**, not in GitHub.

Progress file:
- `paper_reproductions/park2024_ks/data/park2024_ks_paper_eval_progress.json`

Latest known checkpoint at the time of writing:
- stage: `jac_training`
- epoch: `550 / 3000`
- best test loss so far: `32.353279868563575`
- test MSE-only: `0.012462828011864255`
- test relative error: `0.04350115337855833`
- selected/best epoch so far: `550`

This run already completed the tuned MSE stage successfully:
- MSE best checkpoint is around epoch `50`
- that best MSE checkpoint is what will be used downstream

Important:
- this live run is **not** in the GitHub repo
- if the machine is shut down or the working directory is copied elsewhere mid-run, the process does not magically resume

## Documents Already Written

- `STATUS.md`
- `TASKS.md`
- `paper_reproductions/REPRODUCTION_COMPARISON.md`

These are useful, but they reflect different layers:
- `STATUS.md`: broader project status and canonical numbers
- `TASKS.md`: task inventory
- `paper_reproductions/REPRODUCTION_COMPARISON.md`: article-vs-our-results comparison sheet
- `CODEX_HANDOFF.md` (this file): operational resume context for the next Codex

## What Is Not In GitHub

Because of `.gitignore`, the following are not pushed:

- root `data/`
- root `figures/`
- root `reports/`
- root `articles/`
- `_external/`
- PDFs
- `paper_reproductions/**/data/`
- `paper_reproductions/**/results/`
- `paper_reproductions/**/figures/`
- `paper_reproductions/**/final_figure2/`
- `paper_reproductions/park2024_ks/stacNODE-master/`

This was intentional to avoid pushing ~10 GB of local artifacts.

## What To Copy To A New Machine

If the next machine is meant to continue **Linot 2022** and **Ozalp 2024 project-aligned** work efficiently, copy these local folders in addition to cloning the repo:

- `paper_reproductions/linot2021_ks/data/`
- optionally:
  - `paper_reproductions/linot2021_ks/results/`
  - `paper_reproductions/linot2021_ks/figures/`

Why:
- both Linot 2022 project mode and Ozalp project mode reuse the `linot2021_ks` cached `L=22` data

If the next machine only needs **paper-faithful Ozalp**:
- cloning the repo and setting up the environment is enough
- the paper preset can generate its own data

## New-Machine Setup

1. Clone the repo:
   - `git clone https://github.com/NadirBet/MAT6215.git`
2. Create/activate the Python environment
3. Install the JAX/diffrax/optax/matplotlib stack used in this project
4. If doing project-mode Linot/Ozalp work, copy the local `paper_reproductions/linot2021_ks/data/` cache over

## Recommended Next Actions

### If staying on the current machine

1. let the Park paper-faithful rerun finish
2. capture its final Table-5/Figure-8-style results
3. then decide whether to launch:
   - `linot2022_stab`
   - or `ozalp2024_clv`

### If moving to a new machine

1. clone the GitHub repo
2. recreate the environment
3. copy the small Linot 2021 cache if needed
4. choose one of:
   - launch `paper_reproductions/linot2022_stab/run_reproduction.py`
   - launch `paper_reproductions/ozalp2024_clv/run_reproduction.py --preset project`
   - launch `paper_reproductions/ozalp2024_clv/run_reproduction.py --preset paper --allow-generate`

## Known Caveats For The Next Codex

1. The repo contains **code and notes**, not the full local data/results state.
2. The Park live run status should always be read from:
   - `paper_reproductions/park2024_ks/data/park2024_ks_paper_eval_progress.json`
3. The Linot 2021 latent extension in Ozalp is currently **checkpoint-limited**, not conceptually blocked.
4. The most likely near-term heavy runs are:
   - Linot 2022 training
   - Ozalp reference CLV pass
5. Do not assume old PDFs or local caches are present after a clean clone.

## Minimal Resume Checklist

Read, in order:

1. `CODEX_HANDOFF.md`
2. `STATUS.md`
3. `TASKS.md`
4. `paper_reproductions/REPRODUCTION_COMPARISON.md`
5. the relevant paper folder README:
   - `paper_reproductions/linot2022_stab/README.md`
   - `paper_reproductions/ozalp2024_clv/README.md`

That should be enough to continue productively without reconstructing the entire history from scratch.
