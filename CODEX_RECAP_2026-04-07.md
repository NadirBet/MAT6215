# Codex Recap - 2026-04-07

This file records what was done in this session, what is currently understood about the repo, and the concrete next steps for continuing the work.

## Session Goal

Resume the MAT6215 project from GitHub, recover the correct working context, prepare the local machine to run the reproduction code, and start the two active paper reproduction tracks:

- `paper_reproductions/linot2022_stab`
- `paper_reproductions/ozalp2024_clv`

## What I Did

### 1. Local setup and repo recovery

- installed Git on the machine
- cloned `https://github.com/NadirBet/MAT6215`
- verified the cloned branch is `main`
- confirmed that the starting workspace outside the clone was not itself a Git repo

### 2. Read the operational context

- read `CODEX_HANDOFF.md`
- read `STATUS.md`
- read `TASKS.md`
- read `paper_reproductions/linot2022_stab/README.md`
- read `paper_reproductions/ozalp2024_clv/README.md`

### 3. Corrected the work focus

The first pass over the handoff over-weighted the Park section because it had the most live-run detail.

That emphasis was corrected after re-reading the handoff:

- Park is background context and a prior reproduction stream
- the active forward work is the last two paper folders:
  - `linot2022_stab`
  - `ozalp2024_clv`

### 4. Checked runnable infrastructure

- verified both paper folders contain staged runners and the expected module files
- checked for Python and runtime availability on the machine
- found that:
  - WSL is not installed on this machine
  - Windows Python was available at `C:\Users\nadir\AppData\Local\Programs\Python\Python312\python.exe`
  - the required scientific stack was not installed in that interpreter

### 5. Created a local Python environment

- created `.venv/` at the repo root
- added `.venv/` and `venv/` to `.gitignore`
- installed the project runtime stack into `.venv`

Installed packages include:

- `jax`
- `jaxlib`
- `diffrax`
- `optax`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `tqdm`

Validation completed:

- imports succeeded
- JAX device check succeeded
- current backend available on this machine is CPU only

### 6. Fixed a real runner bug in `linot2022_stab`

File changed:

- `paper_reproductions/linot2022_stab/run_reproduction.py`

Problems found:

- `--only data` still fell through into the training logic
- `--force-stage train` was not enough to make the train stage count as requested

Fixes applied:

- added explicit stage gating logic
- made `--only data` terminate cleanly after dataset creation and summary writing
- made the training stage logic stricter when checkpoints are missing

This was not speculative cleanup. It was necessary because the staged runner did not behave correctly for actual staged execution.

### 7. Started the first real reproduction stage

Executed:

- `python -m paper_reproductions.linot2022_stab.run_reproduction --only data --allow-generate`

Result:

- Linot 2022 dataset generation completed successfully
- local generated files now exist under `paper_reproductions/linot2022_stab/data/`

Generated local artifacts:

- `linot2022_ks_train.npy`
- `linot2022_ks_test.npy`
- `linot2022_ks_meta.json`
- `run_reproduction_progress.json`
- `run_reproduction_summary.json`

Current dataset summary:

- source: generated locally
- train shape: `(20000, 64)`
- test shape: `(8000, 64)`

### 8. Training was probed but not left running

I attempted to start the Linot 2022 training stage.

What is true right now:

- training is not currently running
- the detached background attempt did not stay alive
- a later foreground probe was interrupted before completion

So the honest state is:

- environment ready
- Linot 2022 data ready
- no active heavy run currently in progress from this session

## What I Understand Now

### 1. Main project focus

The repo's active reproduction targets are:

- `paper_reproductions/linot2022_stab`
- `paper_reproductions/ozalp2024_clv`

Park remains relevant context, but it is not the main current workstream for this session.

### 2. Repo composition and limits

The GitHub repo contains:

- code
- notes
- orchestration scripts
- markdown status documents

The GitHub repo does not contain most large local artifacts because `.gitignore` excludes:

- root `data/`
- root `figures/`
- root `reports/`
- paper reproduction data/results/figures caches

This means a clean clone is operationally incomplete for heavy reproduction unless datasets are regenerated or copied.

### 3. Linot 2022 current status

`linot2022_stab` is implemented enough to run.

It includes:

- data loading/generation
- three model definitions
- training
- diagnostics
- Lyapunov extension
- staged orchestration

On this machine:

- the data stage now works
- training has not yet been completed
- diagnostics and dynamics depend on training checkpoints

### 4. Ozalp 2024 current status

`ozalp2024_clv` is also implemented enough to run, with two presets:

- `project`
- `paper`

Important distinction:

- `project` expects the cached `linot2021_ks` `L=22` dataset to exist locally
- `paper` can generate its own dataset if run with `--allow-generate`

At the moment, this clone does not contain the ignored `linot2021_ks/data/` cache, so:

- `ozalp --preset project` is currently blocked by missing cached data
- `ozalp --preset paper --allow-generate` is the runnable preset on this machine without extra data copy

### 5. Machine/runtime reality

The handoff mentions a preferred WSL2 GPU environment, but that environment does not exist on this machine right now.

Current practical runtime is:

- Windows
- local `.venv`
- JAX on CPU

That is sufficient to run the code, but slower than the intended GPU path for heavy experiments.

## Current Tracked Code Changes

These tracked repo changes were made in this session:

- `.gitignore`
- `paper_reproductions/linot2022_stab/run_reproduction.py`

This recap file is an additional tracked change.

## Current Untracked or Ignored State

Local generated data now exists for Linot 2022, but it is ignored by Git and will not be pushed.

That is expected and consistent with the repo policy.

## What I Will Do Next

### Immediate execution order

1. run `linot2022_stab` training cleanly in the foreground or with a verified detached launch
2. confirm checkpoint creation for:
   - nonlinear NODE
   - fixed-linear NODE
   - CNN NODE
3. run Linot 2022 diagnostics stage
4. run Linot 2022 dynamics extension stage
5. summarize the Linot 2022 outputs against the paper README targets

### Ozalp follow-up plan

After Linot 2022 is moving or finished:

1. decide whether to run Ozalp in `paper` mode immediately on this machine
2. if staying on this machine with no copied Linot 2021 cache:
   - run `ozalp2024_clv` with `--preset paper --allow-generate`
3. if the Linot 2021 cache is later copied in:
   - run `ozalp2024_clv` with `--preset project`
4. if the project preset is used, check whether the Linot extension can actually compare to archived latent parameters or must skip cleanly

### Engineering checks to keep in mind

- do not assume a background process is alive unless its PID and progress files are verified
- use module execution form from repo root:
  - `python -m paper_reproductions.linot2022_stab.run_reproduction ...`
  - `python -m paper_reproductions.ozalp2024_clv.run_reproduction ...`
- keep in mind that large outputs remain local-only because of `.gitignore`

## Short Resume Summary

The repo is cloned, the working focus is corrected to the last two paper reproductions, the machine now has a runnable local Python environment, the Linot 2022 staged runner was repaired, and the Linot 2022 dataset has already been generated locally.

The next real step is straightforward:

- launch and monitor Linot 2022 training

