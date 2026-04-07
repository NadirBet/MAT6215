# Brunton 2016 Wake Reproduction

This folder is for a wake-specific reproduction path based on the SINDy paper in:

- [1509.03580_sindy.pdf](c:/Users/nadir/Desktop/MAT6215/1509.03580_sindy.pdf)
- [1509.03580.pdf](c:/Users/nadir/Desktop/MAT6215/articles/1509.03580.pdf)

## Goal

Reproduce the cylinder-wake part of the paper as faithfully as practical:

- obtain a `Re = 100` cylinder-wake dataset
- extract POD / shift-mode coordinates
- recover the low-dimensional wake model with sparse regression
- compare against the wake coefficient table and phase-space figures in the paper

## Reproduction Standard

For this project, "good enough" means:

- same physical regime: `Re = 100` 2D cylinder wake
- same qualitative dynamical structure:
  - unstable steady wake
  - mean flow distortion
  - limit-cycle shedding
- same reduced-order modeling workflow:
  - snapshots
  - POD / reduced coordinates
  - sparse identification
- same qualitative conclusions as the paper

It does **not** require the exact original Brunton DNS snapshots, as long as we are explicit that our dataset is a close match rather than the exact paper dataset.

## Current Data Source

Primary path now on disk:

- `Zenodo` dataset: fixed circular cylinder at `Re = 100`
- raw file: [fixed_cylinder_atRe100](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/data/fixed_cylinder_atRe100)
- official parser helper: [text_flow_zenodo.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/text_flow_zenodo.py)

Why this is currently the best practical choice:

- free and already downloaded
- same benchmark family: `2D`, `Re = 100`, vortex shedding around a fixed cylinder
- provides full-state flow data, not just force coefficients
- much cheaper than generating new CFD data

Fallback path if we later need more control:

- `hydrogym` cylinder-wake environment at `Re = 100`

## Files In This Folder

- [DATA_ACQUISITION_PLAN.md](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/DATA_ACQUISITION_PLAN.md): concrete steps and cost estimates
- [zenodo_re100.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/zenodo_re100.py): local loader for the downloaded wake dataset
- [inspect_zenodo_re100.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/inspect_zenodo_re100.py): lightweight inspection script for metadata and mesh consistency

## Target Outputs

Eventually this folder should contain:

- a data-generation script for uncontrolled `Re = 100` wake snapshots
- POD / coordinate extraction code
- SINDy identification code
- reproduced wake figures
- reproduced sparse coefficient table
