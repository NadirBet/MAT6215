# Linot 2021 KS Reproduction

This folder is for a faithful reproduction path of:

- [2109.00060_reduced_manifold.pdf](c:/Users/nadir/Desktop/MAT6215/2109.00060_reduced_manifold.pdf)

## Core Claim

For the Kuramoto-Sivashinsky equation, learning dynamics on a reduced
manifold is more faithful than learning directly in the full ambient state.

## Immediate Focus

Start with the `L = 22` case and build the reproduction in paper order:

1. `Figure 2`:
   reconstruction MSE vs latent dimension `d`
2. `Figure 3` and `Figure 4`:
   latent-space NODE vs full-space NODE failure
3. `tau` study:
   `Figures 5-9`
4. latent-dimension sweep:
   `Figures 10-12`

## Data

We now use cached Linot-specific KS datasets prepared in:

- [data](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data)

The key cached files are:

- [ks_l22_n64_dt025_train.npy](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data/ks_l22_n64_dt025_train.npy)
- [ks_l22_n64_dt025_test.npy](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data/ks_l22_n64_dt025_test.npy)
- [ks_l44_n64_dt025_train.npy](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data/ks_l44_n64_dt025_train.npy)
- [ks_l44_n64_dt025_test.npy](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data/ks_l44_n64_dt025_test.npy)
- [ks_l66_n64_dt025_train.npy](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data/ks_l66_n64_dt025_train.npy)
- [ks_l66_n64_dt025_test.npy](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/data/ks_l66_n64_dt025_test.npy)

## Current Files

- [requirements.txt](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/requirements.txt)
- [REPRODUCTION_PLAN.md](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/REPRODUCTION_PLAN.md)
- [run_figure2_l22_basic.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/run_figure2_l22_basic.py)
- [run_figure2_l22_hybrid.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/run_figure2_l22_hybrid.py)
- [prepare_figure2_datasets.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/prepare_figure2_datasets.py)
- [run_figure2_all_hybrid.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/run_figure2_all_hybrid.py)
- [render_figure2_shifted.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/render_figure2_shifted.py)
- [run_figure3_l22_core.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/run_figure3_l22_core.py)

## Figure 2 Status

- `Figure 2` is complete for `L = 22, 44, 66`
- final clean outputs are in [final_figure2](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/final_figure2)
- `prepare_figure2_datasets.py` is the cache-builder for any missing domain trajectories
- `run_figure2_all_hybrid.py` produces the combined multi-domain Figure 2 plot
- `render_figure2_shifted.py` produces the manifold-centered variant with x-axis `d - d_M`
