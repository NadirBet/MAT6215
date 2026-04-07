# Wake SINDy Results

## Dataset Used

Source:

- [fixed_cylinder_atRe100](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/data/fixed_cylinder_atRe100)

Parsed with:

- [zenodo_re100.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/zenodo_re100.py)

Verified metadata:

- `Re = 100`
- `Nt = 201`
- `N_nodes = 82872`
- time range `400.0 -> 420.0`
- `dt ≈ 0.1`
- domain approximately `x in [-40, 120]`, `y in [-60, 60]`

Interpretation:

- this is a post-transient vortex-shedding wake dataset
- it is well-suited for on-attractor wake dynamics
- it is less ideal for Brunton's off-attractor shift-mode identification because the transient from unstable steady wake to mean wake is not obviously present

## What Was Run

Script:

- [run_brunton2016_wake.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/run_brunton2016_wake.py)

Method:

1. load full velocity snapshots
2. build state vectors by concatenating `u` and `v`
3. compute POD using the snapshot method
4. project onto the first `2` and `3` POD coordinates
5. fit SINDy with a polynomial library up to degree `5`
6. sweep STLSQ thresholds and keep the best stable model by rollout error

## POD Summary

From [wake_sindy_results.json](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/results/wake_sindy_results.json):

- mode 1 energy fraction: `0.4989`
- mode 2 energy fraction: `0.4559`
- mode 3 energy fraction: `0.0160`
- first 2 modes cumulative energy: `0.9548`
- first 3 modes cumulative energy: `0.9708`

Interpretation:

- the first two modes dominate, exactly as expected for a periodic cylinder wake
- the third mode is present but much weaker, which is consistent with this dataset being mostly on the limit cycle rather than containing a strong transient shift-mode excursion

Figure:

- [wake_pod_energy.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_pod_energy.png)

## 3D Wake SINDy

Best threshold:

- `0.0003`

Best-model metrics:

- active terms: `25`
- derivative RMSE: `0.0280`
- rollout RMSE: `0.0237`
- stable rollout: `true`

Outputs:

- coefficient table: [wake_sindy_coefficients_3d.csv](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/results/wake_sindy_coefficients_3d.csv)
- trajectory figure: [wake_figure8_like_3d.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure8_like_3d.png)
- paper-style figure: [wake_figure8_paper_style.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure8_paper_style.png)
- table-style figure: [wake_table5_style.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_table5_style.png)

Main structural result:

- the selected `3D` model uses **only constant, linear, and quadratic terms**
- all cubic, quartic, and quintic terms are zero in the selected model

This is the closest match to Brunton's Table 5 that we currently have.

Important caveat:

- the coefficient values do **not** match Table 5 numerically, because our POD coordinates are not the same coordinates used in the paper
- however, the qualitative sparsity pattern is encouraging: higher-order terms are not needed in the selected `3D` model

## 2D Wake SINDy

Best threshold:

- `0.00003`

Best-model metrics:

- active terms: `15`
- derivative RMSE: `0.0585`
- rollout RMSE: `0.1994`
- stable rollout: `true`

Output:

- coefficient table: [wake_sindy_coefficients_2d.csv](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/results/wake_sindy_coefficients_2d.csv)

Main structural result:

- the selected `2D` model includes **cubic terms**

This is actually very interesting because it mirrors Brunton's discussion:

- with only on-attractor wake information, SINDy tends to identify a Hopf-normal-form-like model with cubic nonlinearities
- recovering the clean quadratic mean-field structure requires the third coordinate associated with the shift mode

## What This Means

This first-pass reproduction already says something useful:

1. the free `Re = 100` DNS dataset is good enough to recover the dominant low-dimensional wake geometry
2. a `3D` SINDy model on the first three POD coordinates can fit the wake trajectory very accurately
3. the selected `3D` model stays purely quadratic even though the library includes terms up to degree `5`
4. the selected `2D` model uses cubic terms, which is qualitatively consistent with Brunton's warning about insufficient off-attractor information

## Figure 7 Analogue

I also generated a Figure-7-like panel:

- [wake_figure7_analogue.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure7_analogue.png)

What is data-driven in this figure:

- the `A` shedding state
- the `B` mean flow
- the first two POD mode fields
- the third POD mode field
- the reduced-coordinate trajectory

What is approximate:

- `C*` is a shift-like reconstruction built from the third reduced coordinate, not Brunton's exact unstable fixed point
- the left-panel slow manifold surface is an illustrative cone anchored to the actual POD trajectory, not a recovered invariant manifold

So this figure is a useful analogue, but it should be presented as a close interpretation rather than an exact reproduction of Brunton's Figure 7.

## What Is Still Missing Relative To Brunton

We have **not** yet reproduced the paper exactly because:

- this is not the original immersed-boundary DNS dataset
- the data appears largely post-transient
- we do not have an explicit unstable steady wake state
- we have not yet constructed the exact shift mode used in the paper's mean-field model

So the current status is:

- good reproduction of the **wake SINDy story**
- partial reproduction of the **Table 5 quadratic structure**
- not yet a full faithful reproduction of the **off-attractor mean-field wake model**

## Paper-Style Figures Added

I also generated paper-style figure files from the current fit:

- [wake_figure8_paper_style.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure8_paper_style.png)
- [wake_table5_style.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_table5_style.png)

And two clearly labeled analogues for the Brunton off-attractor figures:

- [wake_figure9_analogue.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure9_analogue.png)
- [wake_figure10_analogue.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure10_analogue.png)

These analogues are useful for visualization, but they are **not** the same as Brunton's true DNS-vs-identified transient comparisons because our dataset does not include the corresponding off-attractor DNS trajectories.

## Saved Artifacts

- summary json: [wake_sindy_results.json](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/results/wake_sindy_results.json)
- POD coefficients and rollouts: [wake_pod_coefficients.npz](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/results/wake_pod_coefficients.npz)
- 3D coefficient table: [wake_sindy_coefficients_3d.csv](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/results/wake_sindy_coefficients_3d.csv)
- 2D coefficient table: [wake_sindy_coefficients_2d.csv](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/results/wake_sindy_coefficients_2d.csv)
- energy figure: [wake_pod_energy.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_pod_energy.png)
- coefficient time series: [wake_pod_timeseries.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_pod_timeseries.png)
- Figure 8 style comparison: [wake_figure8_like_3d.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure8_like_3d.png)
- Paper-style Figure 8: [wake_figure8_paper_style.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure8_paper_style.png)
- Table 5 style figure: [wake_table5_style.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_table5_style.png)
- Figure 7 analogue: [wake_figure7_analogue.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure7_analogue.png)
- Figure 9 analogue: [wake_figure9_analogue.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure9_analogue.png)
- Figure 10 analogue: [wake_figure10_analogue.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/figures/wake_figure10_analogue.png)
