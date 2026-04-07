# Park 2024 KS Reference Targets

This file collects the exact paper-side targets we want to compare our local
run against when [run_park2024_ks.py](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/run_park2024_ks.py)
finishes.

## Main Paper Targets

- **Figure 8**
  Solution plot of the Kuramoto-Sivashinsky system with:
  - `127` inner nodes
  - `c = 0.4`
  - columns: `True`, `MSE`, `JAC`

- **Table 5, KS row**
  First `15` Lyapunov exponents for:
  - true KS
  - MSE Neural ODE
  - JAC Neural ODE

## Exact KS Lyapunov Targets From Table 5

`Lambda_true`

```text
[0.3036, 0.2733, 0.2592, 0.2257, 0.2050, 0.1888, 0.1649, 0.1496,
 0.1288, 0.1128, 0.0992, 0.0776, 0.0646, 0.0492, 0.0342]
```

`Lambda_mse`

```text
[0.1652, 0.1647, 0.1540, 0.1524, 0.1443, 0.1411, 0.1336, 0.1262,
 0.1236, 0.1143, 0.1141, 0.1091, 0.1045, 0.0971, 0.0985]
```

`Lambda_jac`

```text
[0.2904, 0.2622, 0.2293, 0.1990, 0.1701, 0.1584, 0.1320, 0.1071,
 0.0912, 0.0724, 0.0591, 0.0442, 0.0306, 0.0157, 0.0023]
```

## Exact Figure 8 Caption

```text
Figure 8: Solution plot of Kuramoto-Sivashinksy system when number of
inner nodes is 127 and c=0.4 (see [BW14] for the parameter c). True solution
(left), solution of the Neural ODE with mean squared loss (1) (center column),
solution of the Neural ODE trained with Jacobian-matching loss defined in
(3) (right)
```

## Exported Comparison Images

These images were exported from the paper PDF so we can compare visually:

- [figure8_page38.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/reference/figure8_page38.png)
- [table5_page42.png](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/reference/table5_page42.png)

If you want tighter crops later, we can crop these exported pages down to just
the figure/table panels.

## Repo-Side KS Artifacts

The downloaded `stacNODE` repo also contains the saved KS model weights:

- [best_model.pth](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/stacNODE-master/plot/Vector_field/KS/MLP_MSE_fullbatch/best_model.pth)
- [best_model.pth](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/stacNODE-master/plot/Vector_field/KS/MLP_Jacobian_fullbatch/best_model.pth)
- [lowest_testloss_model.pth](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/stacNODE-master/plot/Vector_field/KS/MLP_Jacobian_fullbatch/lowest_testloss_model.pth)

Those are useful later if we want to compare our learned model behavior to the
authors' saved KS checkpoints, not just to the paper PDF.
