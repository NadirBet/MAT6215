# Reproduction Comparison Sheet

This note is a compact side-by-side summary of the two active KS paper
reproductions in this repo:

- [Linot 2021 reduced manifold paper](/c:/Users/nadir/Desktop/MAT6215/2109.00060_reduced_manifold.pdf)
- [Park 2024 Jacobian matching paper](/c:/Users/nadir/Desktop/MAT6215/2411.06311v2.pdf)

It is intentionally short and focused on:

1. what the paper claims
2. the article figures/results we targeted
3. what our current local reproduction produced

## Linot 2021

Paper reference:
- [2109.00060_reduced_manifold.pdf](/c:/Users/nadir/Desktop/MAT6215/2109.00060_reduced_manifold.pdf)

Local outputs:
- [figure2_all_hybrid_d.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/final_figure2/figure2_all_hybrid_d.png)
- [figure2_all_hybrid_d_minus_dm.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/final_figure2/figure2_all_hybrid_d_minus_dm.png)
- [figure3_l22_core.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/figures/figure3_l22_core.png)
- [figure4_l22_core.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/figures/figure4_l22_core.png)
- [figure5_like_l22_latent.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/figures/figure5_like_l22_latent.png)

Article targets:
- Figure 2: low-dimensional elbows near `d_M ~= 8, 18, 28` for `L = 22, 44, 66`
- Figures 3-4: latent-space NODE should stay closer to the attractor than the full physical-space and Fourier-space NODEs

Current reproduction verdict:
- Figure 2 is qualitatively reproduced across all three domains
- The `L=22` dynamics comparison is directionally reproduced: latent is best, physical is worse, Fourier fails badly

Key comparison:

| Quantity | Paper-side target | Our result |
| --- | --- | --- |
| Figure 2 elbow, `L=22` | around `d=8` | strong drop by `d=8` |
| Figure 2 elbow, `L=44` | around `d=18` | strong drop by `d=18` |
| Figure 2 elbow, `L=66` | around `d=28` | strong drop in the `28 -> 32` window |
| `L=22` latent rollout quality | latent should beat full-space baselines | latent MSE `1.4510`, physical `2.2702`, Fourier `230.7009` |
| Attractor scale | latent should stay close to true | true energy `83.13`, latent `85.14`, physical `92.29`, Fourier `14626.79` |

Main Linot artifacts:

![Linot Figure 2](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/final_figure2/figure2_all_hybrid_d.png)

![Linot Figure 3 Core](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/figures/figure3_l22_core.png)

![Linot Figure 4 Core](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/figures/figure4_l22_core.png)

![Linot Figure 5-like Latent View](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/figures/figure5_like_l22_latent.png)

Supporting files:
- [figure2_all_hybrid_d.json](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/final_figure2/figure2_all_hybrid_d.json)
- [figure3_l22_core_summary.json](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/results/figure3_l22_core_summary.json)
- [figure5_like_l22_latent.json](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/linot2021_ks/results/figure5_like_l22_latent.json)

## Park 2024

Paper reference:
- [2411.06311v2.pdf](/c:/Users/nadir/Desktop/MAT6215/2411.06311v2.pdf)

Article figure extracts already saved locally:
- [figure8_page38.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/reference/figure8_page38.png)
- [table5_page42.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/reference/table5_page42.png)
- [page35_figure3_loss.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/reference/page35_figure3_loss.png)

Local outputs:
- [park2024_figure8_ks.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/figures/park2024_figure8_ks.png)
- [park2024_ks_loss.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/figures/park2024_ks_loss.png)
- [park2024_ks_mse_probe_lr1e4.png](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/figures/park2024_ks_mse_probe_lr1e4.png)

Article targets:
- Figure 8: side-by-side true / MSE / JAC KS comparison
- Table 5: JAC Lyapunov spectrum should be much closer to true than MSE

Important note:
- The paper's absolute `Lambda_true` scale in Table 5 does not match the released KS code path exactly
- For the local reproduction, the fairest comparison is **model vs our local true spectrum**

Strongest completed Park result so far:

| Quantity | Article-side target | Our completed local result |
| --- | --- | --- |
| Leading true exponent | positive chaotic instability | `lambda_1(true) = 0.07842` |
| MSE leading exponent | positive but reduced vs true | `lambda_1(MSE) = 0.000031` |
| JAC leading exponent | close to true | `lambda_1(JAC) = 0.07374` |
| 15-exponent MAE vs true | JAC << MSE | MSE `0.03394`, JAC `0.00654` |

Figure-8-style rollout statistics from the completed run:

| System | rollout std | max `|u|` |
| --- | --- | --- |
| true | `1.2457` | `3.2321` |
| mse | `1.1520` | `2.9618` |
| jac | `1.3201` | `4.4844` |

Reference images from the Park paper:

![Park Paper Figure 8](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/reference/figure8_page38.png)

![Park Paper Table 5](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/reference/table5_page42.png)

Our current Park outputs:

![Park Local Figure 8](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/figures/park2024_figure8_ks.png)

![Park Local Loss](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/figures/park2024_ks_loss.png)

![Park Tuned MSE Probe](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/figures/park2024_ks_mse_probe_lr1e4.png)

Current paper-faithful rerun status:
- the tuned Park rerun is still active in [park2024_ks_paper_eval_progress.json](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/data/park2024_ks_paper_eval_progress.json)
- tuned MSE checkpoint selected so far: epoch `50`
- tuned MSE best test loss: `0.02595`
- tuned MSE best one-step relative error: `0.06251`
- latest saved JAC checkpoint in the active rerun: epoch `325`, best test loss `37.36496`

Supporting files:
- [park2024_ks_progress.json](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/data/park2024_ks_progress.json)
- [park2024_ks_mse_history_paper_eval.json](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/data/park2024_ks_mse_history_paper_eval.json)
- [park2024_ks_paper_eval_progress.json](/c:/Users/nadir/Desktop/MAT6215/paper_reproductions/park2024_ks/data/park2024_ks_paper_eval_progress.json)

## Bottom Line

- **Linot 2021**: the main reduced-manifold claim is already reproduced well enough to compare latent vs full-space behavior locally.
- **Park 2024**: the JAC-vs-MSE dynamical-fidelity claim is already visible in the completed run, and the tuned paper-faithful rerun is refining that comparison now.
