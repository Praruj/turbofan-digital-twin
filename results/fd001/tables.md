### RUL on the official held-out test engines (n = 100)

| Method | RMSE (95% CI) | MAE (95% CI) | PHM08 score (95% CI) | 80% interval coverage | median width |
|---|---|---|---|---|---|
| Physics only (twin) | 16.4 [13.8, 18.9] | 12.9 [10.9, 14.9] | 483 [258, 775] | 74% | 32 |
| ML only (GBM) | 13.8 [11.2, 16.3] | 10.0 [8.1, 11.9] | 293 [179, 434] | 80% | 29 |
| Hybrid (GBM + twin) | 12.3 [10.0, 14.3] | 8.9 [7.2, 10.5] | 221 [140, 325] | 65% | 17 |

| Paired difference in RMSE | estimate (95% CI) |
|---|---|
| hybrid-ml | -1.58 [-3.90, 0.87] |
| hybrid-physics | -4.18 [-5.76, -2.60] |
| ml-physics | -2.61 [-5.52, 0.43] |

### Data scarcity: test RMSE vs number of training engines

| training engines | subsets | Physics only (twin) | ML only (GBM) | Hybrid (GBM + twin) |
|---|---|---|---|---|
| 5 | 5 | 20.9 (18.6-23.5) | 25.7 (23.4-27.2) | 20.2 (18.7-21.9) |
| 10 | 5 | 16.6 (16.1-17.1) | 20.5 (18.0-24.5) | 15.1 (13.9-16.4) |
| 20 | 5 | 17.1 (16.9-17.3) | 16.3 (15.5-18.1) | 14.5 (13.2-15.8) |
| 50 | 5 | 16.6 (16.4-16.7) | 14.4 (13.8-15.1) | 12.5 (12.0-13.0) |
| 100 | 1 | 16.4 | 13.8 | 12.3 |

### Physics validation

| Sensor | drift in data (start → end of life) | model drift at mean failure damage |
|---|---|---|
| T24 | +1.214 | +1.328 |
| T30 | +14.539 | +13.036 |
| T50 | +24.563 | +21.233 |
| P30 | -2.320 | -1.529 |
| Ps30 | +0.743 | +0.385 |
| phi | -1.975 | -2.191 |
| BPR | +0.097 | +0.145 |
| W31 | -0.461 | -0.659 |
| W32 | -0.276 | -0.396 |

Sign agreement: 9/9 sensors.

| Subspace (fit to train, scored on test engines) | variance explained |
|---|---|
| physics_1dof | 0.418 |
| physics_2dof | 0.481 |
| pca_1dof | 0.499 |
| pca_2dof | 0.562 |

Lifetimes: real 206 ± 46, simulated 206 ± 40 cycles; KS D = 0.084, p = 0.58.

Parameter recovery on simulated engines: kappa_lpc 0.293 → 0.295, kappa_hpc 0.414 → 0.415, mach_30 0.950 → 0.950, fail_mean 0.027 → 0.028, dir_eta 0.552 → 0.554; damage RMSE 0.0006 (2% of failure damage); RUL RMSE 25.1; 80% interval coverage 76%.
