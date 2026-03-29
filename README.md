# streaming-svd
Streaming power iteration for tracking SVD of optimizer momentum as in https://kexue.fm/archives/11673

Version by @YouJiacheng https://x.com/YouJiacheng/status/2038321307641921975

Output
```
Streaming power iteration, fp32 only
power_shift  = 0.0001
jitter1      = 1e-05
jitter2      = 1e-06
diag_floor_mult = 10.0
Evaluating gaussian...
Evaluating scaled_columns_1e4...
Evaluating controlled_spectrum_1e3...
Evaluating controlled_spectrum_1e6...
Evaluating near_rank_deficient...
Evaluating repeated_singular_values...
Evaluating correlated_columns_rho_0.999...
Evaluating low_rank_plus_spikes...
Evaluating sparseish_2pct...
Evaluating row_scale_imbalance_1e5...
Evaluating almost_duplicate_columns...
Evaluating cancellation_structure...
Evaluating tall_skinny_16384x128...
Evaluating bad_v0_random_dense...
Evaluating bad_v0_nearly_collinear...
Evaluating bad_v0_random_orth...

Single-step suite
case                         | n        | m         | ms       | gflops    | finite | orth_fro | orth_max | residual | cond_est  | cos_min | cos_mean | cos_max
-----------------------------+----------+-----------+----------+-----------+--------+----------+----------+----------+-----------+---------+----------+--------
gaussian                     | 1.64e+04 | 4096.0000 | 296.1273 | 3867.6761 | yes    | 1.75e-06 | 5.25e-06 | 1.52e-06 | 2.9867    | 1.0000  | 1.0000   | 1.0000
scaled_columns_1e4           | 1.64e+04 | 4096.0000 | 295.3890 | 3877.3435 | yes    | 3.35e-06 | 4.20e-05 | 4.14e-06 | 1.25e+04  | 1.0000  | 1.0000   | 1.0000
controlled_spectrum_1e3      | 1.64e+04 | 4096.0000 | 295.1052 | 3881.0720 | yes    | 1.60e-05 | 2.38e-05 | 3.13e-06 | 999.9917  | 1.0000  | 1.0000   | 1.0000
controlled_spectrum_1e6      | 1.64e+04 | 4096.0000 | 294.7749 | 3885.4209 | yes    | 1.98e-05 | 3.78e-05 | 3.92e-06 | 9.75e+05  | 0.9999  | 1.0000   | 1.0000
near_rank_deficient          | 1.64e+04 | 4096.0000 | 311.1447 | 3681.0027 | yes    | 3.55e-05 | 2.03e-04 | 4.70e-06 | 3.25e+07  | 0.9998  | 1.0000   | 1.0000
repeated_singular_values     | 1.64e+04 | 4096.0000 | 295.9926 | 3869.4366 | yes    | 2.16e-05 | 6.10e-05 | 3.40e-06 | 1000.4827 | 0.9999  | 1.0000   | 1.0000
correlated_columns_rho_0.999 | 1.64e+04 | 4096.0000 | 289.0539 | 3962.3218 | yes    | 1.43e-04 | 2.61e-04 | 1.04e-05 | 2829.9729 | 0.9995  | 1.0000   | 1.0004
low_rank_plus_spikes         | 1.64e+04 | 4096.0000 | 294.0648 | 3894.8035 | yes    | 5.54e-05 | 0.0017   | 1.68e-05 | 1.30e+04  | 0.9989  | 1.0000   | 1.0002
sparseish_2pct               | 1.64e+04 | 4096.0000 | 282.9093 | 4048.3810 | yes    | 1.79e-06 | 5.42e-06 | 1.56e-06 | 3.0509    | 1.0000  | 1.0000   | 1.0000
row_scale_imbalance_1e5      | 1.64e+04 | 4096.0000 | 295.3150 | 3878.3148 | yes    | 1.07e-05 | 1.90e-05 | 2.49e-06 | 86.5335   | 1.0000  | 1.0000   | 1.0000
almost_duplicate_columns     | 1.64e+04 | 4096.0000 | 285.1484 | 4016.5907 | yes    | 8.24e-05 | 0.0037   | 3.72e-05 | 1.28e+06  | 0.9975  | 1.0000   | 1.0002
cancellation_structure       | 1.64e+04 | 4096.0000 | 302.7516 | 3783.0509 | yes    | 5.43e-05 | 0.0019   | 2.88e-05 | 1.82e+05  | 0.9988  | 1.0000   | 1.0002
tall_skinny_16384x128        | 1.64e+04 | 128.0000  | 3.0289   | 183.2498  | yes    | 9.83e-07 | 1.49e-06 | 9.88e-07 | 1.1865    | 0.9999  | 1.0000   | 1.0000
bad_v0_random_dense          | 1.64e+04 | 4096.0000 | 297.8991 | 3844.6734 | yes    | 0.0019   | 0.1190   | 9.15e-06 | 9.99e+04  | 0.9381  | 1.0000   | 1.0000
bad_v0_nearly_collinear      | 1.64e+04 | 4096.0000 | 290.9164 | 3936.9539 | no     | nan      | nan      | nan      | 9.99e+04  | nan     | nan      | nan
bad_v0_random_orth           | 1.64e+04 | 4096.0000 | 298.0618 | 3842.5746 | yes    | 1.89e-05 | 2.99e-05 | 3.71e-06 | 9.99e+04  | 0.9999  | 1.0000   | 1.0000

Single-step summary
cases   | failures | mean_ms  | worst_orth_fro | worst_residual | worst_cond_est | worst_cos_min
--------+----------+----------+----------------+----------------+----------------+--------------
16.0000 | 1.0000   | 276.7302 | 0.0019         | 3.72e-05       | 3.25e+07       | 0.9381

Streaming example: drifting_gaussian
experiment        | t      | ms       | gflops    | finite | orth_fro | residual | cos_min | cos_mean | cos_max
------------------+--------+----------+-----------+--------+----------+----------+---------+----------+--------
drifting_gaussian | 0.0000 | 294.5695 | 3888.1300 | yes    | 1.74e-06 | 1.53e-06 | 1.0000  | 1.0000   | 1.0000
drifting_gaussian | 1.0000 | 312.2218 | 3668.3049 | yes    | 1.64e-06 | 1.66e-06 | 1.0000  | 1.0000   | 1.0000
drifting_gaussian | 2.0000 | 299.4207 | 3825.1345 | yes    | 1.47e-06 | 1.66e-06 | 1.0000  | 1.0000   | 1.0000
drifting_gaussian | 3.0000 | 299.7425 | 3821.0280 | yes    | 1.45e-06 | 1.68e-06 | 1.0000  | 1.0000   | 1.0000
drifting_gaussian | 4.0000 | 299.8481 | 3819.6826 | yes    | 1.42e-06 | 1.69e-06 | 1.0000  | 1.0000   | 1.0000
drifting_gaussian | 5.0000 | 299.8883 | 3819.1709 | yes    | 1.41e-06 | 1.71e-06 | 1.0000  | 1.0000   | 1.0000
drifting_gaussian | 6.0000 | 300.2795 | 3814.1950 | yes    | 1.39e-06 | 1.70e-06 | 1.0000  | 1.0000   | 1.0000
drifting_gaussian | 7.0000 | 299.1135 | 3829.0634 | yes    | 1.38e-06 | 1.70e-06 | 1.0000  | 1.0000   | 1.0000
drifting_gaussian | 8.0000 | 298.5298 | 3836.5503 | yes    | 1.39e-06 | 1.71e-06 | 1.0000  | 1.0000   | 1.0000

Streaming example: rotating_subspace
experiment        | t      | ms       | gflops    | finite | orth_fro | residual | cos_min | cos_mean | cos_max
------------------+--------+----------+-----------+--------+----------+----------+---------+----------+--------
rotating_subspace | 0.0000 | 295.2111 | 3879.6799 | yes    | 1.60e-05 | 3.12e-06 | 1.0000  | 1.0000   | 1.0000
rotating_subspace | 1.0000 | 299.3873 | 3825.5615 | yes    | 2.71e-06 | 1.96e-06 | 1.0000  | 1.0000   | 1.0000
rotating_subspace | 2.0000 | 299.2679 | 3827.0880 | yes    | 2.26e-06 | 1.95e-06 | 1.0000  | 1.0000   | 1.0000
rotating_subspace | 3.0000 | 298.7630 | 3833.5551 | yes    | 2.18e-06 | 2.00e-06 | 1.0000  | 1.0000   | 1.0000
rotating_subspace | 4.0000 | 299.2087 | 3827.8458 | yes    | 2.13e-06 | 1.99e-06 | 1.0000  | 1.0000   | 1.0000
rotating_subspace | 5.0000 | 299.4180 | 3825.1700 | yes    | 2.12e-06 | 1.98e-06 | 1.0000  | 1.0000   | 1.0000
rotating_subspace | 6.0000 | 298.9575 | 3831.0618 | yes    | 2.10e-06 | 2.02e-06 | 1.0000  | 1.0000   | 1.0000
rotating_subspace | 7.0000 | 299.4446 | 3824.8294 | yes    | 2.11e-06 | 1.99e-06 | 1.0000  | 1.0000   | 1.0000
rotating_subspace | 8.0000 | 299.6648 | 3822.0190 | yes    | 2.09e-06 | 1.98e-06 | 1.0000  | 1.0000   | 1.0000

Streaming summary
experiment        | steps  | mean_ms  | mean_gflops | worst_orth_fro | worst_residual | worst_cos_min     
------------------+--------+----------+-------------+----------------+----------------+--------------     
drifting_gaussian | 9.0000 | 300.4015 | 3813.4733   | 1.74e-06       | 1.71e-06       | 1.0000
rotating_subspace | 9.0000 | 298.8137 | 3832.9789   | 1.60e-05       | 3.12e-06       | 1.0000

Recovery from nearly-collinear V0
t      | ms       | finite | orth_fro | residual | cos_min | cos_mean | cos_max
-------+----------+--------+----------+----------+---------+----------+--------
0.0000 | 292.9585 | no     | nan      | nan      | nan     | nan      | nan
1.0000 | 286.8142 | no     | nan      | nan      | nan     | nan      | nan
2.0000 | 288.6512 | no     | nan      | nan      | nan     | nan      | nan
3.0000 | 289.9204 | no     | nan      | nan      | nan     | nan      | nan
4.0000 | 288.4125 | no     | nan      | nan      | nan     | nan      | nan
5.0000 | 289.0785 | no     | nan      | nan      | nan     | nan      | nan
6.0000 | 289.6854 | no     | nan      | nan      | nan     | nan      | nan
7.0000 | 290.4516 | no     | nan      | nan      | nan     | nan      | nan
8.0000 | 291.8094 | no     | nan      | nan      | nan     | nan      | nan
```
