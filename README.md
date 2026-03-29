# streaming-svd
Streaming power iteration for tracking SVD of optimizer momentum as in https://kexue.fm/archives/11673

Version by @YouJiacheng https://x.com/YouJiacheng/status/2038321307641921975

Output
```
Streaming power iteration, fp32 only
power_shift = 0.0001
jitter1     = 1e-05
jitter2     = 1e-06
diag_floor  = 10.0

Single-step suite
case                         | n         | m        | ms     | finite | orth_fro | orth_max | residual | cond_est  | cos_min | cos_mean | cos_max
-----------------------------+-----------+----------+--------+--------+----------+----------+----------+-----------+---------+----------+--------
gaussian                     | 2048.0000 | 128.0000 | 3.3364 | yes    | 1.07e-06 | 1.43e-06 | 9.42e-07 | 1.6330    | 0.9999  | 1.0000   | 1.0000
scaled_columns_1e4           | 2048.0000 | 128.0000 | 3.0770 | yes    | 2.40e-06 | 8.46e-06 | 1.61e-06 | 1.03e+04  | 0.9999  | 1.0000   | 1.0000
controlled_spectrum_1e3      | 2048.0000 | 128.0000 | 3.0638 | yes    | 2.48e-05 | 3.37e-05 | 1.93e-06 | 1000.0031 | 0.9999  | 1.0000   | 1.0000
controlled_spectrum_1e6      | 2048.0000 | 128.0000 | 3.0580 | yes    | 3.09e-05 | 5.25e-05 | 2.93e-06 | 9.99e+05  | 0.9999  | 0.9999   | 1.0000
near_rank_deficient          | 2048.0000 | 128.0000 | 5.2790 | yes    | 1.25e-04 | 9.53e-04 | 4.61e-06 | 1.97e+06  | 0.9995  | 0.9999   | 1.0000
repeated_singular_values     | 2048.0000 | 128.0000 | 1.8520 | yes    | 2.66e-05 | 6.12e-05 | 2.38e-06 | 1000.1260 | 0.9999  | 1.0000   | 1.0000
correlated_columns_rho_0.999 | 2048.0000 | 128.0000 | 1.8463 | yes    | 1.92e-05 | 1.42e-04 | 9.29e-06 | 578.5292  | 0.9999  | 1.0000   | 1.0000
low_rank_plus_spikes         | 2048.0000 | 128.0000 | 4.1434 | yes    | 8.63e-05 | 8.90e-04 | 7.46e-06 | 1790.3322 | 0.9996  | 1.0000   | 1.0000
sparseish_2pct               | 2048.0000 | 128.0000 | 1.9526 | yes    | 1.08e-06 | 1.61e-06 | 9.68e-07 | 2.0494    | 0.9999  | 1.0000   | 1.0000
row_scale_imbalance_1e5      | 2048.0000 | 128.0000 | 1.8602 | yes    | 2.49e-06 | 3.70e-06 | 8.90e-07 | 5.7913    | 0.9999  | 1.0000   | 1.0000
almost_duplicate_columns     | 2048.0000 | 128.0000 | 1.7877 | yes    | 1.60e-05 | 1.32e-04 | 1.08e-05 | 1.48e+05  | 0.9999  | 1.0000   | 1.0000
cancellation_structure       | 2048.0000 | 128.0000 | 1.7960 | yes    | 2.25e-04 | 0.0023   | 9.32e-06 | 2.21e+04  | 0.9988  | 0.9999   | 1.0000
tall_skinny_16384x128        | 1.64e+04  | 128.0000 | 1.8875 | yes    | 9.83e-07 | 1.49e-06 | 9.88e-07 | 1.1865    | 0.9999  | 1.0000   | 1.0000
bad_v0_random_dense          | 2048.0000 | 128.0000 | 1.0330 | yes    | 4.25e-05 | 2.01e-04 | 2.39e-06 | 1.00e+05  | 0.9999  | 1.0000   | 1.0000
bad_v0_nearly_collinear      | 2048.0000 | 128.0000 | 1.8353 | yes    | 0.4305   | 0.6407   | 4.04e-04 | 1.00e+05  | 0.0147  | 0.8089   | 1.0050
bad_v0_random_orth           | 2048.0000 | 128.0000 | 1.9577 | yes    | 2.78e-05 | 5.12e-05 | 2.47e-06 | 1.00e+05  | 0.9999  | 1.0000   | 1.0000

Single-step summary
cases   | failures | mean_ms | worst_orth_fro | worst_residual | worst_cond_est | worst_cos_min
--------+----------+---------+----------------+----------------+----------------+--------------
16.0000 | 0.0000   | 2.4854  | 0.4305         | 4.04e-04       | 1.97e+06       | 0.0147

Streaming example: drifting_gaussian
experiment        | t      | ms     | finite | orth_fro | residual | cos_min | cos_mean | cos_max
------------------+--------+--------+--------+----------+----------+---------+----------+--------
drifting_gaussian | 0.0000 | 2.0585 | yes    | 1.09e-06 | 1.00e-06 | 0.9999  | 1.0000   | 1.0000
drifting_gaussian | 1.0000 | 2.0368 | yes    | 1.06e-06 | 9.77e-07 | 0.9999  | 1.0000   | 1.0000
drifting_gaussian | 2.0000 | 1.8374 | yes    | 1.06e-06 | 9.97e-07 | 0.9999  | 1.0000   | 1.0000
drifting_gaussian | 3.0000 | 1.8941 | yes    | 1.03e-06 | 9.88e-07 | 0.9999  | 1.0000   | 1.0000
drifting_gaussian | 4.0000 | 2.1611 | yes    | 1.01e-06 | 9.68e-07 | 0.9999  | 1.0000   | 1.0000
drifting_gaussian | 5.0000 | 1.9930 | yes    | 1.04e-06 | 1.03e-06 | 0.9999  | 1.0000   | 1.0000
drifting_gaussian | 6.0000 | 2.0955 | yes    | 9.95e-07 | 9.87e-07 | 0.9999  | 1.0000   | 1.0000
drifting_gaussian | 7.0000 | 1.8809 | yes    | 1.03e-06 | 1.05e-06 | 0.9999  | 1.0000   | 1.0000
drifting_gaussian | 8.0000 | 1.3371 | yes    | 1.01e-06 | 1.01e-06 | 0.9999  | 1.0000   | 1.0000

Streaming example: rotating_subspace
experiment        | t      | ms     | finite | orth_fro | residual | cos_min | cos_mean | cos_max
------------------+--------+--------+--------+----------+----------+---------+----------+--------
rotating_subspace | 0.0000 | 1.9175 | yes    | 2.30e-05 | 1.74e-06 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 1.0000 | 1.8616 | yes    | 1.58e-06 | 9.13e-07 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 2.0000 | 1.1650 | yes    | 1.15e-06 | 9.90e-07 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 3.0000 | 1.8857 | yes    | 1.06e-06 | 9.61e-07 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 4.0000 | 2.0277 | yes    | 1.05e-06 | 9.48e-07 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 5.0000 | 2.0910 | yes    | 1.04e-06 | 1.02e-06 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 6.0000 | 1.8805 | yes    | 1.02e-06 | 1.05e-06 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 7.0000 | 0.7444 | yes    | 1.04e-06 | 9.57e-07 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 8.0000 | 1.2037 | yes    | 1.03e-06 | 1.02e-06 | 0.9999  | 1.0000   | 1.0000

Streaming summary
experiment        | steps  | mean_ms | worst_orth_fro | worst_residual | worst_cos_min
------------------+--------+---------+----------------+----------------+--------------
drifting_gaussian | 9.0000 | 1.9216  | 1.09e-06       | 1.05e-06       | 0.9999
rotating_subspace | 9.0000 | 1.6419  | 2.30e-05       | 1.74e-06       | 0.9999
rotating_subspace | 4.0000 | 2.0277 | yes    | 1.05e-06 | 9.48e-07 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 5.0000 | 2.0910 | yes    | 1.04e-06 | 1.02e-06 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 6.0000 | 1.8805 | yes    | 1.02e-06 | 1.05e-06 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 7.0000 | 0.7444 | yes    | 1.04e-06 | 9.57e-07 | 0.9999  | 1.0000   | 1.0000
rotating_subspace | 8.0000 | 1.2037 | yes    | 1.03e-06 | 1.02e-06 | 0.9999  | 1.0000   | 1.0000

Streaming summary
experiment        | steps  | mean_ms | worst_orth_fro | worst_residual | worst_cos_min
------------------+--------+---------+----------------+----------------+--------------
drifting_gaussian | 9.0000 | 1.9216  | 1.09e-06       | 1.05e-06       | 0.9999
rotating_subspace | 9.0000 | 1.6419  | 2.30e-05       | 1.74e-06       | 0.9999

Streaming summary
experiment        | steps  | mean_ms | worst_orth_fro | worst_residual | worst_cos_min
------------------+--------+---------+----------------+----------------+--------------
drifting_gaussian | 9.0000 | 1.9216  | 1.09e-06       | 1.05e-06       | 0.9999
rotating_subspace | 9.0000 | 1.6419  | 2.30e-05       | 1.74e-06       | 0.9999
------------------+--------+---------+----------------+----------------+--------------
drifting_gaussian | 9.0000 | 1.9216  | 1.09e-06       | 1.05e-06       | 0.9999
rotating_subspace | 9.0000 | 1.6419  | 2.30e-05       | 1.74e-06       | 0.9999
drifting_gaussian | 9.0000 | 1.9216  | 1.09e-06       | 1.05e-06       | 0.9999
rotating_subspace | 9.0000 | 1.6419  | 2.30e-05       | 1.74e-06       | 0.9999
rotating_subspace | 9.0000 | 1.6419  | 2.30e-05       | 1.74e-06       | 0.9999

Recovery from nearly-collinear V0
Recovery from nearly-collinear V0
t      | ms     | finite | orth_fro | residual | cos_min | cos_mean | cos_max
-------+--------+--------+----------+----------+---------+----------+--------
0.0000 | 1.3173 | yes    | 0.4080   | 3.71e-04 | 0.0362  | 0.8252   | 1.0007
1.0000 | 2.0379 | yes    | 1.40e-06 | 9.50e-07 | 0.9999  | 1.0000   | 1.0000
2.0000 | 1.5047 | yes    | 1.11e-06 | 1.05e-06 | 0.9999  | 1.0000   | 1.0000
3.0000 | 1.8587 | yes    | 1.04e-06 | 9.81e-07 | 0.9999  | 0.9999   | 1.0000
4.0000 | 2.2258 | yes    | 1.02e-06 | 9.59e-07 | 0.9999  | 0.9999   | 1.0000
5.0000 | 1.8941 | yes    | 1.04e-06 | 1.04e-06 | 0.9999  | 0.9999   | 1.0000
6.0000 | 1.9183 | yes    | 1.01e-06 | 1.09e-06 | 0.9999  | 0.9999   | 1.0000
7.0000 | 1.7846 | yes    | 1.02e-06 | 1.08e-06 | 0.9999  | 0.9999   | 1.0000
8.0000 | 1.8438 | yes    | 1.02e-06 | 1.09e-06 | 0.9999  | 1.0000   | 1.0000
```
