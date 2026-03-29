# streaming-svd
Streaming power iteration for tracking SVD of optimizer momentum as in https://kexue.fm/archives/11673

Version by @YouJiacheng https://x.com/YouJiacheng/status/2038321307641921975

Output

```
Hard Test Suite (16384x4096)
Evaluating large_gaussian_16384x4096...
Evaluating large_ill_conditioned_1e7_16384x4096...
Evaluating large_staircase_16384x4096...
Evaluating large_cliff_128_16384x4096...
Evaluating large_low_rank_noise_16384x4096...
Evaluating large_sparseish_1pct_16384x4096...
case                                 | n        | m         | ms       | gflops    | finite | orth_fro | residual
-------------------------------------+----------+-----------+----------+-----------+--------+----------+---------
large_gaussian_16384x4096            | 1.64e+04 | 4096.0000 | 293.9173 | 1870.4437 | yes    | 1.72e-06 | 1.51e-06
large_ill_conditioned_1e7_16384x4096 | 1.64e+04 | 4096.0000 | 298.3996 | 1842.3477 | yes    | 2.09e-05 | 4.06e-06
large_staircase_16384x4096           | 1.64e+04 | 4096.0000 | 293.8350 | 1870.9675 | yes    | 1.48e-05 | 2.96e-06
large_cliff_128_16384x4096           | 1.64e+04 | 4096.0000 | 294.0274 | 1869.7436 | yes    | 4.96e-05 | 7.79e-06
large_low_rank_noise_16384x4096      | 1.64e+04 | 4096.0000 | 295.7725 | 1858.7119 | yes    | 4.64e-05 | 7.55e-06
large_sparseish_1pct_16384x4096      | 1.64e+04 | 4096.0000 | 282.7830 | 1944.0909 | yes    | 1.69e-06 | 1.51e-06

Streaming Hard Case: Ill-Conditioned (1e7)
t      | ms       | gflops    | orth_fro | residual
-------+----------+-----------+----------+---------
0.0000 | 295.0673 | 1863.1540 | 2.09e-05 | 4.06e-06
1.0000 | 303.9845 | 1808.4997 | 1.69e-06 | 1.77e-06
2.0000 | 302.0260 | 1820.2270 | 1.42e-06 | 1.70e-06
3.0000 | 304.9995 | 1802.4809 | 1.36e-06 | 1.71e-06
4.0000 | 302.5721 | 1816.9413 | 1.35e-06 | 1.66e-06
5.0000 | 303.9028 | 1808.9860 | 1.33e-06 | 1.64e-06
6.0000 | 303.6850 | 1810.2832 | 1.33e-06 | 1.63e-06
7.0000 | 302.9157 | 1814.8805 | 1.33e-06 | 1.60e-06
8.0000 | 305.9985 | 1796.5962 | 1.33e-06 | 1.62e-06
9.0000 | 302.7638 | 1815.7911 | 1.33e-06 | 1.66e-06
```
