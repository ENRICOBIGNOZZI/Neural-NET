# SVD equal-compute falsification protocol

The structural-bridge experiment asks whether target-free truncated-SVD recompression can realize a low-rank model without the basis-dependence of deleting learned factor coordinates. This protocol then applies the harder compute test.

For each target rank r in {2,3,4}, the compact-from-start control is obtained by SVD-recompressing the identical initial rank-8 network to rank r. At checkpoint t the wide path has trained rank 8 for t steps, is SVD-recompressed to r, and trains for a further horizon H. The compact control is given

`ceil((C_8 * t + C_r * H) / C_r)`

steps, where `C_r` is the analytic factorized-matmul MAC proxy. The ceiling deliberately gives the compact control weakly more proxy compute.

The primary statistic is

`R_wide_then_SVD - R_compact_equal_compute`.

Negative values favor temporary overparameterization; positive values favor compact-from-start. Checkpoint zero is an invariant check because both paths start from the same SVD-compressed network. We also record same-step structural damage and the singular-value tail fraction at contraction.

This test is target-free at the contraction step. Its remaining limitation is that compute is matched with an analytic matmul MAC proxy rather than profiler-measured training FLOPs or accelerator wall time.
