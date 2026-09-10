# Temporary-width break-even protocol

This experiment answers a stricter question than structural safety. A contraction can preserve the state learned by a wide model and still be a bad use of compute if the final compact architecture would have learned as well or better from the beginning.

For each target rank `r in {2,3,4}` we compare one wide rank-8 path against two compact controls. The **matched compact** control is the target-free truncated-SVD recompression of the identical initial wide model. The **native compact** control is an independently initialized rank-r model using the same architecture and optimizer. The matched control is the cleaner causal comparison because checkpoint zero is exactly identical; the native control answers the conventional “why not train the small model from scratch?” question.

At checkpoint `t`, the wide path has trained rank 8 for `t` full-batch SGD updates, is SVD-recompressed to rank `r`, and then trains for horizon `H`. Let `C_8` and `C_r` be the analytic factorized-matmul MAC proxies. The equal-compute compact budget is

`N_r(t,H) = ceil((C_8 * t + C_r * H) / C_r)`.

The compact control is evaluated both after the same number of updates `K=t+H` and after `N_r(t,H)` updates. This gives the exact decomposition

`equal-compute gap = compute opportunity gain - discovery advantage`,

where

`discovery advantage = R_compact(K) - R_wide_then_SVD(t,H)`

and

`compute opportunity gain = R_compact(K) - R_compact(N_r)`.

Temporary width wins at equal compute if and only if discovery advantage exceeds compute opportunity gain. The script asserts this identity numerically for every record.

The experiment also records SVD tail energy, same-step damage relative to continuing the dense checkpoint, parameter reduction, and both matched/native win fractions. Per-checkpoint confidence intervals are computed across independent seeds. For the overall post-initialization summary, checkpoints are first averaged within seed and the confidence interval is then computed across seed means; repeated checkpoints from the same seed are **not** treated as independent observations.

This is still a controlled small-network experiment. Compute uses an analytic matmul proxy rather than profiler-measured training FLOPs, and there is no claim yet that the result transfers to accelerator wall time or modern large architectures.