# SVD equal-compute falsification: first result and next diagnostic

## Result

The first 10-seed equal-compute SVD experiment is a clean negative result for the naive claim that temporary overparameterization is automatically useful once contraction becomes safe.

The setup uses a rank-8 contractible MLP, teacher rank 2, target ranks 2/3/4, full-batch SGD, zero label noise, checkpoints 0/20/.../120, and a 40-step post-contraction horizon. The compact matched control is the target-free truncated-SVD recompression of the identical initial wide network. Its number of updates is increased to match the analytic factorized-matmul MAC cost of the wide-then-small path; the ceiling weakly favors compact-from-start.

For the primary statistic

`R_wide_then_SVD - R_compact_equal_compute`,

negative values favor temporary width and positive values favor compact-from-start. At **every nonzero checkpoint for every target rank**, the mean gap is positive and its checkpoint-wise 95% interval across the 10 seeds is also positive. For example:

| rank | checkpoint | mean equal-compute gap | 95% interval | compact extra steps |
|---:|---:|---:|---:|---:|
| 2 | 20 | 0.003447 | [0.002109, 0.004786] | 24 |
| 2 | 120 | 0.006869 | [0.002514, 0.011224] | 144 |
| 3 | 20 | 0.001867 | [0.000945, 0.002788] | 17 |
| 3 | 120 | 0.004129 | [0.001350, 0.006909] | 100 |
| 4 | 20 | 0.001531 | [0.000621, 0.002440] | 12 |
| 4 | 120 | 0.002792 | [0.000946, 0.004639] | 69 |

The important contrast is that, after checkpoint 20, the same-step SVD structural damage relative to simply continuing the wide checkpoint is tiny and its 95% interval includes zero at every rank/checkpoint. Thus the experiment has already separated two questions that must not be conflated:

**The contraction can be safe while the preceding wide phase is still not worth its compute.**

The first sweep's pooled post-initialization summary treated repeated checkpoints as separate observations. Those pooled intervals should therefore be regarded as descriptive only. The new break-even sweep corrects this by averaging repeated checkpoints within seed before constructing an overall confidence interval across independent seed means.

## Theory forced by the falsification

For a target rank `r`, write `K=t+H` for the same-update compact budget and

`N_r = ceil((C_wide t + C_r H)/C_r)`

for the equal-compute compact budget. Define

`discovery advantage = R_compact(K) - R_wide_then_small(t,H)`

and

`compute opportunity gain = R_compact(K) - R_compact(N_r)`.

Then exactly

`R_wide_then_small - R_compact(N_r) = compute opportunity gain - discovery advantage`.

Temporary overparameterization pays for itself if and only if its same-update discovery advantage exceeds the extra optimization progress the compact model can buy with the saved compute. A safe-pruning certificate controls neither term by itself.

## Next experiment now running

The follow-up sweep records this decomposition directly. It compares both (i) the matched SVD-initialized compact control and (ii) a naturally initialized compact-from-start control, reports same-step and equal-compute risks, verifies the break-even identity numerically, and uses seed-level aggregation for overall uncertainty.

The scientific decision after that run is sharp. If discovery advantage is nonpositive, the current DGP does not contain a useful temporary-width discovery mechanism and we should change the controlled model rather than tune the controller. If discovery advantage is positive but smaller than compute opportunity gain, the mechanism exists but is too expensive; then the problem becomes finding a cheaper onset or cheaper wide exploratory phase. Only if discovery advantage exceeds compute opportunity gain do we have evidence for a true learn-wide-then-shrink compute benefit.