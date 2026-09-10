# SVD equal-compute falsification and break-even decomposition

## 1. First equal-compute result

The first 10-seed equal-compute SVD experiment is a clean negative result for the naive claim that temporary overparameterization is automatically useful once contraction becomes safe.

The setup uses a rank-8 contractible MLP, teacher rank 2, target ranks 2/3/4, full-batch SGD, zero label noise, checkpoints 0/20/.../120, and a 40-step post-contraction horizon. The compact matched control is the target-free truncated-SVD recompression of the identical initial wide network. Its number of updates is increased to match the analytic factorized-matmul MAC cost of the wide-then-small path; the ceiling weakly favors compact-from-start.

For the primary statistic

`R_wide_then_SVD - R_compact_equal_compute`,

negative values favor temporary width and positive values favor compact-from-start. At every nonzero checkpoint for every target rank, the mean gap is positive and its checkpoint-wise 95% interval across the 10 seeds is also positive. Representative points are:

| rank | checkpoint | mean equal-compute gap | 95% interval | compact extra steps |
|---:|---:|---:|---:|---:|
| 2 | 20 | 0.003447 | [0.002109, 0.004786] | 24 |
| 2 | 120 | 0.006869 | [0.002514, 0.011224] | 144 |
| 3 | 20 | 0.001867 | [0.000945, 0.002788] | 17 |
| 3 | 120 | 0.004129 | [0.001350, 0.006909] | 100 |
| 4 | 20 | 0.001531 | [0.000621, 0.002440] | 12 |
| 4 | 120 | 0.002792 | [0.000946, 0.004639] | 69 |

After checkpoint 20, however, the same-step SVD structural damage relative to simply continuing the wide checkpoint is tiny and its 95% interval includes zero at every rank/checkpoint. Thus the experiment separates two questions that must not be conflated:

**The contraction can be safe while the preceding wide phase is still not worth its compute.**

## 2. Exact break-even decomposition

For target rank `r`, write `K=t+H` for the same-update compact budget and

`N_r = ceil((C_wide t + C_r H)/C_r)`

for the equal-compute compact budget. Define

`discovery advantage = R_compact(K) - R_wide_then_small(t,H)`

and

`compute opportunity gain = R_compact(K) - R_compact(N_r)`.

Then exactly

`R_wide_then_small - R_compact(N_r) = compute opportunity gain - discovery advantage`.

Temporary overparameterization pays for itself if and only if same-update discovery advantage exceeds the extra optimization progress the compact model can buy with the saved compute. A safe-pruning certificate controls neither term by itself.

## 3. Follow-up result: the current DGP has essentially no matched discovery advantage

The second sweep computes this decomposition directly. It uses the same 10 seeds and additionally evaluates two compact controls: a **matched compact** model obtained by target-free SVD recompression of the identical initial wide network, and a **native compact** model independently initialized at the target rank. Per-checkpoint uncertainty is computed across seeds. For the post-initialization aggregate, checkpoints are first averaged within each seed and the confidence interval is then computed across the 10 independent seed means.

The matched control gives the clean causal answer. Across post-initialization checkpoints, the discovery advantage is statistically indistinguishable from zero at every target rank, while the compute opportunity gain is positive and clearly larger:

| rank | matched discovery advantage `D` | matched compute opportunity gain `O` | break-even margin `D-O` |
|---:|---:|---:|---:|
| 2 | 0.000284 [-0.000796, 0.001365] | 0.005173 [0.002882, 0.007463] | **-0.004888 [-0.007476, -0.002300]** |
| 3 | 0.000729 [-0.000388, 0.001846] | 0.003439 [0.001845, 0.005034] | **-0.002710 [-0.004339, -0.001082]** |
| 4 | 0.000518 [-0.000121, 0.001157] | 0.002425 [0.001425, 0.003425] | **-0.001907 [-0.003079, -0.000735]** |

The exact identity is satisfied numerically to floating-point precision. On the matched comparison, the wide-then-small path loses at equal compute in every seed-averaged rank-2 and rank-3 trajectory; rank 4 has only one of ten seed-averaged trajectories with a positive break-even margin. The result is therefore not a controller-tuning failure. It says that this particular rank-factor teacher--student DGP does not generate a material same-update state advantage from temporary rank 8.

This is scientifically useful. The target rank is already sufficient for a teacher of rank 2, and the compact factorization can rotate its factors during training. Extra rank therefore behaves mainly as extra optimization cost rather than as a genuine discovery mechanism. Once contraction becomes safe, the compact model has simply bought more useful gradient steps with the same compute.

## 4. Native compact-from-start comparison

The independently initialized compact baseline is substantially noisier. Its seed-aggregated mean discovery advantage is positive for ranks 2/3/4, but all corresponding 95% intervals include zero. The equal-compute gap is also statistically unresolved:

| rank | native discovery advantage `D` | native equal-compute gap `R_wide-small - R_native` |
|---:|---:|---:|
| 2 | 0.014759 [-0.005988, 0.035506] | -0.000036 [-0.007638, 0.007566] |
| 3 | 0.017382 [-0.002644, 0.037408] | -0.004699 [-0.014163, 0.004765] |
| 4 | 0.018403 [-0.003679, 0.040485] | -0.007381 [-0.019344, 0.004581] |

This does not rescue a temporary-width claim. It mainly shows why independent random initialization is a much noisier causal comparison: it mixes the width question with initialization luck. The matched SVD control is therefore the primary diagnostic for mechanism discovery; the native control remains an important conventional baseline for later large-scale experiments.

## 5. Consequence for the research program

We should **not tune the pruning controller on this DGP to manufacture a win**. The next controlled experiment must instead instantiate a mechanism already predicted by the theory: temporary width must increase target accessibility or shorten target-discovery time before contraction. A natural candidate is a bank of rank-one exploratory directions in which the target is initially hidden among many weak/random directions, wide training allows a target-aligned outlier to emerge, and post-separation contraction keeps only the resolved direction. This is the empirical counterpart of the existing dynamic-BBP and rank-one discovery-delay results.

The next protocol should be precommitted before its first run. Its decisive statistics are the same-step discovery advantage `D`, compute opportunity gain `O`, equal-compute margin `D-O`, target overlap/accessibility through time, and the onset time at which the target-aligned mode becomes statistically distinguishable from the nuisance bulk. Only a setting with `D>0` demonstrates that temporary overparameterization has created something the compact model would not have reached equally quickly; only `D>O` establishes that the discovery benefit actually repays its compute.