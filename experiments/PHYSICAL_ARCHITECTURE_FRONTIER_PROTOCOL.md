# Physical architecture frontier: precommitted confirmatory protocol

## Why this experiment exists

The frozen 50-seed quadratic factor-network bridge established a sharp mechanism result but not an equal-compute speedup at the architecture-implied wide/compact MAC ratio `rho=4`: temporary width creates a decisively better compact state at matched update count, contraction becomes safe only after a discovery/separation phase, yet the compact-from-start model can spend the saved compute on extra SGD updates and erase the advantage.

That result leaves one precise question: **is there a physically realizable amount of temporary overparameterization for which the discovery benefit is large enough to repay its own cost?**

This experiment answers that question without changing the mechanism, loss, optimizer, contraction rule, or evaluation target. It varies the actual factor count of the wide model. Because factor count changes both compute and random-matrix aspect ratio, this is explicitly an **architecture frontier**, not a pure causal cost-ratio ablation.

The new confirmatory seeds `200--249` are disjoint from all previous confirmatory seeds. This protocol is committed before any result on those seeds is inspected. No architecture/checkpoint cell will be selected or retuned after seeing the confirmatory outcomes.

## Fixed DGP and optimization

For every architecture,

`f_W(x,z) = x' W W' z`,

with independent standard Gaussian `x,z` and teacher

`A* = U U'`,

where `U` contains the first `k=12` Euclidean basis vectors. Labels are noiseless in the primary experiment,

`y = x' A* z`.

Population evaluation remains exact:

`R(W) = 0.5 ||W W' - A*||_F^2`.

Optimization is frozen to vanilla SGD, learning rate `0.005`, batch size `256`, no momentum, float32 training, and the same pre-generated stochastic stream within each architecture/seed comparison.

## Frozen architecture frontier

Ambient dimension and final compact rank are fixed:

- `d = 24`;
- `k = r_C = 12`.

The temporary wide factor count is

`r_W in {24, 30, 36, 42, 48}`.

The leading forward MAC cost is

`C(d,r)=(2d+1)r`,

so the architecture-implied wide/compact cost ratios are exactly

`rho in {2.0, 2.5, 3.0, 3.5, 4.0}`.

For each cell, initialization is

`W_0(i,j) ~ N(0,1/r_W)`.

Thus changing `r_W` also changes the Wishart aspect ratio `gamma=d/r_W`; this is part of the physical architecture change and must not be interpreted as a pure exogenous price change. The matched compact initialization for a given wide architecture is the target-free top-12 SVD truncation of that architecture's identical wide initialization.

## Frozen timing grid

- post-contraction horizon: `H=40` updates;
- checkpoints: `20, 40, 60, 80, 100, 120`.

At every checkpoint and for every architecture, compare:

1. full wide continuation for `H` additional updates;
2. physical top-12 SVD contraction at the checkpoint followed by `H` updates;
3. matched compact-from-start for `t+H` updates;
4. matched compact-from-start for equal analytic forward compute,

   `N_eq(t,r_W) = ceil(H + rho(r_W) t)`.

The ceiling weakly favors the compact baseline.

## Primary estimands

Future structural damage:

`S(t,r) = R_wide->compact(t+H) - R_wide(t+H)`.

Same-update discovery advantage:

`D(t,r) = R_compact(t+H) - R_wide->compact(t+H)`.

Compact compute opportunity gain:

`O(t,r) = R_compact(t+H) - R_compact(N_eq(t,r))`.

Equal-compute margin:

`M(t,r) = D(t,r)-O(t,r)`

`         = R_compact(N_eq(t,r)) - R_wide->compact(t+H)`.

Positive `M` means temporary width has produced enough state-quality advantage to repay the **actual architecture-implied factorized forward MAC cost**.

## Spectral mechanism diagnostics

Without using the teacher to select the contraction, record:

- top-12 teacher-subspace overlap, for diagnosis only;
- signal floor `lambda_12`;
- lower-sector edge `lambda_13`;
- eigengap `lambda_12-lambda_13`;
- lower-sector covariance trace fraction;
- immediate population-risk jump from top-12 SVD contraction.

The contraction itself always keeps the top singular directions of `W`; it never uses teacher directions or test outcomes.

## Confirmatory inference

Use 50 independent seeds `200--249` for every architecture/checkpoint cell. The family contains `5 x 6 = 30` frozen cells.

For the decisive equal-compute margin `M`, report Bonferroni simultaneous two-sided 95% paired Student-t intervals over **all 30 cells**. A physical equal-compute regime is declared only if at least one cell has simultaneous lower confidence bound strictly greater than zero.

The same 30-cell simultaneous correction is reported for `D` and `S` so mechanism claims are not based on cherry-picked cells. Descriptive mean-optimal cells are reported separately and are not called statistically positive unless their simultaneous interval satisfies the corresponding sign criterion.

The primary headline outcomes are:

- **mechanism failure**: no cell has simultaneous `D>0`;
- **mechanism only**: some cells have simultaneous `D>0`, but no cell has simultaneous `M>0`;
- **physical equal-compute regime**: at least one cell has simultaneous `M>0`.

The architecture frontier is additionally summarized by the earliest safe checkpoint per `r_W`, where safe means the simultaneous upper bound for `S` is at or below zero.

## Falsification discipline

Seeds `200--249` cannot be used to change `d`, `k`, the rank grid, learning rate, batch size, horizon, checkpoint grid, initialization scale, contraction rule, or inference family. If all `M` intervals fail, that is the result for this architecture family.

Any subsequent attempt to improve the economics of temporary width must change a scientifically interpretable axis on a fresh seed family—for example stochastic-gradient noise/batch size, structured cost asymmetry, or an ordinary nonlinear architecture—and must receive its own protocol before those seeds are evaluated.

## Interpretation ceiling

A positive `M` cell would establish a controlled **finite-sample, physically contractible, equal-analytic-compute** regime in which temporary overparameterization is useful. It would not yet establish GPU wall-clock gains, because diagnostic overhead and hardware utilization are absent from this factor-level MAC accounting.

A completely negative frontier would be equally informative: it would show that the phase-discovery mechanism survives finite-sample training but does not repay even moderate architecture-implied width costs in this family.