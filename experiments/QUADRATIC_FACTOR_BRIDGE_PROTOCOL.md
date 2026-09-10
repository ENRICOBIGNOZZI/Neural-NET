# Quadratic factor-network bridge: precommitted confirmatory protocol

## Purpose

The exact dynamic-BBP experiment established a positive mechanism in population covariance dynamics: temporary high-rank geometry can discover a target-aligned outlier, after which target-free spectral contraction can preserve the discovered state. The previous ordinary rank-factor MLP, however, showed essentially no matched same-update discovery advantage and lost at equal compute.

This experiment is the missing bridge between those two results. It uses a literal trainable factorization whose **population gradient flow induces the same Riccati covariance dynamics as the exact phase theorem**, but trains it from finite stochastic samples and performs a physical factor-count reduction. The central question is therefore no longer whether a hand-declared cost ratio can be repaid. The wide/compact cost ratio is fixed by the actual factorized forward computation.

Development seeds `0--9` were used only to verify code, identify a numerically stable learning-rate scale, and choose this single confirmatory configuration. They are permanently excluded from confirmatory inference. The first confirmatory run uses seeds `100--149`. The configuration, checkpoints, estimands, and inference rule below are frozen before those seeds are evaluated.

## Network and exact population target

The network is

`f_W(x,z) = x' W W' z`,

with `W in R^{d x r}`. Inputs `x,z` are independent standard Gaussian vectors. The teacher is

`A* = theta U U'`,

where `U` contains the first `k` Euclidean basis vectors and `theta=1`. Labels are

`y = x' A* z`

in the primary confirmatory experiment. No additive label noise is used in the primary run; stochasticity already enters through fresh finite mini-batches. Noise robustness, if run later, will be a separately labeled secondary experiment.

For this DGP the exact population half-MSE is

`R(W) = 0.5 ||W W' - A*||_F^2`.

Thus every reported terminal risk is an analytic population quantity, not a finite test-set estimate, and no test labels can leak into contraction.

## Frozen architecture

- ambient dimension: `d=32`;
- wide factor count: `r_W=48`;
- teacher / compact rank: `k=r_C=12`;
- teacher strength: `theta=1`.

The wide initialization is `W_0(i,j) ~ N(0,1/r_W)`, hence `W_0 W_0'` is a full-rank Wishart covariance with aspect ratio `gamma=d/r_W=2/3`. This retains the random-matrix bulk/outlier geometry of the phase model.

The compact matched initialization is the **target-free rank-12 truncated SVD of the identical wide initialization**. The teacher is never used to choose retained factors.

## Training

- optimizer: vanilla SGD without momentum;
- learning rate: `0.005`;
- batch size: `256`;
- precision: float32 for SGD, float64 for spectral diagnostics;
- post-contraction horizon: `H=40` SGD updates;
- checkpoints: `20, 40, 60, 80, 100, 120, 160` wide updates.

Within a seed, wide and compact paths consume the same pre-generated stochastic sample stream whenever their step indices overlap.

## Structural action

At checkpoint `t`, compute the SVD of the current wide factor matrix and keep its largest `k=12` singular directions. This is equivalent to retaining the top-12 eigenmodes of `W_t W_t'`. The contraction rule is target-free and physically rebuilds `W` from shape `32 x 48` to `32 x 12`.

The leading forward multiply-accumulate count is

`C(d,r)=2 d r + r = (2d+1)r`.

Therefore the **actual analytic wide/compact forward-MAC ratio is exactly**

`rho = C(32,48)/C(32,12) = 4`.

This is not a declared hypothetical ratio. It follows from the architecture. It is still an analytic operation count rather than a wall-clock hardware claim; a later implementation must measure realized accelerator time and controller overhead separately.

## Matched paths

For each checkpoint `t` compare four quantities:

1. `wide`: continue the full rank-48 network for `H` more updates;
2. `wide -> compact`: truncate the checkpoint to rank 12 and continue for `H` updates;
3. `matched compact, same steps`: truncate the identical initialization to rank 12 at time zero and train for `t+H` updates;
4. `matched compact, equal compute`: train that same compact initialization for

   `N_eq(t) = ceil(H + rho t)`

   updates.

Because the factorized cost is linear in `r`, path 4 has at least as many analytic forward MACs as the wide-then-compact path; the ceiling weakly favors the compact baseline.

## Primary estimands

The **future structural damage** of contracting at `t` is

`S(t) = R_wide->compact(t+H) - R_wide(t+H)`.

Negative values mean contraction improves population risk by removing nuisance covariance; positive values mean contraction destroys useful future learning.

The **same-step discovery advantage** is

`D(t) = R_compact(t+H) - R_wide->compact(t+H)`.

Positive `D` means the wide phase has produced a better compact state than the matched compact path reaches in the same number of updates.

The **compute opportunity gain** is

`O(t) = R_compact(t+H) - R_compact(N_eq(t))`.

The decisive **equal-compute margin** is

`M(t) = D(t)-O(t) = R_compact(N_eq(t)) - R_wide->compact(t+H)`.

Positive `M` means the benefit created by the temporary wide phase is large enough to repay its actual 4x factorized forward cost.

## Spectral diagnostics

At each checkpoint record, without using them for the structural choice:

- teacher-subspace overlap of the top-12 covariance eigenspace,
  `||V_12' U||_F^2 / 12`;
- signal floor `lambda_12`;
- empirical bulk edge `lambda_13`;
- eigengap `lambda_12-lambda_13`;
- fraction of covariance trace below the retained top-12 sector;
- immediate population-risk jump caused by SVD contraction.

The mechanism prediction is a temporal ordering: early contraction should be harmful; target-subspace overlap and the signal/bulk gap should then rise; structural damage should cross into the safe region; only afterwards can an equal-compute advantage, if one exists, emerge.

## Confirmatory inference

Use 50 independent seeds, `100--149`. All checkpoint-level intervals are paired across seed. To avoid calling a favorable checkpoint significant merely because seven checkpoints were inspected, use **Bonferroni simultaneous two-sided 95% Student-t intervals across the seven frozen checkpoints** for each primary checkpoint curve.

The primary mechanism criterion is satisfied only if there is at least one checkpoint for which the simultaneous interval for `D(t)` lies strictly above zero and at least one earlier checkpoint where contraction is materially harmful followed by a later checkpoint whose simultaneous interval for `S(t)` is at or below zero.

The stronger compute criterion is satisfied only if there is at least one frozen checkpoint for which the simultaneous interval for `M(t)` lies strictly above zero. The earliest such checkpoint, if any, is reported without interpolation. The checkpoint maximizing the mean `M(t)` is descriptive unless its simultaneous interval is also positive.

## Falsification outcomes

Three outcomes are scientifically distinct and will not be collapsed:

- `D <= 0`: the finite-sample factor network fails to reproduce a temporary discovery advantage;
- `D > 0` but `M <= 0`: temporary width creates a better state, but the compact model can buy enough extra SGD progress to erase it at the true 4x factor cost;
- `M > 0`: temporary overparameterization has positive equal-compute value in a physical, finite-sample, structurally contractible network.

A negative result does not trigger checkpoint, rank, learning-rate, or seed retuning on this confirmatory sample. Any new DGP or architecture is a new experiment with a new protocol.

## What a positive result would and would not establish

A positive result would be the first direct bridge from the dynamic spectral phase theorem to an actual trainable network with finite-sample SGD and physical rank reduction. It would establish that the `discover wide -> separate -> contract` mechanism can survive exact compute accounting in this controlled architecture.

It would **not** establish a general neural-network speedup, GPU wall-clock improvement, or a deployable bulk detector. Those require, in order: an observable onset rule that does not know the teacher, a measured-overhead implementation, and replication in ordinary MLP/CNN/Transformer architectures.