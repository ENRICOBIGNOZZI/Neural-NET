# Non-oracle controller: precommitted physical confirmation

## Scientific question

Can a neural network decide **when** overparameterization has stopped providing learning value, and **how much** structure can be removed, using only its current spectrum?

The development family on seeds `0--49` was used once to choose the controller below. This document freezes that controller before confirmatory seeds `300--349` are evaluated.

## Frozen DGP and optimizer

The model and data remain

```text
f_W(x,z) = x' W W' z,
A* = U U',
y = x' A* z,
```

with

- ambient dimension `d = 24`;
- teacher rank `12`, evaluation only;
- teacher strength `theta = 1`;
- noiseless labels;
- vanilla SGD;
- learning rate `0.005`;
- batch size `256`;
- float32 training;
- post-contraction horizon `H = 40`;
- checkpoints `20, 40, 60, 80, 100, 120`;
- wide factor counts `24, 30, 36, 42, 48`.

The teacher rank, teacher basis, population risk after future updates, and all equal-compute outcomes are forbidden inputs to the controller.

## Frozen target-free controller

At a checkpoint let

```text
lambda_1 >= ... >= lambda_d >= 0
```

be the eigenvalues of the current covariance `W W'`.

Two rank estimates are formed independently.

### Largest-gap rank

```text
k_gap = argmax_k (lambda_k - lambda_{k+1}).
```

### Two-cluster log-spectrum rank

For each split `k`, divide the log eigenvalues into top and bottom clusters and minimize total within-cluster squared dispersion. The minimizer is `k_log`.

### Phase certificate

The controller contracts at the **first** frozen checkpoint satisfying all three conditions

```text
k_gap = k_log = k_hat,
lambda_k_hat / lambda_(k_hat+1) >= 1.5,
sum_{j<=k_hat} lambda_j / sum_j lambda_j <= 0.90.
```

It then physically contracts the factorization to rank `k_hat` by top-`k_hat` SVD.

The final mass condition prevents a split that merely isolates a numerically vanishing tail from being called a meaningful phase transition.

There is no teacher-rank parameter, target-overlap threshold, future-loss probe, or ex-post checkpoint choice in this controller.

## What the development family established

The thresholds above are frozen from the completed development artifacts only. On the `5 x 50 = 250` development trajectories, the stopping rule triggered on `196` trajectories. Among those development triggers:

- under-ranking events relative to the evaluation-only teacher rank: `0`;
- triggers before the finite-horizon oracle-safe time for the selected structural action: `0`;
- positive finite-horizon structural-damage realizations at the trigger: `0`.

These are development facts, not confirmatory claims.

The rule is deliberately conservative: when it triggers, its delay relative to the evaluation-only safe time is typically one or two checkpoint intervals. False-late decisions spend compute; false-early decisions destroy learned geometry, so the asymmetry is intentional.

## Fresh confirmatory family

The confirmatory seeds are exactly

```text
300, 301, ..., 349.
```

They are disjoint from the development family and from the earlier physical architecture frontier.

No threshold, checkpoint, architecture, optimizer setting, horizon, DGP parameter, or inference rule may change after any result from seeds `300--349` is inspected.

## Evaluation-only quantities

For diagnosis, the experiment may record

- whether `k_hat` equals, exceeds, or falls below teacher rank `12`;
- teacher-subspace coverage;
- finite-horizon structural damage;
- the first checkpoint at which contraction to the **selected rank** would have had non-positive future structural damage;
- the teacher-rank oracle-safe checkpoint;
- same-update discovery advantage;
- equal-compute margin.

None may feed back into controller selection.

For onset calibration, the relevant oracle is the safe time of the **same rank selected by the controller**, not a different oracle rank.

## Physical equal-compute comparison

If the controller triggers at checkpoint `t` with selected rank `k_hat`, define

```text
C(d,r) = (2d+1) r,
rho_hat = C(d,r_W) / C(d,k_hat).
```

The matched small-from-start baseline is initialized by the target-free top-`k_hat` SVD truncation of the identical wide initialization.

It is trained to

```text
N_eq = ceil(H + rho_hat t)
```

updates, which weakly favors the small-from-start baseline.

At the controller-selected trigger define

```text
S = R_wide->small(t+H) - R_wide(t+H),
D = R_small(t+H) - R_wide->small(t+H),
O = R_small(t+H) - R_small(N_eq),
M = D - O
  = R_small(N_eq) - R_wide->small(t+H).
```

Thus `S <= 0` is structural safety and `M > 0` means temporary overparameterization repaid its own architecture-implied compute cost.

No-trigger runs are **abstentions**. The controller is not forced to compress when its certificate is absent.

## Confirmatory inference

There are five architecture-level policy cells, one per wide factor count. Checkpoint selection is performed by the frozen target-free stopping rule rather than by outcome search.

For the three inferential estimands `S`, `D`, and `M`, report two-sided 95% paired Student-t intervals with Bonferroni correction over all

```text
5 architectures x 3 estimands = 15 tests.
```

An architecture needs at least `20` triggered seeds to support an interval-based claim.

The family-level non-oracle physical regime is declared only if all of the following hold:

1. zero under-ranking events across confirmatory triggers;
2. zero controller triggers before the evaluation-only safe time of the selected structural action;
3. at least one architecture with enough triggered seeds has simultaneously
   - upper confidence bound for `S` at or below zero,
   - lower confidence bound for `D` above zero,
   - lower confidence bound for `M` above zero.

If this fails, the claim is not rescued by selecting another checkpoint or retuning the thresholds on seeds `300--349`.

## Interpretation ceiling

A pass would establish a controlled, finite-sample, target-free **spectral phase certificate** for physically removing parameters after their transient learning value has expired, together with at least one equal-analytic-compute regime.

It would still not establish GPU wall-clock gains or universality across ordinary nonlinear networks. Those require a separate precommitted stage.
