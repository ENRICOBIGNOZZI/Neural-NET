# Finite-horizon structural-value audit

This experiment measures the empirical object that the master safe-contraction theorem bounds.

At a dense checkpoint `t`, two branches are created from exactly the same network state and receive the same terminal readout action and optimizer reset:

- full-width ridge-refit continuation;
- structural last-hidden-layer contraction, the same ridge refit, and continuation.

After horizon `H`, define probe structural damage

`D_probe(t,H,S) = R_probe,reduced(t+H) - R_probe,dense-refit(t+H)`.

The *current* structural damage feature is readout-matched as well:

`D_probe(t,0,S) = R_probe,reduced-refit(t) - R_probe,dense-refit(t)`.

The first development audit uses Digits and Cancer, seeds 0--3, checkpoints 80/120/160, horizon 40, and candidate retained-width fractions 0.75 and 0.875. The first checkpoint of each trajectory has no previous target-angle velocity and is excluded, leaving 32 transitions across 8 held-out trajectory groups. These configurations are permanently development-only.

## Leave-one-trajectory-out forecast of future probe damage

| State | R² | Pearson | MAE | correlation of forecast with ex-post test damage |
|---|---:|---:|---:|---:|
| time + contraction size | -0.234 | -0.455 | 0.00757 | +0.401 |
| current structural damage + time + size | 0.160 | +0.416 | 0.00672 | -0.320 |
| target-value state | **0.362** | **+0.618** | **0.00586** | **-0.492** |

The target-value state clearly contains incremental information about **future probe damage**. That is useful, but it is not the result we need.

## The important negative result: the probe is not resolving the structural effect

Across all 32 transitions, realized future probe damage and realized future test damage have Pearson correlation

`-0.581`.

The sign remains negative inside each dataset:

- Cancer: `-0.599`;
- Digits: `-0.522`.

Therefore a forecaster can become better at predicting the current probe while becoming *less* informative about the ex-post test effect. This is not evidence that the target-value theory is wrong. It means the finite-horizon structural effect being estimated here is often smaller than the sampling variation of the small probe/test split.

The scale makes this visible. On Digits the standard deviation of future probe structural damage is only about `0.0048`; on Cancer it is about `0.0117`, while individual test-damage realizations on Cancer are much noisier. Several candidate contractions even reverse sign between probe and test.

## Consequence for the algorithm

The next controller must not treat point-estimated probe damage as a reliable oracle. The empirical problem has moved one level deeper:

> **Before deciding whether a capacity sector has low future value, we must know whether its structural damage is statistically resolvable at all.**

This creates a fifth gate in addition to target value, geometry motion, structural realizability, and compute value: **finite-sample observability of the deletion effect**.

The theory is being extended with a paired structural-damage confidence bound. The correct statistic is the paired per-example loss difference between the reduced and dense continuations on an independent probe. Because the two predictors are evaluated on the same examples, a paired concentration bound can be substantially sharper than separately estimating their risks. A contraction can be certified only when an upper confidence bound on paired structural damage lies below its risk budget / compute value.

## Current interpretation

This audit rejects a tempting shortcut: “fit a good predictor of future probe damage and use it as the controller.” The target-value state improved held-out probe forecasting from `R²=0.160` to `0.362` relative to current-damage/time/size, but the probe label itself did not transfer to test in this small experiment. The next milestone is therefore not another threshold tuning exercise. It is a statistically calibrated structural-damage estimator and a larger or variance-reduced probe protocol.
