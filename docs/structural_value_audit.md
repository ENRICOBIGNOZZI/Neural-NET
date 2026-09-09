# Finite-horizon structural-value audit

This experiment measures the empirical object that the master safe-contraction theorem bounds.

At a dense checkpoint `t`, two branches are created from exactly the same network state and receive the same terminal readout action and optimizer reset:

- full-width ridge-refit continuation;
- structural last-hidden-layer contraction, the same ridge refit, and continuation.

After horizon `H`, define probe structural damage

`D_probe(t,H,S) = R_probe,reduced(t+H) - R_probe,dense-refit(t+H)`.

The controller should ultimately predict an upper confidence bound for this quantity from information available at `t`. Test labels are recorded only for ex-post transfer auditing and never enter the forecaster.

The *current* structural damage feature is readout-matched as well:

`D_probe(t,0,S) = R_probe,reduced-refit(t) - R_probe,dense-refit(t)`.

This correction matters. Comparing a contracted ridge-refit branch to the network's unsolved current readout would contaminate structural damage with the independent value of solving the readout.

The first development audit uses Digits and Cancer, seeds 0--3, checkpoints 80/120/160, horizon 40, and candidate retained-width fractions 0.75 and 0.875. These configurations are permanently development-only.

The main comparison is whether a target-value state (resolved accessibility, target-angle motion, current structural damage, effective rank, and candidate size) forecasts future probe damage under leave-one-trajectory-out validation better than:

1. time + contraction size;
2. current readout-matched structural damage + time + size.

A useful result is not merely high in-sample correlation. It must survive whole-trajectory holdout and its predictions must correlate with ex-post test structural damage, while test labels remain completely absent from the forecaster.
